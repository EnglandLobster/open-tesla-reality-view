"""Unified clip directory dataset and pre-cached feature shard dataset for Colab/GPU training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class DrivingClip:
    """Standardized multi-camera + third-person ground-truth clip representation.

    Unified target format produced by all Stage 0 data adapters:
      - ``nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`` (1,601 pre-reconstructed 3DGS scenes)
      - ``CARLA`` synthetic chase-camera collector
      - ``nuScenes`` / ``Waymo`` reconstructed via ``OmniRe`` + ``Difix3D+``
      - Procedural synthetic scene generator (for zero-download pipeline tests)
    """

    clip_id: str
    surround_rgb: torch.Tensor  # [T, K, 3, H_src, W_src] in [0, 1]
    surround_depth: torch.Tensor | None  # [T, K, 1, H_src, W_src] metric depth (meters)
    src_intrinsics: torch.Tensor  # [T, K, 3, 3]
    src_c2e: torch.Tensor  # [T, K, 4, 4]
    chase_gt_rgb: torch.Tensor  # [T, 3, H_tgt, W_tgt] in [0, 1]
    tgt_intrinsics: torch.Tensor  # [T, 3, 3]
    tgt_c2e: torch.Tensor  # [T, 4, 4]


def _load_rgb_tensor(path: Path, size_hw: tuple[int, int] | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size_hw is not None and (img.height, img.width) != size_hw:
        img = img.resize((size_hw[1], size_hw[0]), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class ClipDirectoryDataset(Dataset[DrivingClip]):
    """Reads standardized Stage 0 clip directories on disk.

    Expected clip layout::

        clips_root/<clip_id>/
            metadata.json            # camera_names, src_intrinsics, src_c2e, tgt_intrinsics, tgt_c2e
            surround/<cam_name>/000000.png ...
            chase_gt/000000.png ...
            depths/<cam_name>/000000.npy ...   (optional; if absent, DA3 predicts depth online)
    """

    def __init__(
        self,
        clips_root: str | Path,
        *,
        num_frames: int = 49,
        src_hw: tuple[int, int] | None = None,
        tgt_hw: tuple[int, int] | None = (480, 832),
    ) -> None:
        self.clips_root = Path(clips_root)
        self.num_frames = num_frames
        self.src_hw = src_hw
        self.tgt_hw = tgt_hw
        self.clip_dirs = sorted(
            p for p in self.clips_root.iterdir() if p.is_dir() and (p / "metadata.json").is_file()
        )

    def __len__(self) -> int:
        return len(self.clip_dirs)

    def __getitem__(self, idx: int) -> DrivingClip:
        clip_dir = self.clip_dirs[idx]
        meta = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
        cam_names: list[str] = meta["camera_names"]

        chase_paths = sorted((clip_dir / "chase_gt").glob("*.png"))[: self.num_frames]
        t_len = len(chase_paths)
        chase_gt = torch.stack([_load_rgb_tensor(p, self.tgt_hw) for p in chase_paths], dim=0)

        surround_frames: list[torch.Tensor] = []
        depth_frames: list[torch.Tensor] = []
        has_depths = (clip_dir / "depths").is_dir()

        for t in range(t_len):
            cam_imgs = []
            cam_depths = []
            for cam in cam_names:
                p = sorted((clip_dir / "surround" / cam).glob("*.png"))[t]
                cam_imgs.append(_load_rgb_tensor(p, self.src_hw))
                if has_depths:
                    dp = sorted((clip_dir / "depths" / cam).glob("*.npy"))[t]
                    d_arr = np.load(dp).astype(np.float32)
                    cam_depths.append(torch.from_numpy(d_arr).unsqueeze(0))
            surround_frames.append(torch.stack(cam_imgs, dim=0))
            if has_depths:
                depth_frames.append(torch.stack(cam_depths, dim=0))

        return DrivingClip(
            clip_id=clip_dir.name,
            surround_rgb=torch.stack(surround_frames, dim=0),
            surround_depth=torch.stack(depth_frames, dim=0) if has_depths else None,
            src_intrinsics=torch.tensor(meta["src_intrinsics"][:t_len], dtype=torch.float32),
            src_c2e=torch.tensor(meta["src_c2e"][:t_len], dtype=torch.float32),
            chase_gt_rgb=chase_gt,
            tgt_intrinsics=torch.tensor(meta["tgt_intrinsics"][:t_len], dtype=torch.float32),
            tgt_c2e=torch.tensor(meta["tgt_c2e"][:t_len], dtype=torch.float32),
        )


def save_cached_shard(
    out_path: str | Path,
    *,
    clean_latents: torch.Tensor,
    vace_context: torch.Tensor,
    vision_tokens: torch.Tensor,
    plucker_rays: torch.Tensor,
    clip_id: str,
) -> None:
    """Persist a single pre-cached clip shard in BF16 for high-throughput Colab training."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "clip_id": clip_id,
            "clean_latents": clean_latents.detach().cpu().to(torch.bfloat16),
            "vace_context": vace_context.detach().cpu().to(torch.bfloat16),
            "vision_tokens": vision_tokens.detach().cpu().to(torch.bfloat16),
            "plucker_rays": plucker_rays.detach().cpu().to(torch.bfloat16),
        },
        path,
    )


class CachedShardDataset(Dataset[dict[str, Any]]):
    """Loads pre-cached ``.pt`` feature shards produced by ``scripts/precache_features.py``.

    Because frozen encoders (Alpamayo2-Edge, DA3, AutoencoderKLWan) are pre-computed offline,
    Colab A100 / PRO 6000 / H100 training devotes ~100% of VRAM and FLOPs to the adapter + DiT.
    """

    def __init__(self, shards_dir: str | Path) -> None:
        self.shards_dir = Path(shards_dir)
        self.shard_paths = sorted(self.shards_dir.glob("*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No cached .pt shards found in {self.shards_dir}")

    def __len__(self) -> int:
        return len(self.shard_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return torch.load(self.shard_paths[idx], map_location="cpu", weights_only=True)
