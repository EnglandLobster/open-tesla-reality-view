#!/usr/bin/env python3
"""Predict per-camera metric depth maps for Stage 0 clips using Depth Anything 3 (DA3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Depth Anything 3 on Stage 0 clip surround images.")
    parser.add_argument("--clips-dir", required=True, help="Root directory containing clip folders.")
    parser.add_argument("--model-id", default="depth-anything/DA3-BASE")
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()

    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "Depth Anything 3 is not installed. Install from:\n"
            "  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git && pip install -e Depth-Anything-3"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model_id).to(device)
    model.eval()

    clips_root = Path(args.clips_dir)
    clip_dirs = sorted(p for p in clips_root.iterdir() if p.is_dir() and (p / "metadata.json").is_file())
    if not clip_dirs:
        raise FileNotFoundError(f"No clip directories with metadata.json found in {clips_root}")

    for clip_dir in clip_dirs:
        meta = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
        cam_names: list[str] = meta["camera_names"]
        print(f"Estimating DA3 depth for clip '{clip_dir.name}' ({len(cam_names)} cameras) ...")

        for cam in cam_names:
            src_imgs = sorted((clip_dir / "surround" / cam).glob("*.png"))
            depth_dir = clip_dir / "depths" / cam
            depth_dir.mkdir(parents=True, exist_ok=True)

            for start in range(0, len(src_imgs), args.batch_size):
                batch_paths = [str(p) for p in src_imgs[start : start + args.batch_size]]
                pred = model.inference(batch_paths)
                depths = pred.depth  # [N, H, W]
                for img_path, depth_map in zip(src_imgs[start : start + args.batch_size], depths, strict=True):
                    arr = depth_map.detach().cpu().numpy() if isinstance(depth_map, torch.Tensor) else np.asarray(depth_map)
                    np.save(depth_dir / f"{img_path.stem}.npy", arr.astype(np.float32))

        print(f"  Saved depths -> {clip_dir / 'depths'}")


if __name__ == "__main__":
    main()
