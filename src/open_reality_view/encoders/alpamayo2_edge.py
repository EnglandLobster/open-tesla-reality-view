"""Frozen vision tower wrapper for nvidia/Alpamayo2-Edge (cosmos3_edge_vision / SigLIP2).

Loads the exact 437 driving-trained SigLIP2 vision weights directly from
``nvidia/Alpamayo2-Edge`` (``00000.safetensors``, prefix ``vlm.model.visual.``) into
``transformers.Siglip2VisionModel(vision_use_head=False)`` with ``strict=True`` — requiring
zero custom NVIDIA package installs and only ~0.82 GB BF16 VRAM.
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from open_reality_view.encoders.base import SharedVisionEncoder, VisionEncoderOutput


def patchify_siglip2(images: torch.Tensor, patch_size: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert ``[N, 3, H, W]`` images into SigLIP2's exact ``(P, P, C)`` patch token layout.

    Matches ``transformers.models.siglip2.image_processing_siglip2.convert_image_to_patches``
    bit-for-bit: ``[N, C, H_p, P, W_p, P] -> permute(0, 2, 4, 3, 5, 1) -> [N, H_p*W_p, P*P*C]``.
    """
    bsz, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image spatial size ({height}, {width}) must be divisible by patch_size={patch_size}."
        )
    h_p = height // patch_size
    w_p = width // patch_size
    patches = images.reshape(bsz, channels, h_p, patch_size, w_p, patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(bsz, h_p * w_p, patch_size * patch_size * channels)
    return patches.contiguous(), (h_p, w_p)


class Alpamayo2EdgeVisionEncoder(SharedVisionEncoder):
    """Frozen driving-domain SigLIP2 vision encoder from ``nvidia/Alpamayo2-Edge``.

    Verified upstream ``config.json`` specs:
      - ``model_type``: ``cosmos3_edge_vision`` (437 tensors in ``00000.safetensors``)
      - ``hidden_size``: ``1152``, ``intermediate_size``: ``4304``
      - ``num_hidden_layers``: ``27``, ``num_attention_heads``: ``16``
      - ``patch_size``: ``16``, ``num_patches``: ``256``, ``spatial_merge_size``: ``2``
      - Image normalization: ``mean=[0.5, 0.5, 0.5]``, ``std=[0.5, 0.5, 0.5]`` (pixels in ``[-1, 1]``)
    """

    WEIGHT_PREFIX = "vlm.model.visual."

    def __init__(
        self,
        vision_tower: nn.Module,
        *,
        hidden_size: int = 1152,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        is_siglip2_naflex: bool = True,
        target_hw: tuple[int, int] | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self._hidden_size = hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.is_siglip2_naflex = is_siglip2_naflex
        self.target_hw = target_hw

        if freeze:
            self.requires_grad_(False)
            self.eval()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def patch_stride(self) -> int:
        return self.patch_size

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "nvidia/Alpamayo2-Edge",
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        target_hw: tuple[int, int] | None = (224, 384),
        freeze: bool = True,
        cache_dir: str | None = None,
    ) -> Alpamayo2EdgeVisionEncoder:
        """Download and strictly load ONLY the 437 vision weights from ``nvidia/Alpamayo2-Edge``."""
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import Siglip2VisionConfig, Siglip2VisionModel

        cfg_path = hf_hub_download(model_id, "config.json", cache_dir=cache_dir)
        index_path = hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=cache_dir)

        with open(cfg_path, encoding="utf-8") as f:
            raw_cfg = json.load(f)
        vis_cfg = raw_cfg["vlm_config"]["vision_config"]

        with open(index_path, encoding="utf-8") as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]

        vision_shards = sorted(
            {shard for key, shard in weight_map.items() if key.startswith(cls.WEIGHT_PREFIX)}
        )
        if not vision_shards:
            raise KeyError(
                f"No keys with prefix '{cls.WEIGHT_PREFIX}' found in {model_id} weight_map."
            )

        siglip_cfg = Siglip2VisionConfig(
            hidden_size=vis_cfg.get("hidden_size", 1152),
            intermediate_size=vis_cfg.get("intermediate_size", 4304),
            num_hidden_layers=vis_cfg.get("num_hidden_layers", 27),
            num_attention_heads=vis_cfg.get("num_attention_heads", 16),
            num_channels=vis_cfg.get("num_channels", 3),
            patch_size=vis_cfg.get("patch_size", 16),
            num_patches=vis_cfg.get("num_patches", 256),
            layer_norm_eps=vis_cfg.get("layer_norm_eps", 1e-6),
            hidden_act=vis_cfg.get("hidden_act", "gelu_pytorch_tanh"),
            vision_use_head=False,
        )
        vision_tower = Siglip2VisionModel(siglip_cfg)

        state_dict: dict[str, torch.Tensor] = {}
        for shard_name in vision_shards:
            shard_path = hf_hub_download(model_id, shard_name, cache_dir=cache_dir)
            with safe_open(shard_path, framework="pt", device="cpu") as sf:
                for key in sf.keys():  # noqa: SIM118
                    if key.startswith(cls.WEIGHT_PREFIX):
                        short_key = key[len(cls.WEIGHT_PREFIX) :]
                        state_dict[short_key] = sf.get_tensor(key)

        vision_tower.load_state_dict(state_dict, strict=True)
        vision_tower = vision_tower.to(device=device, dtype=dtype)

        return cls(
            vision_tower=vision_tower,
            hidden_size=siglip_cfg.hidden_size,
            patch_size=siglip_cfg.patch_size,
            spatial_merge_size=vis_cfg.get("spatial_merge_size", 2),
            is_siglip2_naflex=True,
            target_hw=target_hw,
            freeze=freeze,
        )

    def _run_tower(self, flat_images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Run vision tower on ``[N, 3, H, W]`` (assumed in ``[0, 1]`` or ``[-1, 1]``)."""
        # Normalize [0, 1] inputs to [-1, 1] matching Alpamayo2-Edge mean=0.5, std=0.5.
        if flat_images.min() >= 0.0 and flat_images.max() <= 1.0 + 1e-3:
            flat_images = flat_images * 2.0 - 1.0

        if self.target_hw is not None and tuple(flat_images.shape[-2:]) != self.target_hw:
            flat_images = F.interpolate(
                flat_images,
                size=self.target_hw,
                mode="bilinear",
                align_corners=False,
            )

        tower_dtype = next(self.vision_tower.parameters()).dtype
        flat_images = flat_images.to(dtype=tower_dtype)

        if self.is_siglip2_naflex:
            patches, (h_p, w_p) = patchify_siglip2(flat_images, patch_size=self.patch_size)
            num_imgs, num_patches, _ = patches.shape
            attn_mask = torch.ones((num_imgs, num_patches), device=patches.device, dtype=torch.long)
            spatial_shapes = torch.tensor(
                [[h_p, w_p]], device=patches.device, dtype=torch.long
            ).expand(num_imgs, 2)
            out: Any = self.vision_tower(
                pixel_values=patches,
                pixel_attention_mask=attn_mask,
                spatial_shapes=spatial_shapes,
            )
            tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            return tokens, (h_p, w_p)

        out = self.vision_tower(pixel_values=flat_images)
        tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h_p = flat_images.shape[-2] // self.patch_stride
        w_p = flat_images.shape[-1] // self.patch_stride
        expected = h_p * w_p
        if tokens.shape[1] == expected + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != expected:
            side = math.isqrt(tokens.shape[1])
            if side * side == tokens.shape[1]:
                h_p, w_p = side, side
        return tokens, (h_p, w_p)

    @torch.no_grad()
    def encode_multiview(
        self,
        images: torch.Tensor,
        *,
        micro_batch_size: int = 24,
    ) -> VisionEncoderOutput:
        """Encode ``[B, T, K, 3, H, W]`` surround frames into ``[B, T, K, H_p, W_p, D]``."""
        bsz, num_frames, num_cams, channels, height, width = images.shape
        flat = images.reshape(bsz * num_frames * num_cams, channels, height, width)

        chunks: list[torch.Tensor] = []
        grid_hw: tuple[int, int] = (0, 0)
        for start in range(0, flat.shape[0], micro_batch_size):
            tok, grid_hw = self._run_tower(flat[start : start + micro_batch_size])
            chunks.append(tok)

        tokens = torch.cat(chunks, dim=0)
        h_p, w_p = grid_hw
        tokens = tokens.reshape(bsz, num_frames, num_cams, h_p, w_p, self.hidden_size)
        return VisionEncoderOutput(
            tokens=tokens,
            grid_hw=(h_p, w_p),
            hidden_size=self.hidden_size,
        )
