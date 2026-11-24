"""Frozen vision tower wrapper for nvidia/Alpamayo2-Edge (cosmos3_edge_vision / SigLIP2)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from open_reality_view.encoders.base import SharedVisionEncoder, VisionEncoderOutput


class Alpamayo2EdgeVisionEncoder(SharedVisionEncoder):
    """Wraps the SigLIP2 vision tower inside ``nvidia/Alpamayo2-Edge``.

    Verified upstream ``config.json`` specs:
      - ``model_type``: ``cosmos3_edge_vision``
      - ``hidden_size``: ``1152``
      - ``num_hidden_layers``: ``27``, ``num_attention_heads``: ``16``
      - ``patch_size``: ``16``, ``spatial_merge_size``: ``2``
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        *,
        hidden_size: int = 1152,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        full_vla_model: nn.Module | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self._hidden_size = hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.full_vla_model = full_vla_model

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
        enable_policy_branch: bool = False,
        freeze: bool = True,
        trust_remote_code: bool = True,
    ) -> Alpamayo2EdgeVisionEncoder:
        """Load the frozen vision tower from ``nvidia/Alpamayo2-Edge``.

        When ``enable_policy_branch=False`` (default), only the vision tower (~0.41B, ~0.82 GB BF16)
        is kept on GPU and the LLM/action expert are discarded to maximize VRAM headroom.
        """
        from transformers import AutoModel

        full_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        vlm = getattr(full_model, "vlm", full_model)
        vision_tower = (
            getattr(vlm, "vision_tower", None)
            or getattr(vlm, "visual", None)
            or getattr(vlm, "vision_model", None)
        )
        if vision_tower is None:
            raise AttributeError(
                f"Could not locate vision tower attribute on loaded model '{model_id}'. "
                f"Inspect model attributes: {list(vlm._modules.keys())}"
            )

        vision_cfg = getattr(getattr(full_model.config, "vlm_config", full_model.config), "vision_config", None)
        hidden_size = getattr(vision_cfg, "hidden_size", 1152)
        patch_size = getattr(vision_cfg, "patch_size", 16)
        spatial_merge_size = getattr(vision_cfg, "spatial_merge_size", 2)

        kept_vla = full_model.to(device=device, dtype=dtype) if enable_policy_branch else None
        vision_tower = vision_tower.to(device=device, dtype=dtype)

        return cls(
            vision_tower=vision_tower,
            hidden_size=hidden_size,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            full_vla_model=kept_vla,
            freeze=freeze,
        )

    def _extract_tokens(self, flat_images: torch.Tensor) -> torch.Tensor:
        """Run underlying vision tower and return ``[N, L, D]`` patch tokens."""
        out: Any = self.vision_tower(pixel_values=flat_images)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, (tuple, list)):
            return out[0]
        raise TypeError(f"Unsupported vision tower output type: {type(out)}")

    @torch.no_grad()
    def encode_multiview(self, images: torch.Tensor) -> VisionEncoderOutput:
        """Encode ``[B, T, K, 3, H, W]`` into ``[B, T, K, H_p, W_p, D]``."""
        bsz, num_frames, num_cams, channels, height, width = images.shape
        flat = images.reshape(bsz * num_frames * num_cams, channels, height, width)
        tokens = self._extract_tokens(flat)  # [N, L, D]

        h_p = height // self.patch_stride
        w_p = width // self.patch_stride
        expected = h_p * w_p
        num_tokens = tokens.shape[1]

        if num_tokens == expected + 1:
            # Strip leading CLS token if present.
            tokens = tokens[:, 1:, :]
        elif num_tokens != expected:
            # Infer square/aspect grid if vision tower internally resized or merged patches.
            side = math.isqrt(num_tokens)
            if side * side == num_tokens:
                h_p, w_p = side, side
            else:
                merged_h = h_p // self.spatial_merge_size
                merged_w = w_p // self.spatial_merge_size
                if merged_h * merged_w == num_tokens:
                    h_p, w_p = merged_h, merged_w
                else:
                    raise ValueError(
                        f"Vision tower returned {num_tokens} tokens, incompatible with "
                        f"({h_p}x{w_p}={expected}) or merged ({merged_h}x{merged_w})."
                    )

        tokens = tokens.reshape(bsz, num_frames, num_cams, h_p, w_p, self.hidden_size)
        return VisionEncoderOutput(
            tokens=tokens,
            grid_hw=(h_p, w_p),
            hidden_size=self.hidden_size,
        )
