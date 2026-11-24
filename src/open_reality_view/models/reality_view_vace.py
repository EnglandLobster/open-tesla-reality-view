"""Wan2.1-VACE-1.3B Reality View generator wrapper and Flow-Matching training objective."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from open_reality_view.models.context_adapter import AlpamayoContextAdapter


def pack_vace_mask(
    hole_mask: torch.Tensor,
    *,
    spatial_stride: int = 8,
    temporal_stride: int = 4,
) -> torch.Tensor:
    """Pack pixel-space binary/soft mask into Wan2.1-VACE's 64-channel latent mask layout.

    Wan2.1 causal 3D VAE maps ``T = 1 + 4 * k`` pixel frames to ``F_lat = 1 + k`` latent frames
    and downsamples spatial dimensions by ``8x8`` (pixel-unshuffled into ``64`` channels).

    Args:
        hole_mask: ``[B, 1, T, H, W]`` where ``1.0`` denotes regions to generate/inpaint.
        spatial_stride: Spatial VAE downsampling ratio (default ``8`` -> ``64`` channels).
        temporal_stride: Temporal VAE downsampling ratio (default ``4``).

    Returns:
        Packed VACE mask ``[B, spatial_stride**2, F_lat, H // spatial_stride, W // spatial_stride]``.
    """
    bsz, channels, num_frames, height, width = hole_mask.shape
    if channels != 1:
        raise ValueError(f"Expected single-channel mask [B, 1, T, H, W], got {channels} channels.")
    if height % spatial_stride != 0 or width % spatial_stride != 0:
        raise ValueError(
            f"Spatial dims ({height}, {width}) must be divisible by spatial_stride={spatial_stride}."
        )
    if (num_frames - 1) % temporal_stride != 0:
        raise ValueError(
            f"Wan causal 3D-VAE requires T = 1 + {temporal_stride}*k frames, got T={num_frames}."
        )

    # First frame is its own latent chunk; remaining frames group in blocks of temporal_stride.
    first = hole_mask[:, :, :1]  # [B, 1, 1, H, W]
    if num_frames > 1:
        rest = hole_mask[:, :, 1:].reshape(
            bsz, 1, (num_frames - 1) // temporal_stride, temporal_stride, height, width
        )
        rest = rest.mean(dim=3)  # [B, 1, k, H, W]
        temporal_down = torch.cat([first, rest], dim=2)  # [B, 1, F_lat, H, W]
    else:
        temporal_down = first

    f_lat = temporal_down.shape[2]
    h_lat = height // spatial_stride
    w_lat = width // spatial_stride

    # Spatial pixel-unshuffle: [B, 1, F_lat, H_lat, s, W_lat, s] -> [B, s*s, F_lat, H_lat, W_lat]
    unshuffled = temporal_down.reshape(
        bsz, 1, f_lat, h_lat, spatial_stride, w_lat, spatial_stride
    )
    unshuffled = unshuffled.permute(0, 1, 4, 6, 2, 3, 5).reshape(
        bsz, spatial_stride * spatial_stride, f_lat, h_lat, w_lat
    )
    return unshuffled.contiguous()


def pack_vace_control_tensor(
    inactive_latents: torch.Tensor,
    reactive_latents: torch.Tensor,
    packed_mask: torch.Tensor,
) -> torch.Tensor:
    """Assemble the 96-channel VACE control latent ``[B, 96, F_lat, H_lat, W_lat]``.

    - ``inactive_latents`` (16 ch): VAE-encoded observed pixels ``coarse_rgb * (1 - hole_mask)``
    - ``reactive_latents`` (16 ch): VAE-encoded blind-spot / synthesis pixels ``coarse_rgb * hole_mask``
    - ``packed_mask`` (64 ch): Pixel-unshuffled hole mask from :func:`pack_vace_mask`
    """
    return torch.cat([inactive_latents, reactive_latents, packed_mask], dim=1)


class RealityViewVACEModel(nn.Module):
    """End-to-end trainable Reality View model: ``AlpamayoContextAdapter`` + ``Wan2.1-VACE-1.3B``."""

    def __init__(
        self,
        transformer: nn.Module,
        context_adapter: AlpamayoContextAdapter,
        *,
        flow_shift: float = 3.0,
        vace_context_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.context_adapter = context_adapter
        self.flow_shift = flow_shift
        self.vace_context_scale = vace_context_scale

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        context_adapter: AlpamayoContextAdapter | None = None,
        *,
        dtype: torch.dtype = torch.bfloat16,
        flow_shift: float = 3.0,
        vace_context_scale: float = 1.0,
        gradient_checkpointing: bool = True,
    ) -> RealityViewVACEModel:
        """Load ``WanVACETransformer3DModel`` without the UMT5-XXL text encoder."""
        from diffusers import WanVACETransformer3DModel

        transformer = WanVACETransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        if gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()

        adapter = context_adapter if context_adapter is not None else AlpamayoContextAdapter()
        return cls(
            transformer=transformer,
            context_adapter=adapter,
            flow_shift=flow_shift,
            vace_context_scale=vace_context_scale,
        )

    def freeze_backbone_for_stage1a(self) -> None:
        """Stage 1A warmup: freeze Wan2.1-VACE DiT and train only the AlpamayoContextAdapter."""
        self.transformer.requires_grad_(False)
        self.context_adapter.requires_grad_(True)

    def configure_stage1b_lora(
        self,
        rank: int = 64,
        alpha: int = 64,
        train_vace_blocks: bool = True,
    ) -> None:
        """Stage 1B fine-tuning: attach LoRA to Wan2.1 attention layers and unfreeze VACE blocks."""
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.05,
        )
        self.transformer = get_peft_model(self.transformer, lora_cfg)
        self.context_adapter.requires_grad_(True)

        if train_vace_blocks:
            for name, param in self.transformer.named_parameters():
                if "vace" in name.lower():
                    param.requires_grad_(True)

    def sample_flow_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample shifted logit-normal flow-matching timesteps ``t in (0, 1)``."""
        u = torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32))
        if self.flow_shift != 1.0:
            u = (self.flow_shift * u) / (1.0 + (self.flow_shift - 1.0) * u)
        return u.clamp(1e-4, 1.0 - 1e-4)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        vace_context: torch.Tensor,
        vision_tokens: torch.Tensor,
        plucker_rays: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict Flow-Matching velocity ``v_theta(x_t, t, c)``."""
        encoder_hidden_states = self.context_adapter(
            vision_tokens=vision_tokens,
            plucker_rays=plucker_rays,
            camera_ids=camera_ids,
        ).to(dtype=noisy_latents.dtype)

        out: Any = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            control_hidden_states=vace_context,
            control_hidden_states_scale=self.vace_context_scale,
            return_dict=True,
        )
        return out.sample if hasattr(out, "sample") else out[0]

    def compute_flow_matching_loss(
        self,
        clean_latents: torch.Tensor,
        vace_context: torch.Tensor,
        vision_tokens: torch.Tensor,
        plucker_rays: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Rectified Flow / Flow-Matching velocity loss used in v1 Teacher training.

        Path: ``x_t = (1 - t) * x_0 + t * eps``, target velocity: ``v* = eps - x_0``.
        """
        bsz = clean_latents.shape[0]
        device = clean_latents.device
        dtype = clean_latents.dtype

        t = self.sample_flow_timesteps(bsz, device=device)  # [B] in (0, 1)
        noise = torch.randn_like(clean_latents)

        t_view = t.view(bsz, 1, 1, 1, 1).to(dtype=dtype)
        noisy_latents = (1.0 - t_view) * clean_latents + t_view * noise
        target_velocity = noise - clean_latents

        # Diffusers Wan transformer expects timesteps scaled to [0, 1000].
        pred_velocity = self.forward(
            noisy_latents=noisy_latents,
            timesteps=(t * 1000.0).to(dtype=dtype),
            vace_context=vace_context,
            vision_tokens=vision_tokens,
            plucker_rays=plucker_rays,
            camera_ids=camera_ids,
        )
        loss = F.mse_loss(pred_velocity.float(), target_velocity.float())
        return {"loss": loss, "timestep_mean": t.mean().detach()}
