"""End-to-end v1 Quality-First Flow-Matching inference pipeline (Alpamayo2-Edge + Wan2.1-VACE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from open_reality_view.encoders.base import SharedVisionEncoder
from open_reality_view.geometry.ego_car import EgoVehicleCompositor
from open_reality_view.geometry.plucker import compute_plucker_rays
from open_reality_view.geometry.projector import MultiViewDepthProjector
from open_reality_view.models.reality_view_vace import (
    RealityViewVACEModel,
    decode_wan_latents,
    encode_vace_control,
)


@dataclass
class RealityViewOutput:
    """Outputs of a Reality View generation pass."""

    generated_rgb: torch.Tensor  # [B, 3, T, H, W] in [0, 1]
    coarse_rgb: torch.Tensor  # [B, 3, T, H, W] in [0, 1]
    hole_mask: torch.Tensor  # [B, 1, T, H, W] in {0, 1}


class RealityViewVACEPipeline:
    """Runs multi-camera 3D projection + frozen Alpamayo2-Edge + Wan2.1-VACE denoising."""

    def __init__(
        self,
        *,
        vision_encoder: SharedVisionEncoder,
        model: RealityViewVACEModel,
        vae: Any,
        scheduler: Any,
        projector: MultiViewDepthProjector | None = None,
        ego_compositor: EgoVehicleCompositor | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.vision_encoder = vision_encoder
        self.model = model
        self.vae = vae
        self.scheduler = scheduler
        self.projector = projector if projector is not None else MultiViewDepthProjector()
        self.ego_compositor = ego_compositor if ego_compositor is not None else EgoVehicleCompositor()
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.no_grad()
    def project_clip(
        self,
        surround_rgb: torch.Tensor,
        surround_depth: torch.Tensor,
        src_intrinsics: torch.Tensor,
        src_c2e: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_c2e: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a sequence ``[B, T, K, ...]`` into coarse target video + blind-spot hole mask."""
        _bsz, num_frames = surround_rgb.shape[:2]
        coarse_frames: list[torch.Tensor] = []
        mask_frames: list[torch.Tensor] = []

        for t in range(num_frames):
            proj = self.projector(
                images=surround_rgb[:, t],
                depths=surround_depth[:, t],
                src_intrinsics=src_intrinsics[:, t],
                src_c2e=src_c2e[:, t],
                tgt_intrinsics=tgt_intrinsics[:, t],
                tgt_c2e=tgt_c2e[:, t],
                target_hw=target_hw,
            )
            proj = self.ego_compositor(proj, tgt_intrinsics[:, t], tgt_c2e[:, t])
            coarse_frames.append(proj.coarse_rgb)
            mask_frames.append(proj.hole_mask)

        coarse_video = torch.stack(coarse_frames, dim=2)  # [B, 3, T, H, W]
        hole_mask = torch.stack(mask_frames, dim=2)  # [B, 1, T, H, W]
        return coarse_video, hole_mask

    @torch.no_grad()
    def __call__(
        self,
        surround_rgb: torch.Tensor,
        surround_depth: torch.Tensor,
        src_intrinsics: torch.Tensor,
        src_c2e: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_c2e: torch.Tensor,
        *,
        target_hw: tuple[int, int] = (480, 832),
        num_inference_steps: int = 25,
        temporal_stride: int = 4,
        spatial_stride: int = 8,
        generator: torch.Generator | None = None,
    ) -> RealityViewOutput:
        """Synthesize a photorealistic third-person Reality View clip from surround cameras."""
        surround_rgb = surround_rgb.to(device=self.device, dtype=torch.float32)
        surround_depth = surround_depth.to(device=self.device, dtype=torch.float32)
        src_intrinsics = src_intrinsics.to(device=self.device, dtype=torch.float32)
        src_c2e = src_c2e.to(device=self.device, dtype=torch.float32)
        tgt_intrinsics = tgt_intrinsics.to(device=self.device, dtype=torch.float32)
        tgt_c2e = tgt_c2e.to(device=self.device, dtype=torch.float32)

        bsz, num_frames, _num_cams, _, h_src, w_src = surround_rgb.shape

        # 1. Branch A: Pixel-aligned 3D projection + 3D Ego-Car -> 96-ch VACE control latent
        coarse_01, hole_mask = self.project_clip(
            surround_rgb,
            surround_depth,
            src_intrinsics,
            src_c2e,
            tgt_intrinsics,
            tgt_c2e,
            target_hw=target_hw,
        )
        coarse_m11 = coarse_01 * 2.0 - 1.0
        vace_context = encode_vace_control(
            self.vae,
            coarse_m11,
            hole_mask,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
        ).to(dtype=self.dtype)

        # 2. Branch B: Frozen Alpamayo2-Edge vision tokens (subsampled at Wan latent timestamps)
        key_indices = list(range(0, num_frames, temporal_stride))
        enc_out = self.vision_encoder.encode_multiview(surround_rgb[:, key_indices])
        h_p, w_p = enc_out.grid_hw
        tgt_e2c = torch.linalg.inv(tgt_c2e[:, key_indices])[:, :, None]  # [B, T_lat, 1, 4, 4]
        plucker = compute_plucker_rays(
            intrinsics=src_intrinsics[:, key_indices],
            c2w=src_c2e[:, key_indices],
            grid_height=h_p,
            grid_width=w_p,
            orig_height=h_src,
            orig_width=w_src,
            target_w2c=tgt_e2c,
        ).to(dtype=enc_out.tokens.dtype)

        encoder_hidden_states = self.model.context_adapter(
            vision_tokens=enc_out.tokens,
            plucker_rays=plucker,
        ).to(dtype=self.dtype)

        # 3. Flow-Matching denoising loop
        f_lat = vace_context.shape[2]
        h_lat, w_lat = vace_context.shape[3], vace_context.shape[4]
        latents = torch.randn(
            (bsz, 16, f_lat, h_lat, w_lat),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        for timestep in self.scheduler.timesteps:
            t_batch = timestep.expand(bsz).to(device=self.device, dtype=self.dtype)
            noise_pred = self.model.forward(
                noisy_latents=latents,
                timesteps=t_batch,
                vace_context=vace_context,
                vision_tokens=enc_out.tokens,
                encoder_hidden_states=encoder_hidden_states,
            )
            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        decoded_m11 = decode_wan_latents(self.vae, latents)
        generated_01 = ((decoded_m11.float() + 1.0) * 0.5).clamp(0.0, 1.0)

        return RealityViewOutput(
            generated_rgb=generated_01,
            coarse_rgb=coarse_01,
            hole_mask=hole_mask,
        )
