"""Fast tensor-contract and gradient-flow tests (no multi-GB weight downloads)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from open_reality_view.config import load_config
from open_reality_view.encoders import Alpamayo2EdgeVisionEncoder
from open_reality_view.geometry import (
    MultiViewDepthProjector,
    build_chase_camera_pose,
    build_orbit_chase_pose,
    compute_plucker_rays,
)
from open_reality_view.models import (
    AlpamayoContextAdapter,
    RealityViewVACEModel,
    pack_vace_control_tensor,
    pack_vace_mask,
)


def test_config_dimensions_match_verified_upstream_specs() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/v1_edge_wan21.yaml")
    assert cfg.encoder.vision_hidden_size == 1152
    assert cfg.context_adapter.adapter_dim == 1152
    assert cfg.context_adapter.output_dim == 4096
    assert cfg.generator.vace_in_channels == 96
    assert (cfg.data.clip_num_frames - 1) % cfg.generator.vae_temporal_stride == 0


def test_chase_and_orbit_camera_poses_are_valid_se3() -> None:
    for pose in [build_chase_camera_pose(), build_orbit_chase_pose(azimuth_deg=35.0)]:
        rot = pose[:3, :3]
        assert torch.allclose(rot.T @ rot, torch.eye(3), atol=1e-5)
        assert torch.isclose(torch.linalg.det(rot), torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(pose[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_plucker_rays_unit_direction_and_orthogonal_moment() -> None:
    bsz, num_cams = 2, 6
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(bsz, num_cams, 1, 1)
    intrinsics[..., 0, 0] = 320.0
    intrinsics[..., 1, 1] = 320.0
    intrinsics[..., 0, 2] = 160.0
    intrinsics[..., 1, 2] = 120.0

    c2w = torch.eye(4).view(1, 1, 4, 4).repeat(bsz, num_cams, 1, 1)
    c2w[..., :3, 3] = torch.randn(bsz, num_cams, 3)
    target_w2c = torch.linalg.inv(build_chase_camera_pose()).view(1, 1, 4, 4)

    rays = compute_plucker_rays(
        intrinsics,
        c2w,
        grid_height=12,
        grid_width=16,
        orig_height=240,
        orig_width=320,
        target_w2c=target_w2c,
    )
    assert rays.shape == (bsz, num_cams, 6, 12, 16)
    moments = rays[:, :, :3]
    dirs = rays[:, :, 3:]
    assert torch.allclose(dirs.norm(dim=2), torch.ones(bsz, num_cams, 12, 16), atol=1e-5)
    assert torch.allclose((moments * dirs).sum(dim=2), torch.zeros(bsz, num_cams, 12, 16), atol=1e-4)


def test_multiview_depth_projector_zbuffer_occlusion_and_holes() -> None:
    # Two identical source cameras observing a near red plane (z=5m) and a far blue plane (z=20m)
    # projected directly into the same target camera pose: near red plane must win everywhere.
    h, w = 32, 32
    k = torch.tensor([[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]])  # [1, 3, 3]
    src_k = k[:, None].repeat(1, 2, 1, 1)
    pose = build_chase_camera_pose(offset_xyz=(0.0, 0.0, 1.5), pitch_deg=0.0)[None]  # [1, 4, 4]
    src_c2e = pose[:, None].repeat(1, 2, 1, 1)

    images = torch.zeros(1, 2, 3, h, w)
    images[:, 0, 0] = 1.0  # cam 0 = red
    images[:, 1, 2] = 1.0  # cam 1 = blue

    depths = torch.zeros(1, 2, 1, h, w)
    depths[:, 0] = 5.0   # near
    depths[:, 1] = 20.0  # far
    # Mask top-left corner invalid in both cameras to verify hole_mask == 1 there.
    depths[:, :, :, :4, :4] = 0.0

    projector = MultiViewDepthProjector(splat_radius_px=0)
    res = projector(images, depths, src_k, src_c2e, k, pose, target_hw=(h, w))

    assert res.coarse_rgb.shape == (1, 3, h, w)
    assert res.hole_mask.shape == (1, 1, h, w)
    assert res.target_depth.shape == (1, 1, h, w)
    # Center pixel must be red (depth 5.0), not blue (depth 20.0)
    assert res.coarse_rgb[0, 0, 16, 16] == pytest.approx(1.0)
    assert res.coarse_rgb[0, 2, 16, 16] == pytest.approx(0.0)
    assert res.target_depth[0, 0, 16, 16] == pytest.approx(5.0, abs=1e-3)
    assert res.hole_mask[0, 0, 16, 16] == pytest.approx(0.0)
    assert res.hole_mask[0, 0, 1, 1] == pytest.approx(1.0)


class _DummyVisionTower(nn.Module):
    def __init__(self, hidden_size: int = 1152, patch_size: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        feats = self.conv(pixel_values).flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=feats)


class _MockWanVACETransformer(nn.Module):
    """Lightweight stub asserting exact Wan2.1-VACE-1.3B tensor contracts."""

    def __init__(self) -> None:
        super().__init__()
        self.latent_proj = nn.Conv3d(16 + 96, 16, kernel_size=1)
        self.ctx_proj = nn.Linear(4096, 16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        control_hidden_states_scale: float = 1.0,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        assert hidden_states.shape[1] == 16
        assert control_hidden_states.shape[1] == 96
        assert encoder_hidden_states.shape[-1] == 4096
        assert timestep.shape[0] == hidden_states.shape[0]

        ctx_bias = self.ctx_proj(encoder_hidden_states.mean(dim=1))[:, :, None, None, None]
        fused = torch.cat([hidden_states, control_hidden_states * control_hidden_states_scale], dim=1)
        out = self.latent_proj(fused) + ctx_bias
        return SimpleNamespace(sample=out)


def test_end_to_end_vace_flow_matching_step_and_gradients() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bsz, num_frames_lat, num_cams = 1, 3, 6  # T_pixel = 1 + 4*(3-1) = 9
    t_pixel, h_tgt, w_tgt = 9, 64, 64
    h_lat, w_lat = h_tgt // 8, w_tgt // 8

    # 1. Frozen Alpamayo2-Edge Vision Encoder contract (D=1152, patch=16)
    encoder = Alpamayo2EdgeVisionEncoder(
        vision_tower=_DummyVisionTower(hidden_size=1152, patch_size=16),
        hidden_size=1152,
        patch_size=16,
        freeze=True,
    ).to(device)
    surround_imgs = torch.randn(bsz, num_frames_lat, num_cams, 3, 64, 64, device=device)
    enc_out = encoder(surround_imgs)
    assert enc_out.tokens.shape == (bsz, num_frames_lat, num_cams, 4, 4, 1152)

    # 2. Pack VACE 96-channel control latent (16 inactive + 16 reactive + 64 spatial mask)
    hole_mask = torch.randint(0, 2, (bsz, 1, t_pixel, h_tgt, w_tgt), device=device).float()
    packed_mask = pack_vace_mask(hole_mask, spatial_stride=8, temporal_stride=4)
    assert packed_mask.shape == (bsz, 64, num_frames_lat, h_lat, w_lat)

    inactive_lat = torch.randn(bsz, 16, num_frames_lat, h_lat, w_lat, device=device)
    reactive_lat = torch.randn(bsz, 16, num_frames_lat, h_lat, w_lat, device=device)
    vace_ctrl = pack_vace_control_tensor(inactive_lat, reactive_lat, packed_mask)
    assert vace_ctrl.shape == (bsz, 96, num_frames_lat, h_lat, w_lat)

    # 3. Context Adapter (1152 -> 4096) + WanVACE Flow-Matching loss + backward pass
    adapter = AlpamayoContextAdapter(
        in_dim=1152,
        adapter_dim=128,  # compact for unit test speed; production config uses 1152
        out_dim=4096,
        num_latents=32,
        num_layers=2,
        num_heads=4,
    ).to(device)
    model = RealityViewVACEModel(
        transformer=_MockWanVACETransformer().to(device),
        context_adapter=adapter,
    )
    model.freeze_backbone_for_stage1a()

    plucker = torch.randn(bsz, num_frames_lat, num_cams, 6, 4, 4, device=device)
    clean_latents = torch.randn(bsz, 16, num_frames_lat, h_lat, w_lat, device=device)

    stats = model.compute_flow_matching_loss(
        clean_latents=clean_latents,
        vace_context=vace_ctrl,
        vision_tokens=enc_out.tokens,
        plucker_rays=plucker,
    )
    assert torch.isfinite(stats["loss"])
    stats["loss"].backward()

    # Stage 1A invariant: adapter receives gradients, frozen backbone does not.
    assert model.context_adapter.out_proj.weight.grad is not None
    assert model.transformer.latent_proj.weight.grad is None
