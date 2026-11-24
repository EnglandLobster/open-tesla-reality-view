"""Fast tensor-contract, geometry, SigLIP2, ego-car, and Flow-Matching tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Siglip2VisionConfig, Siglip2VisionModel

from open_reality_view.config import load_config
from open_reality_view.data import (
    CachedShardDataset,
    generate_synthetic_driving_clip,
    save_cached_shard,
)
from open_reality_view.encoders import Alpamayo2EdgeVisionEncoder, patchify_siglip2
from open_reality_view.geometry import (
    EgoVehicleCompositor,
    MultiViewDepthProjector,
    ProjectionResult,
    Temporal3DCacheProjector,
    build_chase_camera_pose,
    build_orbit_chase_pose,
    composite_trajectory_ribbon,
    compute_plucker_rays,
    load_glb_ego_mesh,
)
from open_reality_view.models import (
    AlpamayoContextAdapter,
    RealityViewVACEModel,
    pack_vace_control_tensor,
    pack_vace_mask,
)
from open_reality_view.rendering import render_gaussians_triton


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
    h, w = 32, 32
    k = torch.tensor([[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]])
    src_k = k[:, None].repeat(1, 2, 1, 1)
    pose = build_chase_camera_pose(offset_xyz=(0.0, 0.0, 1.5), pitch_deg=0.0)[None]
    src_c2e = pose[:, None].repeat(1, 2, 1, 1)

    images = torch.zeros(1, 2, 3, h, w)
    images[:, 0, 0] = 1.0  # cam 0 = red
    images[:, 1, 2] = 1.0  # cam 1 = blue

    depths = torch.zeros(1, 2, 1, h, w)
    depths[:, 0] = 5.0   # near
    depths[:, 1] = 20.0  # far
    depths[:, :, :, :4, :4] = 0.0

    projector = MultiViewDepthProjector(splat_radius_px=0)
    res = projector(images, depths, src_k, src_c2e, k, pose, target_hw=(h, w))

    assert res.coarse_rgb[0, 0, 16, 16] == pytest.approx(1.0)
    assert res.coarse_rgb[0, 2, 16, 16] == pytest.approx(0.0)
    assert res.target_depth[0, 0, 16, 16] == pytest.approx(5.0, abs=1e-3)
    assert res.hole_mask[0, 0, 16, 16] == pytest.approx(0.0)
    assert res.hole_mask[0, 0, 1, 1] == pytest.approx(1.0)


def test_ego_vehicle_compositor_fills_ego_hole_with_zbuffer() -> None:
    h, w = 64, 96
    k_tgt = torch.tensor([[[80.0, 0.0, w / 2], [0.0, 80.0, h / 2], [0.0, 0.0, 1.0]]])
    pose = build_chase_camera_pose(offset_xyz=(-7.5, 0.0, 4.2), pitch_deg=-18.0)[None]
    empty = ProjectionResult(
        coarse_rgb=torch.zeros(1, 3, h, w),
        hole_mask=torch.ones(1, 1, h, w),
        target_depth=torch.zeros(1, 1, h, w),
    )
    compositor = EgoVehicleCompositor(points_per_meter=14, splat_radius_px=1)
    skybox = torch.full((6, 16, 16, 3), 0.65)
    out = compositor(empty, k_tgt, pose, skybox_textures=skybox, T_rig_world=torch.eye(4))

    observed_pixels = (out.hole_mask == 0.0).sum().item()
    assert observed_pixels > 100
    assert out.coarse_rgb.max().item() > 0.2
    assert out.target_depth[out.hole_mask == 0.0].min().item() > 3.0


def test_nurec_base_albedo_extracts_dc_term_without_fourier_washout() -> None:
    from open_reality_view.data.nurec_renderer import _extract_base_albedo

    albedo_2d = torch.randn(10, 3)
    assert torch.equal(_extract_base_albedo(albedo_2d), albedo_2d)

    # NuRec 26.10 stores DC base color at col 0 and zero-centered Fourier harmonics at cols 1..K-1
    albedo_3d = torch.zeros(10, 5, 3)
    albedo_3d[:, 0, :] = torch.linspace(-1.0, 1.0, 30).reshape(10, 3)
    extracted = _extract_base_albedo(albedo_3d)
    assert extracted.shape == (10, 3)
    assert torch.equal(extracted, albedo_3d[:, 0, :])


def test_siglip2_patchify_and_naflex_encoder_contract() -> None:
    # Verify exact (P, P, C) flattening order matching transformers Siglip2ImageProcessor
    img = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    patches, grid_hw = patchify_siglip2(img, patch_size=2)
    assert grid_hw == (2, 2)
    assert patches.shape == (1, 4, 2 * 2 * 3)
    # First patch (y in 0..1, x in 0..1), pixel (0,0) channels [0, 16, 32] come first
    assert torch.equal(patches[0, 0, :3], torch.tensor([0.0, 16.0, 32.0]))

    tiny_cfg = Siglip2VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_channels=3,
        patch_size=16,
        num_patches=16,
        vision_use_head=False,
    )
    tower = Siglip2VisionModel(tiny_cfg)
    encoder = Alpamayo2EdgeVisionEncoder(
        vision_tower=tower,
        hidden_size=64,
        patch_size=16,
        is_siglip2_naflex=True,
        target_hw=(32, 48),
        freeze=True,
    )
    imgs = torch.rand(1, 2, 6, 3, 64, 96)
    enc_out = encoder.encode_multiview(imgs)
    assert enc_out.grid_hw == (2, 3)
    assert enc_out.tokens.shape == (1, 2, 6, 2, 3, 64)


def test_synthetic_clip_and_cached_shard_roundtrip(tmp_path: Path) -> None:
    clip = generate_synthetic_driving_clip(
        clip_id="test_clip",
        num_frames=5,
        src_hw=(64, 96),
        tgt_hw=(64, 96),
    )
    assert clip.surround_rgb.shape == (5, 6, 3, 64, 96)
    assert clip.surround_depth is not None and clip.surround_depth.shape == (5, 6, 1, 64, 96)
    assert clip.chase_gt_rgb.shape == (5, 3, 64, 96)

    shard_path = tmp_path / "test_clip.pt"
    save_cached_shard(
        shard_path,
        clean_latents=torch.randn(16, 2, 8, 12),
        vace_context=torch.randn(96, 2, 8, 12),
        vision_tokens=torch.randn(2, 6, 4, 6, 1152),
        plucker_rays=torch.randn(2, 6, 6, 4, 6),
        clip_id="test_clip",
    )
    ds = CachedShardDataset(tmp_path)
    assert len(ds) == 1
    sample = ds[0]
    assert sample["clean_latents"].dtype == torch.bfloat16
    assert sample["vace_context"].shape == (96, 2, 8, 12)


class _MockWanVACETransformer(nn.Module):
    """Stub asserting exact diffusers.WanVACETransformer3DModel tensor contracts."""

    def __init__(self, num_vace_layers: int = 15) -> None:
        super().__init__()
        self.config = SimpleNamespace(vace_layers=list(range(0, num_vace_layers * 2, 2)))
        self.latent_proj = nn.Conv3d(16 + 96, 16, kernel_size=1)
        self.ctx_proj = nn.Linear(4096, 16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        control_hidden_states_scale: torch.Tensor,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        assert hidden_states.shape[1] == 16
        assert control_hidden_states.shape[1] == 96
        assert encoder_hidden_states.shape[-1] == 4096
        assert timestep.shape[0] == hidden_states.shape[0]
        assert isinstance(control_hidden_states_scale, torch.Tensor)
        assert control_hidden_states_scale.shape == (len(self.config.vace_layers),)

        ctx_bias = self.ctx_proj(encoder_hidden_states.mean(dim=1))[:, :, None, None, None]
        scale = control_hidden_states_scale.mean()
        fused = torch.cat([hidden_states, control_hidden_states * scale], dim=1)
        out = self.latent_proj(fused) + ctx_bias
        return SimpleNamespace(sample=out)


def test_end_to_end_vace_flow_matching_step_and_gradients() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bsz, num_frames_lat, num_cams = 1, 3, 6
    t_pixel, h_tgt, w_tgt = 9, 64, 64
    h_lat, w_lat = h_tgt // 8, w_tgt // 8

    hole_mask = torch.randint(0, 2, (bsz, 1, t_pixel, h_tgt, w_tgt), device=device).float()
    packed_mask = pack_vace_mask(hole_mask, spatial_stride=8, temporal_stride=4)
    assert packed_mask.shape == (bsz, 64, num_frames_lat, h_lat, w_lat)

    vace_ctrl = pack_vace_control_tensor(
        torch.randn(bsz, 16, num_frames_lat, h_lat, w_lat, device=device),
        torch.randn(bsz, 16, num_frames_lat, h_lat, w_lat, device=device),
        packed_mask,
    )
    assert vace_ctrl.shape == (bsz, 96, num_frames_lat, h_lat, w_lat)

    adapter = AlpamayoContextAdapter(
        in_dim=1152,
        adapter_dim=128,
        out_dim=4096,
        num_latents=32,
        num_layers=2,
        num_heads=4,
    ).to(device)
    model = RealityViewVACEModel(
        transformer=_MockWanVACETransformer(num_vace_layers=15).to(device),
        context_adapter=adapter,
    )
    model.freeze_backbone_for_stage1a()

    vision_tokens = torch.randn(bsz, num_frames_lat, num_cams, 4, 4, 1152, device=device)
    plucker = torch.randn(bsz, num_frames_lat, num_cams, 6, 4, 4, device=device)
    clean_latents = torch.randn(bsz, 16, num_frames_lat, h_lat, w_lat, device=device)

    stats = model.compute_flow_matching_loss(
        clean_latents=clean_latents,
        vace_context=vace_ctrl,
        vision_tokens=vision_tokens,
        plucker_rays=plucker,
    )
    assert torch.isfinite(stats["loss"])
    stats["loss"].backward()

    assert model.context_adapter.out_proj.weight.grad is not None
    assert model.transformer.latent_proj.weight.grad is None


def test_zero_compiler_3dgs_rasterizer_renders_visible_gaussian_and_depth() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    means = torch.tensor([[0.0, 0.0, 5.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.6, 0.6, 0.6]], device=device)
    opacities = torch.tensor([0.95], device=device)
    # SH DC coeff for pure bright red: (1.0 - 0.5) / C0 ~ 1.77
    sh_dc = torch.tensor([[1.77, -1.77, -1.77]], device=device)
    c2w = torch.eye(4, device=device)
    k = torch.tensor([[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]], device=device)

    out = render_gaussians_triton(
        means=means,
        quats_wxyz=quats,
        scales=scales,
        opacities=opacities,
        sh_dc=sh_dc,
        sh_rest=None,
        c2w=c2w,
        intrinsics=k,
        height=64,
        width=64,
    )
    assert out.rgb.shape == (64, 64, 3)
    assert out.depth.shape == (64, 64)
    assert out.alpha.shape == (64, 64)
    assert out.alpha[32, 32].item() > 0.85
    assert out.rgb[32, 32, 0].item() > 0.85
    assert out.rgb[32, 32, 2].item() < 0.1
    assert out.depth[32, 32].item() == pytest.approx(5.0, abs=0.2)


def test_temporal_3d_cache_projector_fuses_past_observations_via_odometry() -> None:
    h, w = 32, 48
    k = torch.tensor([[36.0, 0.0, w / 2], [0.0, 36.0, h / 2], [0.0, 0.0, 1.0]])
    # Single forward camera looking straight ahead (+X_ego)
    cam_fwd = build_chase_camera_pose(offset_xyz=(2.0, 0.0, 1.4), pitch_deg=-15.0)
    tgt_pose = build_chase_camera_pose(offset_xyz=(-7.5, 0.0, 4.2), pitch_deg=-18.0)

    # Ego advances +4m along +X between t=0 and t=1 -> ground seen ahead at t=0 is now directly behind ego at t=1!
    r2w_0 = torch.eye(4)
    r2w_1 = torch.eye(4)
    r2w_1[0, 3] = 4.0

    imgs = torch.ones(1, 2, 1, 3, h, w) * 0.5
    depths = torch.full((1, 2, 1, 1, h, w), 6.0)
    src_k = k.view(1, 1, 1, 3, 3).repeat(1, 2, 1, 1, 1)
    src_c2e = cam_fwd.view(1, 1, 1, 4, 4).repeat(1, 2, 1, 1, 1)
    r2w = torch.stack([r2w_0, r2w_1], dim=0).unsqueeze(0)

    proj = Temporal3DCacheProjector(splat_radius_px=1)
    res_single = proj(imgs[:, 1], depths[:, 1], src_k[:, 1], src_c2e[:, 1], k[None], tgt_pose[None], (h, w))
    res_cache = proj.project_temporal_window(imgs, depths, src_k, src_c2e, r2w, k[None], tgt_pose[None], (h, w))

    # Multi-frame 3D cache should strictly reduce blind-spot holes compared to single-frame projection!
    assert res_cache.hole_mask.mean().item() < res_single.hole_mask.mean().item()


def test_trajectory_ribbon_renders_ahead_and_respects_depth_occlusion() -> None:
    h, w = 64, 96
    k = torch.tensor([[80.0, 0.0, w / 2], [0.0, 80.0, h / 2], [0.0, 0.0, 1.0]])
    pose = build_chase_camera_pose(offset_xyz=(-7.5, 0.0, 4.2), pitch_deg=-18.0)
    bg = torch.full((h, w, 3), 0.2)
    far_depth = torch.full((h, w), 80.0)
    waypoints = torch.stack([torch.linspace(0.0, 30.0, 20), torch.zeros(20), torch.zeros(20)], dim=-1)

    rendered = composite_trajectory_ribbon(bg, far_depth, k, pose, waypoints, start_x_m=3.0)
    # Blue/cyan channel should increase noticeably along the road ahead
    assert (rendered[..., 2] - bg[..., 2]).max().item() > 0.2

    # When a close obstacle (z=2.0m everywhere) occludes the road, ribbon must stay occluded!
    occluded = composite_trajectory_ribbon(bg, torch.full((h, w), 2.0), k, pose, waypoints, start_x_m=3.0)
    assert torch.allclose(occluded, bg, atol=1e-5)


def test_glb_ego_mesh_loader_aligns_and_grounds_when_present() -> None:
    glb = Path(__file__).resolve().parents[1] / "assets/ego_car_concept.glb"
    if not glb.is_file():
        pytest.skip("Optional assets/ego_car_concept.glb not downloaded")
    mesh = load_glb_ego_mesh(glb, ego_box_lwh=(4.72, 1.85, 1.44), paint_theme="pearl_white")
    assert mesh.num_triangles > 10000
    assert mesh.vertex_data.shape == (mesh.num_triangles * 3, 12)
    assert mesh.bbox_min[2] == pytest.approx(0.0, abs=1e-3)
    assert (mesh.bbox_max[0] - mesh.bbox_min[0]) == pytest.approx(4.72, abs=0.05)
