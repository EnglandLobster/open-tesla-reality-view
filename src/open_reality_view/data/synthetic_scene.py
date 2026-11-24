"""Procedural 3D highway scene renderer for zero-download end-to-end verification."""

from __future__ import annotations

import math

import torch

from open_reality_view.data.dataset import DrivingClip
from open_reality_view.geometry.ego_car import EgoVehicleCompositor
from open_reality_view.geometry.projector import (
    MultiViewDepthProjector,
    build_chase_camera_pose,
)


def _pinhole_intrinsics(height: int, width: int, fov_x_deg: float = 90.0) -> torch.Tensor:
    fx = (width * 0.5) / math.tan(math.radians(fov_x_deg * 0.5))
    fy = fx
    return torch.tensor(
        [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _default_surround_rig() -> list[tuple[tuple[float, float, float], float, float]]:
    """6-camera surround rig ``(offset_xyz, pitch_deg, yaw_deg)`` in ego coordinates."""
    return [
        ((1.6, 0.0, 1.5), -3.0, 0.0),      # cam_front
        ((0.9, 0.85, 1.45), -4.0, 55.0),   # cam_front_left
        ((0.9, -0.85, 1.45), -4.0, -55.0), # cam_front_right
        ((-2.2, 0.0, 1.4), -6.0, 180.0),   # cam_back
        ((-0.5, 0.88, 1.45), -4.0, 125.0), # cam_back_left
        ((-0.5, -0.88, 1.45), -4.0, -125.0),  # cam_back_right
    ]


def _render_analytic_road_view(
    c2e: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    ego_travel_x: float,
    traffic_cars: list[tuple[float, float, tuple[float, float, float]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytically raycast a 3-lane highway + sky + neighboring cars for one camera.

    Returns:
        ``(rgb [3, H, W] in [0, 1], depth [1, H, W] in meters)``.
    """
    v = torch.arange(height, dtype=torch.float32) + 0.5
    u = torch.arange(width, dtype=torch.float32) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    x_c = (uu - intrinsics[0, 2]) / intrinsics[0, 0]
    y_c = (vv - intrinsics[1, 2]) / intrinsics[1, 1]
    dirs_cam = torch.stack([x_c, y_c, torch.ones_like(x_c)], dim=-1)  # [H, W, 3]
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)

    rot = c2e[:3, :3]
    origin = c2e[:3, 3]
    dirs_ego = dirs_cam @ rot.T  # [H, W, 3]

    # Sky gradient background
    sky_t = dirs_ego[..., 2].clamp(0.0, 1.0)
    rgb = torch.empty((height, width, 3), dtype=torch.float32)
    rgb[..., 0] = 0.65 - 0.35 * sky_t
    rgb[..., 1] = 0.78 - 0.25 * sky_t
    rgb[..., 2] = 0.94 - 0.12 * sky_t
    depth = torch.full((height, width), 120.0, dtype=torch.float32)

    # Ground plane z = 0 intersection
    dz = dirs_ego[..., 2]
    hit_ground = dz < -1e-4
    t_ground = torch.where(hit_ground, -origin[2] / dz.clamp(max=-1e-4), torch.full_like(dz, 120.0))
    valid_ground = hit_ground & (t_ground > 0.3) & (t_ground < 115.0)

    gx = origin[0] + t_ground * dirs_ego[..., 0] + ego_travel_x
    gy = origin[1] + t_ground * dirs_ego[..., 1]

    # Road width: y in [-6.0, 6.0] (3 lanes of 3.6m + shoulders); grass outside.
    on_road = valid_ground & (gy.abs() <= 6.2)
    on_grass = valid_ground & (gy.abs() > 6.2)

    # Asphalt texture + dashed lane markings + solid yellow shoulder lines
    asphalt = torch.tensor([0.22, 0.23, 0.25])
    grass = torch.tensor([0.20, 0.42, 0.18])
    white_line = torch.tensor([0.92, 0.92, 0.90])
    yellow_line = torch.tensor([0.92, 0.72, 0.15])

    rgb = torch.where(on_grass[..., None], grass, rgb)
    rgb = torch.where(on_road[..., None], asphalt, rgb)

    dashed = (gx % 6.0) < 2.6
    lane_div = on_road & dashed & (((gy - 1.8).abs() < 0.11) | ((gy + 1.8).abs() < 0.11))
    edge_line = on_road & (((gy - 5.4).abs() < 0.14) | ((gy + 5.4).abs() < 0.14))
    rgb = torch.where(lane_div[..., None], white_line, rgb)
    rgb = torch.where(edge_line[..., None], yellow_line, rgb)

    z_cam_ground = t_ground * dirs_cam[..., 2]
    depth = torch.where(valid_ground, z_cam_ground, depth)

    # Ray-AABB intersection for neighboring traffic cars (dimensions 4.5 x 1.9 x 1.5 m)
    for car_x, car_y, color in traffic_cars:
        bmin = torch.tensor([car_x - 2.25, car_y - 0.95, 0.0])
        bmax = torch.tensor([car_x + 2.25, car_y + 0.95, 1.50])
        inv_d = 1.0 / torch.where(dirs_ego.abs() > 1e-6, dirs_ego, torch.full_like(dirs_ego, 1e-6))
        t1 = (bmin - origin) * inv_d
        t2 = (bmax - origin) * inv_d
        tmin = torch.minimum(t1, t2).amax(dim=-1)
        tmax = torch.maximum(t1, t2).amin(dim=-1)
        hit = (tmax > tmin) & (tmin > 0.3)
        z_car = tmin * dirs_cam[..., 2]
        closer = hit & (z_car < depth)
        if closer.any():
            hit_z = origin[2] + tmin * dirs_ego[..., 2]
            shade = (0.55 + 0.45 * (hit_z / 1.5).clamp(0.0, 1.0))[..., None]
            car_col = torch.tensor(color, dtype=torch.float32) * shade
            rgb = torch.where(closer[..., None], car_col, rgb)
            depth = torch.where(closer, z_car, depth)

    return rgb.permute(2, 0, 1).clamp(0.0, 1.0).contiguous(), depth.unsqueeze(0).contiguous()


def generate_synthetic_driving_clip(
    clip_id: str = "synthetic_highway_000",
    *,
    num_frames: int = 9,
    src_hw: tuple[int, int] = (224, 384),
    tgt_hw: tuple[int, int] = (256, 448),
    chase_offset_xyz: tuple[float, float, float] = (-7.5, 0.0, 4.2),
    chase_pitch_deg: float = -18.0,
) -> DrivingClip:
    """Generate a self-consistent 6-camera + third-person GT highway driving clip."""
    rig = _default_surround_rig()
    num_cams = len(rig)
    k_src = _pinhole_intrinsics(*src_hw, fov_x_deg=95.0)
    k_tgt = _pinhole_intrinsics(*tgt_hw, fov_x_deg=72.0)

    src_c2e_single = torch.stack(
        [build_chase_camera_pose(offset_xyz=off, pitch_deg=pitch, yaw_deg=yaw) for off, pitch, yaw in rig],
        dim=0,
    )  # [K, 4, 4]
    tgt_c2e_single = build_chase_camera_pose(offset_xyz=chase_offset_xyz, pitch_deg=chase_pitch_deg)
    ego_compositor = EgoVehicleCompositor(points_per_meter=16, splat_radius_px=1)
    # Use a dummy projection wrapper to composite the ego car directly onto GT chase views.
    _dummy_proj = MultiViewDepthProjector()
    _ = _dummy_proj  # keep geometry pipeline aligned

    surround_rgbs: list[torch.Tensor] = []
    surround_depths: list[torch.Tensor] = []
    chase_gts: list[torch.Tensor] = []

    for t in range(num_frames):
        ego_x = t * 1.4  # ~22 m/s @ 16fps
        # Two dynamic neighbor cars in adjacent lanes (relative to ego frame)
        traffic = [
            (13.0 + t * 0.25, 3.6, (0.78, 0.16, 0.16)),   # Red car ahead-left
            (8.5 - t * 0.15, -3.6, (0.15, 0.38, 0.78)),   # Blue car ahead-right
        ]

        cam_rgbs = []
        cam_ds = []
        for k_idx in range(num_cams):
            rgb_k, d_k = _render_analytic_road_view(
                src_c2e_single[k_idx], k_src, src_hw[0], src_hw[1], ego_x, traffic
            )
            cam_rgbs.append(rgb_k)
            cam_ds.append(d_k)

        surround_rgbs.append(torch.stack(cam_rgbs, dim=0))
        surround_depths.append(torch.stack(cam_ds, dim=0))

        # Ground-truth third-person chase view (analytic road + neighbor cars + 3D ego car)
        gt_rgb, gt_depth = _render_analytic_road_view(
            tgt_c2e_single, k_tgt, tgt_hw[0], tgt_hw[1], ego_x, traffic
        )
        from open_reality_view.geometry.projector import ProjectionResult

        gt_proj = ProjectionResult(
            coarse_rgb=gt_rgb.unsqueeze(0),
            hole_mask=torch.zeros((1, 1, tgt_hw[0], tgt_hw[1]), dtype=torch.float32),
            target_depth=gt_depth.unsqueeze(0),
        )
        gt_with_ego = ego_compositor(gt_proj, k_tgt.unsqueeze(0), tgt_c2e_single.unsqueeze(0))
        chase_gts.append(gt_with_ego.coarse_rgb.squeeze(0))

    return DrivingClip(
        clip_id=clip_id,
        surround_rgb=torch.stack(surround_rgbs, dim=0),
        surround_depth=torch.stack(surround_depths, dim=0),
        src_intrinsics=k_src[None, None].expand(num_frames, num_cams, 3, 3).clone(),
        src_c2e=src_c2e_single[None].expand(num_frames, num_cams, 4, 4).clone(),
        chase_gt_rgb=torch.stack(chase_gts, dim=0),
        tgt_intrinsics=k_tgt[None].expand(num_frames, 3, 3).clone(),
        tgt_c2e=tgt_c2e_single[None].expand(num_frames, 4, 4).clone(),
    )
