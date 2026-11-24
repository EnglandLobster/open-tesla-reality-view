"""Tesla FSD v15-style 3D road trajectory light ribbon and directional chevron overlay.

Renders a depth-tested, translucent cyan-blue planned path ribbon with animated forward chevrons
onto the road plane (``z = 0.05 m`` in ego coordinates), using real future ego waypoints or
Alpamayo2-Edge Action Expert trajectories.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def composite_trajectory_ribbon(
    rgb_hwc: torch.Tensor,
    depth_hw: torch.Tensor,
    intrinsics: torch.Tensor,
    tgt_c2e: torch.Tensor,
    waypoints_ego: torch.Tensor,
    *,
    ribbon_width_m: float = 1.48,
    height_offset_m: float = 0.05,
    start_x_m: float = 3.2,
    phase_sec: float = 0.0,
    max_opacity: float = 0.62,
    core_color: tuple[float, float, float] = (0.14, 0.72, 1.00),
    edge_color: tuple[float, float, float] = (0.58, 0.93, 1.00),
    chevron_color: tuple[float, float, float] = (0.85, 0.98, 1.00),
) -> torch.Tensor:
    """Alpha-blend a depth-tested Tesla-style 3D path ribbon onto ``rgb_hwc [H, W, 3]`` in ``[0, 1]``.

    Args:
        rgb_hwc: Rendered chase-camera RGB image ``[H, W, 3]`` in ``[0, 1]``.
        depth_hw: Metric camera Z depth ``[H, W]`` (``<= 0`` treated as sky / infinite depth).
        intrinsics: Target pinhole intrinsics ``[3, 3]``.
        tgt_c2e: Target camera-to-ego SE(3) matrix ``[4, 4]``.
        waypoints_ego: Ordered future centerline waypoints ``[N, 3]`` (or ``[N, 2]``) in ego frame.
        ribbon_width_m: Total width of the road path ribbon in meters.
        height_offset_m: Height above ``z = 0`` road plane in meters.
        start_x_m: Distance ahead of ego origin where the ribbon smoothly fades in (front bumper).
        phase_sec: Timestamp in seconds used to animate forward-moving chevrons.
        max_opacity: Maximum blending opacity of the translucent carpet.
    """
    if waypoints_ego.numel() < 4:
        return rgb_hwc

    device = rgb_hwc.device
    h, w = rgb_hwc.shape[:2]

    pts = waypoints_ego.to(device=device, dtype=torch.float32)
    if pts.shape[-1] == 2:
        pts = torch.cat([pts, torch.zeros_like(pts[:, :1])], dim=-1)

    # Filter to waypoints ahead of the front bumper (or keep if turning sharply)
    # Build dense arc-length-parameterized polyline in XY
    xy = pts[:, :2]  # [N, 2]
    seg_vec = xy[1:] - xy[:-1]  # [N-1, 2]
    seg_len = seg_vec.norm(dim=-1)
    keep = seg_len > 1e-3
    if not keep.any():
        return rgb_hwc

    p0 = xy[:-1][keep]
    p1 = xy[1:][keep]
    seg_vec = p1 - p0
    seg_len = seg_vec.norm(dim=-1)
    cum_s = torch.cat([torch.zeros(1, device=device), torch.cumsum(seg_len, dim=0)])
    total_len = float(cum_s[-1].item())
    if total_len < 1.5:
        return rgb_hwc

    # Intersect every pixel ray with the z = height_offset_m ground plane in ego frame
    k = intrinsics.to(device=device, dtype=torch.float32)
    c2e = tgt_c2e.to(device=device, dtype=torch.float32)
    cam_pos = c2e[:3, 3]
    r_c2e = c2e[:3, :3]

    py = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    px = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    gy, gx = torch.meshgrid(py, px, indexing="ij")

    ray_c = torch.stack(
        [(gx - k[0, 2]) / k[0, 0], (gy - k[1, 2]) / k[1, 1], torch.ones_like(gx)],
        dim=-1,
    )
    ray_e = ray_c @ r_c2e.T  # [H, W, 3]

    # Plane intersection: cam_pos.z + t * ray_e.z = height_offset_m
    denom = ray_e[..., 2]
    hit = denom < -1e-3
    t_hit = torch.where(hit, (height_offset_m - cam_pos[2]) / denom.clamp(max=-1e-3), torch.full_like(gx, -1.0))
    # Camera-space Z of ground intersection equals t_hit (since ray_c.z == 1.0)
    valid_ground = hit & (t_hit > 0.5) & (t_hit < 75.0)
    if not valid_ground.any():
        return rgb_hwc

    # Strict z-buffer occlusion test against ego car & lead vehicles
    d_scene = depth_hw.to(device=device, dtype=torch.float32)
    scene_unoccluded = (d_scene <= 0.1) | (t_hit <= d_scene + 0.40)

    # Bounding-box prefilter in XY around the polyline
    half_w = ribbon_width_m * 0.5
    pad = half_w + 0.35
    x_min = float(xy[:, 0].min().item()) - pad
    x_max = float(xy[:, 0].max().item()) + pad
    y_min = float(xy[:, 1].min().item()) - pad
    y_max = float(xy[:, 1].max().item()) + pad

    gx_e = cam_pos[0] + t_hit * ray_e[..., 0]
    gy_e = cam_pos[1] + t_hit * ray_e[..., 1]

    cand_mask = (
        valid_ground
        & scene_unoccluded
        & (gx_e >= max(start_x_m - 0.5, x_min))
        & (gx_e <= x_max)
        & (gy_e >= y_min)
        & (gy_e <= y_max)
    )
    if not cand_mask.any():
        return rgb_hwc

    cand_xy = torch.stack([gx_e[cand_mask], gy_e[cand_mask]], dim=-1)  # [P, 2]
    # Compute signed lateral distance & longitudinal arc length s to polyline segments
    # [P, S, 2]
    diff = cand_xy[:, None, :] - p0[None, :, :]
    seg_len_sq = (seg_len * seg_len).clamp_min(1e-6)
    proj_t = ((diff * seg_vec[None, :, :]).sum(dim=-1) / seg_len_sq[None, :]).clamp(0.0, 1.0)
    closest = p0[None, :, :] + proj_t[..., None] * seg_vec[None, :, :]
    delta = cand_xy[:, None, :] - closest
    dist_sq = (delta * delta).sum(dim=-1)  # [P, S]

    min_d2, best_seg = dist_sq.min(dim=-1)  # [P]
    lat_dist = torch.sqrt(min_d2.clamp_min(0.0))  # [P]
    best_t = proj_t[torch.arange(cand_xy.shape[0], device=device), best_seg]
    s_arc = cum_s[best_seg] + best_t * seg_len[best_seg]  # [P]
    cand_x = cand_xy[:, 0]

    inside = lat_dist <= half_w + 0.14
    if not inside.any():
        return rgb_hwc

    # Normalized lateral position u in [0, 1] (0=center, 1=edge)
    u = (lat_dist / half_w).clamp(0.0, 1.4)

    # Smooth translucent interior + bright glowing outer rail edges (Tesla FSD v15 style)
    core_profile = torch.clamp(1.0 - torch.pow(u.clamp(max=1.0), 4.0), 0.0, 1.0)
    edge_glow = torch.exp(-((u - 0.92) / 0.12) ** 2) * 0.95

    # Forward-pointing animated glowing chevrons (`>>>`) every 3.6 meters
    chevron_spacing = 3.6
    # V-shape offset: center leads ahead of edges by 0.65m
    s_chev = (s_arc - phase_sec * 4.5 + u.clamp(max=1.0) * 0.65) % chevron_spacing
    chev_band = torch.exp(-((s_chev - 0.35) / 0.18) ** 2) * (1.0 - (u / 0.88).clamp(0.0, 1.0) ** 3)

    # Smooth longitudinal fade-in past front bumper & fade-out at far horizon
    fade_near = ((cand_x - start_x_m) / 1.4).clamp(0.0, 1.0)
    fade_far = (1.0 - ((s_arc - total_len * 0.62) / max(1.0, total_len * 0.38)).clamp(0.0, 1.0)) ** 1.5
    long_fade = fade_near * fade_far * (1.0 - (u - 1.0).clamp(0.0, 0.25) / 0.25)

    alpha = (0.36 * core_profile + 0.58 * edge_glow + 0.65 * chev_band).clamp(0.0, 1.0) * max_opacity * long_fade

    c_core = torch.tensor(core_color, device=device, dtype=torch.float32)
    c_edge = torch.tensor(edge_color, device=device, dtype=torch.float32)
    c_chev = torch.tensor(chevron_color, device=device, dtype=torch.float32)

    ribbon_col = (
        c_core[None, :] * (1.0 - 0.5 * edge_glow[:, None])
        + c_edge[None, :] * (0.65 * edge_glow[:, None])
        + c_chev[None, :] * (0.85 * chev_band[:, None])
    ).clamp(0.0, 1.0)

    out = rgb_hwc.clone().float()
    orig_pix = out[cand_mask]
    # Soft screen-like luminous blend so asphalt lane markings still show through cleanly
    blended = orig_pix * (1.0 - alpha[:, None] * 0.72) + ribbon_col * alpha[:, None]
    out[cand_mask] = blended.clamp(0.0, 1.0)
    return out.to(dtype=rgb_hwc.dtype)


def extract_future_waypoints_ego(
    rig_ts_us: torch.Tensor | list[float],
    T_rig_worlds: torch.Tensor,
    current_ts_us: float,
    *,
    horizon_sec: float = 4.5,
    num_points: int = 36,
) -> torch.Tensor:
    """Interpolate real future ego positions ``[num_points, 3]`` in the current ego rig frame."""
    import numpy as np

    from open_reality_view.data.nurec_renderer import _interp_se3

    ts_np = np.asarray(rig_ts_us, dtype=np.float64)
    poses_np = T_rig_worlds.detach().cpu().numpy() if isinstance(T_rig_worlds, torch.Tensor) else np.asarray(T_rig_worlds, dtype=np.float32)

    t_cur = _interp_se3(ts_np, poses_np, current_ts_us)
    t_cur_inv = np.linalg.inv(t_cur)

    max_ts = min(float(ts_np[-1]), current_ts_us + horizon_sec * 1e6)
    if max_ts <= current_ts_us + 1e5:
        return torch.empty((0, 3), dtype=torch.float32)

    sample_ts = np.linspace(current_ts_us, max_ts, num_points)
    pts_ego = []
    for ts in sample_ts:
        t_fut = _interp_se3(ts_np, poses_np, float(ts))
        rel = t_cur_inv @ t_fut
        pts_ego.append(rel[:3, 3])
    arr = np.stack(pts_ego, axis=0).astype(np.float32)

    # Check if vehicle actually moved enough; if nearly stationary, draw a subtle straight forward guide
    dist = float(math.hypot(float(arr[-1, 0] - arr[0, 0]), float(arr[-1, 1] - arr[0, 1])))
    if dist < 4.0:
        xs = np.linspace(0.0, 18.0, num_points, dtype=np.float32)
        arr = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=-1)
    return torch.from_numpy(arr)
