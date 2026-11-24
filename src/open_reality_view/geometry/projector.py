"""Multi-camera metric depth unprojection and target chase-camera z-buffer splatting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ProjectionResult:
    """Output of projecting surround cameras into a virtual third-person chase camera."""

    coarse_rgb: torch.Tensor  # [B, 3, H_tgt, W_tgt], in same range as input images (0 where hole)
    hole_mask: torch.Tensor  # [B, 1, H_tgt, W_tgt], 1.0 = blind spot / unobserved, 0.0 = observed
    target_depth: torch.Tensor  # [B, 1, H_tgt, W_tgt], metric depth in meters (0 where hole)


def build_chase_camera_pose(
    offset_xyz: tuple[float, float, float] = (-7.5, 0.0, 4.2),
    pitch_deg: float = -18.0,
    yaw_deg: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an OpenCV-convention target camera-to-ego ``[4, 4]`` transform.

    Ego frame convention: ``+X`` forward, ``+Y`` left, ``+Z`` up.
    Camera frame convention (OpenCV): ``+X`` right, ``+Y`` down, ``+Z`` forward.
    Negative ``pitch_deg`` tilts the camera downward toward the road ahead.
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # Pitch around camera right axis (+X_cam = -Y_ego) in ego frame.
    # Columns are [x_cam, y_cam, z_cam] expressed in ego coordinates.
    rot_pitch = torch.tensor(
        [
            [0.0, sp, cp],
            [-1.0, 0.0, 0.0],
            [0.0, -cp, sp],
        ],
        device=device,
        dtype=dtype,
    )
    rot_yaw = torch.tensor(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )

    c2e = torch.eye(4, device=device, dtype=dtype)
    c2e[:3, :3] = rot_yaw @ rot_pitch
    c2e[:3, 3] = torch.tensor(offset_xyz, device=device, dtype=dtype)
    return c2e


def build_orbit_chase_pose(
    radius: float = 8.5,
    elevation_deg: float = 24.0,
    azimuth_deg: float = 0.0,
    look_at_xyz: tuple[float, float, float] = (2.0, 0.0, 0.8),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an interactive orbit camera pose looking at ``look_at_xyz`` in ego coordinates.

    ``azimuth_deg = 0`` places the camera directly behind the ego vehicle looking forward.
    """
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)

    look_at = torch.tensor(look_at_xyz, device=device, dtype=dtype)
    cam_pos = look_at + torch.tensor(
        [
            -radius * math.cos(el) * math.cos(az),
            -radius * math.cos(el) * math.sin(az),
            radius * math.sin(el),
        ],
        device=device,
        dtype=dtype,
    )

    z_cam = look_at - cam_pos
    z_cam = z_cam / z_cam.norm().clamp_min(1e-6)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    x_cam = torch.linalg.cross(z_cam, world_up)
    x_cam = x_cam / x_cam.norm().clamp_min(1e-6)
    y_cam = torch.linalg.cross(z_cam, x_cam)

    c2e = torch.eye(4, device=device, dtype=dtype)
    c2e[:3, 0] = x_cam
    c2e[:3, 1] = y_cam
    c2e[:3, 2] = z_cam
    c2e[:3, 3] = cam_pos
    return c2e


class MultiViewDepthProjector(nn.Module):
    """Unproject surround-view RGB+Depth into ego 3D and z-buffer splat to a target chase view."""

    def __init__(
        self,
        min_depth_m: float = 0.5,
        max_depth_m: float = 120.0,
        splat_radius_px: int = 1,
    ) -> None:
        super().__init__()
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.splat_radius_px = splat_radius_px

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        src_intrinsics: torch.Tensor,
        src_c2e: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_c2e: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> ProjectionResult:
        """Project ``K`` surround cameras into one target third-person camera.

        Args:
            images: ``[B, K, 3, H_src, W_src]`` RGB frames.
            depths: ``[B, K, 1, H_src, W_src]`` metric depth maps in meters.
            src_intrinsics: ``[B, K, 3, 3]`` source camera intrinsics.
            src_c2e: ``[B, K, 4, 4]`` source camera-to-ego transforms.
            tgt_intrinsics: ``[B, 3, 3]`` target virtual camera intrinsics.
            tgt_c2e: ``[B, 4, 4]`` target camera-to-ego transform.
            target_hw: ``(H_tgt, W_tgt)`` output image resolution.
        """
        bsz, _num_cams, _, h_src, w_src = images.shape
        h_tgt, w_tgt = target_hw
        device = images.device
        dtype = images.dtype

        v_src = torch.arange(h_src, device=device, dtype=torch.float32) + 0.5
        u_src = torch.arange(w_src, device=device, dtype=torch.float32) + 0.5
        vv, uu = torch.meshgrid(v_src, u_src, indexing="ij")  # [H_src, W_src]

        tgt_e2c = torch.linalg.inv(tgt_c2e.float())  # [B, 4, 4]
        src_to_tgt = torch.matmul(tgt_e2c[:, None], src_c2e.float())  # [B, K, 4, 4]

        coarse_rgb = torch.zeros((bsz, 3, h_tgt, w_tgt), device=device, dtype=dtype)
        hole_mask = torch.ones((bsz, 1, h_tgt, w_tgt), device=device, dtype=torch.float32)
        target_depth = torch.zeros((bsz, 1, h_tgt, w_tgt), device=device, dtype=torch.float32)

        offsets = [
            (dy, dx)
            for dy in range(-self.splat_radius_px, self.splat_radius_px + 1)
            for dx in range(-self.splat_radius_px, self.splat_radius_px + 1)
        ]

        for b in range(bsz):
            fx = src_intrinsics[b, :, 0, 0].float()[:, None, None]
            fy = src_intrinsics[b, :, 1, 1].float()[:, None, None]
            cx = src_intrinsics[b, :, 0, 2].float()[:, None, None]
            cy = src_intrinsics[b, :, 1, 2].float()[:, None, None]

            d = depths[b, :, 0].float()  # [K, H_src, W_src]
            valid_d = (d >= self.min_depth_m) & (d <= self.max_depth_m) & torch.isfinite(d)

            x_s = (uu[None] - cx) * d / fx.clamp_min(1e-6)
            y_s = (vv[None] - cy) * d / fy.clamp_min(1e-6)
            pts_src = torch.stack([x_s, y_s, d], dim=-1)  # [K, H_src, W_src, 3]

            rot = src_to_tgt[b, :, :3, :3]  # [K, 3, 3]
            trans = src_to_tgt[b, :, :3, 3]  # [K, 3]
            pts_tgt = torch.einsum("kij,khwj->khwi", rot, pts_src) + trans[:, None, None, :]

            z_t = pts_tgt[..., 2]
            in_front = valid_d & (z_t > 0.1)

            k_tgt = tgt_intrinsics[b].float()
            u_t = torch.round(k_tgt[0, 0] * (pts_tgt[..., 0] / z_t.clamp_min(1e-6)) + k_tgt[0, 2])
            v_t = torch.round(k_tgt[1, 1] * (pts_tgt[..., 1] / z_t.clamp_min(1e-6)) + k_tgt[1, 2])

            u_i = u_t.long()
            v_i = v_t.long()
            in_bounds = in_front & (u_i >= 0) & (u_i < w_tgt) & (v_i >= 0) & (v_i < h_tgt)
            if not in_bounds.any():
                continue

            flat_valid = in_bounds.reshape(-1)
            z_sel = z_t.reshape(-1)[flat_valid]
            u_sel = u_i.reshape(-1)[flat_valid]
            v_sel = v_i.reshape(-1)[flat_valid]
            rgb_sel = images[b].permute(0, 2, 3, 1).reshape(-1, 3)[flat_valid]  # [N, 3]

            # Painter's algorithm: sort far-to-near so nearer points overwrite farther ones.
            order = torch.argsort(z_sel, descending=True)
            z_sorted = z_sel[order]
            u_sorted = u_sel[order]
            v_sorted = v_sel[order]
            rgb_sorted = rgb_sel[order]

            canvas_rgb = coarse_rgb[b].view(3, -1)
            canvas_depth = target_depth[b].view(-1)
            canvas_hole = hole_mask[b].view(-1)

            for dy, dx in offsets:
                vv_s = v_sorted + dy
                uu_s = u_sorted + dx
                ok = (uu_s >= 0) & (uu_s < w_tgt) & (vv_s >= 0) & (vv_s < h_tgt)
                if not ok.any():
                    continue
                flat_idx = vv_s[ok] * w_tgt + uu_s[ok]
                canvas_rgb[:, flat_idx] = rgb_sorted[ok].T
                canvas_depth[flat_idx] = z_sorted[ok]
                canvas_hole[flat_idx] = 0.0

        return ProjectionResult(
            coarse_rgb=coarse_rgb,
            hole_mask=hole_mask,
            target_depth=target_depth,
        )
