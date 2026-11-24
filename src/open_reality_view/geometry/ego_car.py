"""Procedural 3D Ego-Vehicle compositor solving the sensor-blind-spot ego-car hole."""

from __future__ import annotations

import torch
from torch import nn

from open_reality_view.geometry.projector import ProjectionResult


def _sample_box_surface(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color_rgb: tuple[float, float, float],
    points_per_meter: int = 24,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample dense surface points, outward normals, and base RGB on an axis-aligned 3D box."""
    cx, cy, cz = center
    lx, ly, lz = size
    nx = max(4, round(lx * points_per_meter))
    ny = max(4, round(ly * points_per_meter))
    nz = max(4, round(lz * points_per_meter))

    xs = torch.linspace(cx - lx / 2, cx + lx / 2, nx)
    ys = torch.linspace(cy - ly / 2, cy + ly / 2, ny)
    zs = torch.linspace(cz - lz / 2, cz + lz / 2, nz)

    pts_list: list[torch.Tensor] = []
    nrm_list: list[torch.Tensor] = []

    def _add_face(grid_a: torch.Tensor, grid_b: torch.Tensor, fixed: float, axis: int, sign: float) -> None:
        a, b = torch.meshgrid(grid_a, grid_b, indexing="ij")
        c = torch.full_like(a, fixed)
        if axis == 0:
            p = torch.stack([c, a, b], dim=-1)
            n = torch.tensor([sign, 0.0, 0.0])
        elif axis == 1:
            p = torch.stack([a, c, b], dim=-1)
            n = torch.tensor([0.0, sign, 0.0])
        else:
            p = torch.stack([a, b, c], dim=-1)
            n = torch.tensor([0.0, 0.0, sign])
        p_flat = p.reshape(-1, 3)
        pts_list.append(p_flat)
        nrm_list.append(n.expand_as(p_flat))

    # Top, left, right, front, rear (skip bottom underbody for speed)
    _add_face(xs, ys, cz + lz / 2, axis=2, sign=1.0)
    _add_face(xs, zs, cy + ly / 2, axis=1, sign=1.0)
    _add_face(xs, zs, cy - ly / 2, axis=1, sign=-1.0)
    _add_face(ys, zs, cx + lx / 2, axis=0, sign=1.0)
    _add_face(ys, zs, cx - lx / 2, axis=0, sign=-1.0)

    pts = torch.cat(pts_list, dim=0)
    nrms = torch.cat(nrm_list, dim=0)
    cols = torch.tensor(color_rgb, dtype=torch.float32).expand_as(pts)
    return pts, nrms, cols


class EgoVehicleCompositor(nn.Module):
    """Composites a shaded 3D ego vehicle + ground shadow into the projected chase view.

    Because ego-mounted surround cameras and LiDAR mask out the ego vehicle itself, third-person
    renders from raw 3DGS / depth projection leave an empty hole at the ego car position.
    Compositing a physically proportioned 3D ego vehicle with z-buffer occlusion gives
    Wan2.1-VACE an exact geometric anchor (``hole_mask = 0``) to harmonize into photorealism.
    """

    def __init__(
        self,
        ego_box_lwh: tuple[float, float, float] = (4.72, 1.85, 1.44),
        body_color: tuple[float, float, float] = (0.86, 0.88, 0.91),
        glass_color: tuple[float, float, float] = (0.14, 0.17, 0.22),
        points_per_meter: int = 22,
        splat_radius_px: int = 1,
    ) -> None:
        super().__init__()
        self.splat_radius_px = splat_radius_px
        length, width, height = ego_box_lwh

        # Lower body, cabin greenhouse, wheels, tail-lights, and soft ground contact shadow.
        parts = [
            # Main lower chassis
            _sample_box_surface((0.0, 0.0, height * 0.34), (length, width, height * 0.44), body_color, points_per_meter),
            # Upper cabin greenhouse (slightly set back & narrower)
            _sample_box_surface((-length * 0.04, 0.0, height * 0.76), (length * 0.54, width * 0.82, height * 0.42), glass_color, points_per_meter),
            # Roof panel on top of cabin
            _sample_box_surface((-length * 0.04, 0.0, height * 0.98), (length * 0.46, width * 0.78, height * 0.05), body_color, points_per_meter),
            # Rear brake/tail light bar
            _sample_box_surface((-length * 0.505, 0.0, height * 0.44), (0.04, width * 0.82, 0.12), (0.92, 0.12, 0.10), points_per_meter),
            # Front headlights
            _sample_box_surface((length * 0.505, width * 0.34, height * 0.38), (0.04, 0.28, 0.12), (0.98, 0.97, 0.90), points_per_meter),
            _sample_box_surface((length * 0.505, -width * 0.34, height * 0.38), (0.04, 0.28, 0.12), (0.98, 0.97, 0.90), points_per_meter),
            # Ground contact shadow plane (z ~ 0.02m)
            _sample_box_surface((0.0, 0.0, 0.02), (length * 1.06, width * 1.08, 0.01), (0.08, 0.08, 0.09), max(12, points_per_meter // 2)),
        ]
        # Four wheels
        wheel_x = length * 0.31
        wheel_y = width * 0.47
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                parts.append(
                    _sample_box_surface(
                        (sx * wheel_x, sy * wheel_y, 0.33),
                        (0.66, 0.24, 0.66),
                        (0.11, 0.11, 0.12),
                        max(12, points_per_meter // 2),
                    )
                )

        pts = torch.cat([p[0] for p in parts], dim=0)
        nrms = torch.cat([p[1] for p in parts], dim=0)
        cols = torch.cat([p[2] for p in parts], dim=0)

        # Simple directional sky+sun Lambertian shading in ego frame so the 3D form reads cleanly.
        light_dir = torch.tensor([0.35, 0.25, 0.90], dtype=torch.float32)
        light_dir = light_dir / light_dir.norm()
        ndotl = (nrms @ light_dir).clamp(min=0.0)
        shaded_rgb = (cols * (0.45 + 0.55 * ndotl[:, None])).clamp(0.0, 1.0)

        self.register_buffer("ego_points", pts, persistent=False)  # [N, 3]
        self.register_buffer("ego_rgb", shaded_rgb, persistent=False)  # [N, 3]

    @torch.no_grad()
    def forward(
        self,
        projection: ProjectionResult,
        tgt_intrinsics: torch.Tensor,
        tgt_c2e: torch.Tensor,
        *,
        value_range: tuple[float, float] = (0.0, 1.0),
    ) -> ProjectionResult:
        """Composite the 3D ego vehicle into ``projection`` using exact z-buffer occlusion."""
        coarse_rgb = projection.coarse_rgb.clone()
        hole_mask = projection.hole_mask.clone()
        target_depth = projection.target_depth.clone()

        bsz, _, h_tgt, w_tgt = coarse_rgb.shape
        device = coarse_rgb.device
        dtype = coarse_rgb.dtype

        tgt_e2c = torch.linalg.inv(tgt_c2e.float())  # [B, 4, 4]
        pts = self.ego_points.to(device=device, dtype=torch.float32)  # [N, 3]
        rgb = self.ego_rgb.to(device=device, dtype=dtype)  # [N, 3]
        if value_range == (-1.0, 1.0):
            rgb = rgb * 2.0 - 1.0

        offsets = [
            (dy, dx)
            for dy in range(-self.splat_radius_px, self.splat_radius_px + 1)
            for dx in range(-self.splat_radius_px, self.splat_radius_px + 1)
        ]

        for b in range(bsz):
            rot = tgt_e2c[b, :3, :3]
            trans = tgt_e2c[b, :3, 3]
            pts_t = pts @ rot.T + trans[None, :]
            z_t = pts_t[:, 2]
            in_front = z_t > 0.2
            if not in_front.any():
                continue

            k_tgt = tgt_intrinsics[b].float()
            u_i = torch.round(k_tgt[0, 0] * (pts_t[:, 0] / z_t.clamp_min(1e-6)) + k_tgt[0, 2]).long()
            v_i = torch.round(k_tgt[1, 1] * (pts_t[:, 1] / z_t.clamp_min(1e-6)) + k_tgt[1, 2]).long()

            valid = in_front & (u_i >= 0) & (u_i < w_tgt) & (v_i >= 0) & (v_i < h_tgt)
            if not valid.any():
                continue

            z_sel = z_t[valid]
            u_sel = u_i[valid]
            v_sel = v_i[valid]
            rgb_sel = rgb[valid]

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
                existing_d = canvas_depth[flat_idx]
                existing_hole = canvas_hole[flat_idx] > 0.5
                closer = existing_hole | (z_sorted[ok] <= existing_d)
                if not closer.any():
                    continue
                win_idx = flat_idx[closer]
                canvas_rgb[:, win_idx] = rgb_sorted[ok][closer].T
                canvas_depth[win_idx] = z_sorted[ok][closer]
                canvas_hole[win_idx] = 0.0

        return ProjectionResult(
            coarse_rgb=coarse_rgb,
            hole_mask=hole_mask,
            target_depth=target_depth,
        )
