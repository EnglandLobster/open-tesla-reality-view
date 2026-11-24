"""Watertight 3D fastback sedan triangle-mesh Ego-Vehicle compositor (pure PyTorch GPU)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from open_reality_view.geometry.projector import ProjectionResult


def _build_fastback_sedan_mesh(
    ego_box_lwh: tuple[float, float, float],
    body_color: tuple[float, float, float],
    glass_color: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a sleek, watertight sculpted fastback sedan triangle mesh in ego coordinates.

    Ego frame: ``+X`` forward, ``+Y`` left, ``+Z`` up, resting on ``z = 0`` centered at ``(0, 0)``.

    Returns:
        ``(vertices [V, 3], faces [F, 3], face_colors [F, 3], face_shininess [F])``.
    """
    length, width, height = ego_box_lwh
    hl = length * 0.5
    hw = width * 0.5

    verts_list: list[list[float]] = []
    faces_list: list[list[int]] = []
    cols_list: list[tuple[float, float, float]] = []
    shine_list: list[float] = []

    def _add_quad(
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
        p3: tuple[float, float, float],
        color: tuple[float, float, float],
        shininess: float = 0.35,
    ) -> None:
        base = len(verts_list)
        verts_list.extend([list(p0), list(p1), list(p2), list(p3)])
        faces_list.append([base, base + 1, base + 2])
        faces_list.append([base, base + 2, base + 3])
        cols_list.extend([color, color])
        shine_list.extend([shininess, shininess])

    def _add_box(
        cx: float,
        cy: float,
        cz: float,
        lx: float,
        ly: float,
        lz: float,
        color: tuple[float, float, float],
        shininess: float = 0.15,
    ) -> None:
        x0, x1 = cx - lx * 0.5, cx + lx * 0.5
        y0, y1 = cy - ly * 0.5, cy + ly * 0.5
        z0, z1 = cz - lz * 0.5, cz + lz * 0.5
        _add_quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), color, shininess)  # top
        _add_quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1), color, shininess)  # left
        _add_quad((x1, y0, z0), (x0, y0, z0), (x0, y0, z1), (x1, y0, z1), color, shininess)  # right
        _add_quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), color, shininess)  # front
        _add_quad((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), color, shininess)  # rear

    # Longitudinal sculpted body cross-sections: (x_frac, half_width_frac, z_bottom_frac, z_belt_frac, z_roof_frac, roof_w_frac)
    # Runs from rear bumper (-1.0) to front bumper (+1.0) with sleek fastback roof taper.
    sections = [
        (-1.00, 0.84, 0.16, 0.56, 0.56, 0.72),  # rear bumper tip
        (-0.92, 0.94, 0.14, 0.64, 0.66, 0.78),  # trunk lip
        (-0.68, 0.98, 0.12, 0.66, 0.84, 0.74),  # rear fastback glass base
        (-0.42, 1.00, 0.12, 0.65, 0.98, 0.68),  # rear roof peak
        (0.02, 1.00, 0.12, 0.63, 1.00, 0.68),   # mid roof peak
        (0.24, 0.99, 0.12, 0.61, 0.94, 0.69),   # windshield top
        (0.52, 0.97, 0.12, 0.58, 0.60, 0.78),   # windshield base / cowl
        (0.84, 0.92, 0.13, 0.50, 0.50, 0.78),   # hood nose
        (1.00, 0.78, 0.15, 0.36, 0.36, 0.66),   # front bumper tip
    ]

    # Loft lower body shell + tapered cabin greenhouse between adjacent stations
    for i in range(len(sections) - 1):
        x0f, w0f, zb0f, zm0f, zt0f, rw0f = sections[i]
        x1f, w1f, zb1f, zm1f, zt1f, rw1f = sections[i + 1]
        x0, x1 = x0f * hl, x1f * hl
        w0, w1 = w0f * hw, w1f * hw
        rw0, rw1 = rw0f * hw, rw1f * hw
        zb0, zb1 = zb0f * height, zb1f * height
        zm0, zm1 = zm0f * height, zm1f * height
        zt0, zt1 = zt0f * height, zt1f * height

        # Lower body left (+Y) and right (-Y) flanks
        _add_quad((x0, w0, zb0), (x1, w1, zb1), (x1, w1, zm1), (x0, w0, zm0), body_color, 0.45)
        _add_quad((x1, -w1, zb1), (x0, -w0, zb0), (x0, -w0, zm0), (x1, -w1, zm1), body_color, 0.45)

        # Upper greenhouse / hood / trunk panels
        is_glass_side = 1 <= i <= 5
        is_windshield_or_rear = i in (1, 2, 5)
        side_col = glass_color if is_glass_side else body_color
        top_col = glass_color if is_windshield_or_rear else body_color
        top_shine = 0.75 if is_windshield_or_rear else 0.50

        if zt0 > zm0 + 0.01 or zt1 > zm1 + 0.01:
            _add_quad((x0, w0, zm0), (x1, w1, zm1), (x1, rw1, zt1), (x0, rw0, zt0), side_col, 0.70)
            _add_quad((x1, -w1, zm1), (x0, -w0, zm0), (x0, -rw0, zt0), (x1, -rw1, zt1), side_col, 0.70)
            _add_quad((x0, -rw0, zt0), (x1, -rw1, zt1), (x1, rw1, zt1), (x0, rw0, zt0), top_col, top_shine)
        else:
            _add_quad((x0, -w0, zm0), (x1, -w1, zm1), (x1, w1, zm1), (x0, w0, zm0), body_color, 0.50)

    # Cap rear and front bumpers cleanly
    xr, wr, zbr, zmr, _, _ = sections[0]
    _add_quad((xr * hl, wr * hw, zbr * height), (xr * hl, -wr * hw, zbr * height), (xr * hl, -wr * hw, zmr * height), (xr * hl, wr * hw, zmr * height), body_color, 0.35)
    xf, wf, zbf, zmf, _, _ = sections[-1]
    _add_quad((xf * hl, -wf * hw, zbf * height), (xf * hl, wf * hw, zbf * height), (xf * hl, wf * hw, zmf * height), (xf * hl, -wf * hw, zmf * height), body_color, 0.35)

    # Glowing tail-light bar + license plate recess + front headlights
    _add_box(-hl - 0.01, 0.0, height * 0.48, 0.03, width * 0.74, 0.09, (0.92, 0.10, 0.08), 0.90)
    _add_box(-hl - 0.015, 0.0, height * 0.33, 0.02, 0.36, 0.14, (0.94, 0.94, 0.96), 0.10)
    _add_box(hl + 0.01, width * 0.31, height * 0.33, 0.03, 0.28, 0.09, (0.98, 0.97, 0.90), 0.90)
    _add_box(hl + 0.01, -width * 0.31, height * 0.33, 0.03, 0.28, 0.09, (0.98, 0.97, 0.90), 0.90)

    # Four dark rubber wheels tucked under wheel arches
    wheel_x = length * 0.30
    wheel_y = width * 0.45
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            _add_box(sx * wheel_x, sy * wheel_y, 0.33, 0.66, 0.24, 0.66, (0.09, 0.09, 0.10), 0.12)

    verts = torch.tensor(verts_list, dtype=torch.float32)
    faces = torch.tensor(faces_list, dtype=torch.long)
    colors = torch.tensor(cols_list, dtype=torch.float32)
    shine = torch.tensor(shine_list, dtype=torch.float32)
    return verts, faces, colors, shine


class EgoVehicleCompositor(nn.Module):
    """Composites a watertight, smoothly shaded 3D ego sedan + soft contact shadow via z-buffer rasterization."""

    def __init__(
        self,
        ego_box_lwh: tuple[float, float, float] = (4.72, 1.85, 1.44),
        body_color: tuple[float, float, float] = (0.86, 0.88, 0.91),
        glass_color: tuple[float, float, float] = (0.11, 0.14, 0.19),
        points_per_meter: int = 22,  # kept for backwards API compatibility
        splat_radius_px: int = 1,
    ) -> None:
        super().__init__()
        _ = (points_per_meter, splat_radius_px)
        self.ego_box_lwh = ego_box_lwh
        verts, faces, face_colors, face_shine = _build_fastback_sedan_mesh(
            ego_box_lwh=ego_box_lwh,
            body_color=body_color,
            glass_color=glass_color,
        )
        self.register_buffer("ego_points", verts, persistent=False)  # [V, 3] (shiftable for rear-axle alignment)
        self.register_buffer("ego_faces", faces, persistent=False)  # [F, 3]
        self.register_buffer("ego_face_colors", face_colors, persistent=False)  # [F, 3]
        self.register_buffer("ego_face_shine", face_shine, persistent=False)  # [F]

    @torch.no_grad()
    def forward(
        self,
        projection: ProjectionResult,
        tgt_intrinsics: torch.Tensor,
        tgt_c2e: torch.Tensor,
        *,
        value_range: tuple[float, float] = (0.0, 1.0),
    ) -> ProjectionResult:
        """Rasterize the 3D ego sedan & soft contact shadow into ``projection`` with exact z-buffer."""
        coarse_rgb = projection.coarse_rgb.clone()
        hole_mask = projection.hole_mask.clone()
        target_depth = projection.target_depth.clone()

        bsz, _, h_tgt, w_tgt = coarse_rgb.shape
        device = coarse_rgb.device
        dtype = coarse_rgb.dtype

        tgt_e2c = torch.linalg.inv(tgt_c2e.float())
        verts_ego = self.ego_points.to(device=device, dtype=torch.float32)
        faces = self.ego_faces.to(device=device)
        face_cols = self.ego_face_colors.to(device=device, dtype=torch.float32)
        face_shine = self.ego_face_shine.to(device=device, dtype=torch.float32)

        # Compute world/ego face normals and centroids for smooth directional + sky-bounce shading
        v0_e = verts_ego[faces[:, 0]]
        v1_e = verts_ego[faces[:, 1]]
        v2_e = verts_ego[faces[:, 2]]
        normals_e = F.normalize(torch.linalg.cross(v1_e - v0_e, v2_e - v0_e, dim=-1), dim=-1)
        centers_e = (v0_e + v1_e + v2_e) / 3.0

        ego_center_xy = verts_ego[:, :2].mean(dim=0)
        hl_s = self.ego_box_lwh[0] * 0.56
        hw_s = self.ego_box_lwh[1] * 0.58

        sun_dir = F.normalize(torch.tensor([0.35, -0.25, 0.90], device=device), dim=0)
        sky_tint = torch.tensor([0.62, 0.72, 0.88], device=device)

        for b in range(bsz):
            k = tgt_intrinsics[b].float()
            rot = tgt_e2c[b, :3, :3]
            trans = tgt_e2c[b, :3, 3]
            cam_pos_e = tgt_c2e[b, :3, 3].float()

            # 1. Soft elliptical ground contact shadow under the car on z = 0
            # Evaluate on pixels where ground is visible below the car footprint.
            v_cam = verts_ego @ rot.T + trans[None, :]
            if (v_cam[:, 2] <= 0.25).all():
                continue

            u_v = k[0, 0] * (v_cam[:, 0] / v_cam[:, 2].clamp_min(0.25)) + k[0, 2]
            v_v = k[1, 1] * (v_cam[:, 1] / v_cam[:, 2].clamp_min(0.25)) + k[1, 2]

            x_min = max(0, int(torch.floor(u_v.min()).item()) - 12)
            x_max = min(w_tgt, int(torch.ceil(u_v.max()).item()) + 13)
            y_min = max(0, int(torch.floor(v_v.min()).item()) - 8)
            y_max = min(h_tgt, int(torch.ceil(v_v.max()).item()) + 16)
            if x_max <= x_min or y_max <= y_min:
                continue

            py = torch.arange(y_min, y_max, device=device, dtype=torch.float32) + 0.5
            px = torch.arange(x_min, x_max, device=device, dtype=torch.float32) + 0.5
            gy, gx = torch.meshgrid(py, px, indexing="ij")  # [Hb, Wb]

            # Soft shadow on z=0 plane inside bounding box
            ray_c = torch.stack([(gx - k[0, 2]) / k[0, 0], (gy - k[1, 2]) / k[1, 1], torch.ones_like(gx)], dim=-1)
            ray_e = ray_c @ tgt_c2e[b, :3, :3].float().T
            hit_g = ray_e[..., 2] < -1e-4
            t_g = torch.where(hit_g, -cam_pos_e[2] / ray_e[..., 2].clamp(max=-1e-4), torch.full_like(gx, 1e6))
            gx_e = cam_pos_e[0] + t_g * ray_e[..., 0] - ego_center_xy[0]
            gy_e = cam_pos_e[1] + t_g * ray_e[..., 1] - ego_center_xy[1]
            # Rounded rectangle SDF for realistic soft under-car shadow
            dx_s = (gx_e.abs() - hl_s * 0.78).clamp_min(0.0) / (hl_s * 0.25)
            dy_s = (gy_e.abs() - hw_s * 0.72).clamp_min(0.0) / (hw_s * 0.30)
            shadow_strength = torch.exp(-2.6 * (dx_s * dx_s + dy_s * dy_s)) * 0.62
            shadow_strength = torch.where(hit_g & (t_g > 0.5), shadow_strength, torch.zeros_like(shadow_strength))
            if value_range == (-1.0, 1.0):
                patch_01 = (coarse_rgb[b, :, y_min:y_max, x_min:x_max].float() + 1.0) * 0.5
                patch_01 = patch_01 * (1.0 - shadow_strength[None])
                coarse_rgb[b, :, y_min:y_max, x_min:x_max] = (patch_01 * 2.0 - 1.0).to(dtype=dtype)
            else:
                coarse_rgb[b, :, y_min:y_max, x_min:x_max] = (
                    coarse_rgb[b, :, y_min:y_max, x_min:x_max].float() * (1.0 - shadow_strength[None])
                ).to(dtype=dtype)

            # 2. Compute view-dependent face shading (Lambert + hemisphere sky ambient + Fresnel rim)
            view_e = F.normalize(cam_pos_e[None, :] - centers_e, dim=-1)
            # Ensure normal faces the camera (double-sided safety for thin bumper/light panels)
            ndotv = (normals_e * view_e).sum(dim=-1)
            n_eff = torch.where(ndotv[:, None] < 0.0, -normals_e, normals_e)
            ndotv = ndotv.abs().clamp(0.0, 1.0)
            ndotl = (n_eff @ sun_dir).clamp(0.0, 1.0)
            hemi = 0.5 + 0.5 * n_eff[:, 2].clamp(-1.0, 1.0)
            fresnel = ((1.0 - ndotv) ** 3) * face_shine
            half_vec = F.normalize(sun_dir[None, :] + view_e, dim=-1)
            spec = ((n_eff * half_vec).sum(dim=-1).clamp(0.0, 1.0) ** 24) * face_shine * 0.45

            shaded = (
                face_cols * (0.32 + 0.33 * hemi[:, None] + 0.45 * ndotl[:, None])
                + fresnel[:, None] * sky_tint[None, :] * 0.45
                + spec[:, None]
            ).clamp(0.0, 1.0)
            if value_range == (-1.0, 1.0):
                shaded = shaded * 2.0 - 1.0

            # 3. Exact vectorized barycentric triangle rasterization over tight [Hb, Wb] bbox
            p0 = torch.stack([u_v[faces[:, 0]], v_v[faces[:, 0]]], dim=-1)  # [F, 2]
            p1 = torch.stack([u_v[faces[:, 1]], v_v[faces[:, 1]]], dim=-1)
            p2 = torch.stack([u_v[faces[:, 2]], v_v[faces[:, 2]]], dim=-1)
            z0 = v_cam[faces[:, 0], 2]
            z1 = v_cam[faces[:, 1], 2]
            z2 = v_cam[faces[:, 2], 2]

            valid_tri = (z0 > 0.25) & (z1 > 0.25) & (z2 > 0.25)
            p0, p1, p2 = p0[valid_tri], p1[valid_tri], p2[valid_tri]
            z0, z1, z2 = z0[valid_tri], z1[valid_tri], z2[valid_tri]
            shaded_v = shaded[valid_tri]

            area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
            non_deg = area.abs() > 1e-3
            if not non_deg.any():
                continue
            p0, p1, p2 = p0[non_deg], p1[non_deg], p2[non_deg]
            z0, z1, z2 = z0[non_deg], z1[non_deg], z2[non_deg]
            shaded_v = shaded_v[non_deg]
            inv_area = 1.0 / area[non_deg]

            hb, wb = y_max - y_min, x_max - x_min
            best_z = target_depth[b, 0, y_min:y_max, x_min:x_max].clone()
            best_hole = hole_mask[b, 0, y_min:y_max, x_min:x_max].clone()
            best_rgb = coarse_rgb[b, :, y_min:y_max, x_min:x_max].float().clone()
            # Treat existing hole pixels as infinite depth so ego mesh fills them cleanly.
            z_buf = torch.where(best_hole > 0.5, torch.full_like(best_z, 1e6), best_z)

            px_f = gx.reshape(1, hb * wb)
            py_f = gy.reshape(1, hb * wb)
            z_flat = z_buf.reshape(hb * wb)
            rgb_flat = best_rgb.reshape(3, hb * wb)
            hole_flat = best_hole.reshape(hb * wb)

            # Evaluate in chunks of 64 triangles to keep peak memory tiny
            num_tris = p0.shape[0]
            for s in range(0, num_tris, 64):
                e = min(num_tris, s + 64)
                a0, a1, a2 = p0[s:e], p1[s:e], p2[s:e]
                ia = inv_area[s:e, None]

                w0 = ((a1[:, 0:1] - px_f) * (a2[:, 1:2] - py_f) - (a1[:, 1:2] - py_f) * (a2[:, 0:1] - px_f)) * ia
                w1 = ((a2[:, 0:1] - px_f) * (a0[:, 1:2] - py_f) - (a2[:, 1:2] - py_f) * (a0[:, 0:1] - px_f)) * ia
                w2 = 1.0 - w0 - w1

                inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
                if not inside.any():
                    continue
                # Perspective-correct depth interpolation
                inv_z = w0 / z0[s:e, None] + w1 / z1[s:e, None] + w2 / z2[s:e, None]
                z_pix = torch.where(inside, 1.0 / inv_z.clamp_min(1e-6), torch.full_like(w0, 1e6))
                min_z_chunk, best_tri = z_pix.min(dim=0)
                closer = min_z_chunk < z_flat
                if closer.any():
                    z_flat[closer] = min_z_chunk[closer]
                    hole_flat[closer] = 0.0
                    rgb_flat[:, closer] = shaded_v[s:e][best_tri[closer]].T

            coarse_rgb[b, :, y_min:y_max, x_min:x_max] = rgb_flat.reshape(3, hb, wb).to(dtype=dtype)
            target_depth[b, 0, y_min:y_max, x_min:x_max] = torch.where(
                hole_flat.reshape(hb, wb) > 0.5, torch.zeros_like(best_z), z_flat.reshape(hb, wb)
            )
            hole_mask[b, 0, y_min:y_max, x_min:x_max] = hole_flat.reshape(hb, wb)

        return ProjectionResult(
            coarse_rgb=coarse_rgb,
            hole_mask=hole_mask,
            target_depth=target_depth,
        )
