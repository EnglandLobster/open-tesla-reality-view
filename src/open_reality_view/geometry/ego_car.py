"""Smooth-shaded fastback sedan Ego-Vehicle compositor with real NuRec skybox IBL reflections."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from open_reality_view.geometry.projector import ProjectionResult


def _sample_cubemap_dirs(textures: torch.Tensor, dirs_w: torch.Tensor) -> torch.Tensor:
    """Sample a NuRec Z-up 6-face cubemap ``[6, S, S, 3]`` at arbitrary world unit vectors ``[M, 3]``."""
    if textures.ndim == 5:
        textures = textures[0]
    tex_chw = textures.permute(0, 3, 1, 2).float()
    dirs_w = F.normalize(dirs_w.float(), dim=-1)

    # NuRec Z-up (+X fwd, +Y left, +Z up) -> OpenGL Y-up cubemap axes
    dx = -dirs_w[:, 1]
    dy = dirs_w[:, 2]
    dz = dirs_w[:, 0]
    ax, ay, az = dx.abs(), dy.abs(), dz.abs()

    n = dirs_w.shape[0]
    device = dirs_w.device
    face_idx = torch.zeros(n, device=device, dtype=torch.long)
    uc = torch.zeros(n, device=device, dtype=torch.float32)
    vc = torch.zeros(n, device=device, dtype=torch.float32)

    m_px = (ax >= ay) & (ax >= az) & (dx >= 0)
    m_nx = (ax >= ay) & (ax >= az) & (dx < 0)
    m_py = (ay > ax) & (ay >= az) & (dy >= 0)
    m_ny = (ay > ax) & (ay >= az) & (dy < 0)
    m_pz = (az > ax) & (az > ay) & (dz >= 0)
    m_nz = (az > ax) & (az > ay) & (dz < 0)

    eps = 1e-6
    face_idx[m_px], uc[m_px], vc[m_px] = 0, -dz[m_px] / ax[m_px].clamp_min(eps), -dy[m_px] / ax[m_px].clamp_min(eps)
    face_idx[m_nx], uc[m_nx], vc[m_nx] = 1, dz[m_nx] / ax[m_nx].clamp_min(eps), -dy[m_nx] / ax[m_nx].clamp_min(eps)
    face_idx[m_py], uc[m_py], vc[m_py] = 2, dx[m_py] / ay[m_py].clamp_min(eps), dz[m_py] / ay[m_py].clamp_min(eps)
    face_idx[m_ny], uc[m_ny], vc[m_ny] = 3, dx[m_ny] / ay[m_ny].clamp_min(eps), -dz[m_ny] / ay[m_ny].clamp_min(eps)
    face_idx[m_pz], uc[m_pz], vc[m_pz] = 4, dx[m_pz] / az[m_pz].clamp_min(eps), -dy[m_pz] / az[m_pz].clamp_min(eps)
    face_idx[m_nz], uc[m_nz], vc[m_nz] = 5, -dx[m_nz] / az[m_nz].clamp_min(eps), -dy[m_nz] / az[m_nz].clamp_min(eps)

    out = torch.zeros((n, 3), device=device, dtype=torch.float32)
    for f_id in range(6):
        mask = face_idx == f_id
        if not mask.any():
            continue
        grid = torch.stack([uc[mask], vc[mask]], dim=-1).view(1, 1, -1, 2)
        sampled = F.grid_sample(tex_chw[f_id : f_id + 1], grid, mode="bilinear", align_corners=False)
        out[mask] = sampled[0, :, 0, :].T
    return out.clamp(0.0, 1.0)


def _catmull_rom_sample(pts: list[tuple[float, ...]], samples_per_seg: int) -> list[tuple[float, ...]]:
    """Smoothly interpolate control stations along the car length."""
    arr = torch.tensor(pts, dtype=torch.float32)
    n = arr.shape[0]
    out: list[tuple[float, ...]] = []
    for i in range(n - 1):
        p0 = arr[max(0, i - 1)]
        p1 = arr[i]
        p2 = arr[i + 1]
        p3 = arr[min(n - 1, i + 2)]
        steps = samples_per_seg if i < n - 2 else samples_per_seg + 1
        for s in range(steps):
            t = s / float(samples_per_seg)
            t2, t3 = t * t, t * t * t
            val = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(tuple(float(v) for v in val))
    return out


def _build_smooth_sedan_mesh(
    ego_box_lwh: tuple[float, float, float],
    body_color: tuple[float, float, float],
    glass_color: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a smoothly curved Tesla-style fastback sedan with per-vertex smooth normals.

    Returns:
        ``(vertices [V, 3], vert_normals [V, 3], faces [F, 3], face_colors [F, 3],
           face_fresnel [F], face_emissive [F])``.
    """
    length, width, height = ego_box_lwh
    hl, hw = length * 0.5, width * 0.5

    # Control stations: (x_frac, half_width_frac, z_bot, z_belt, z_roof, roof_w_frac)
    ctrl = [
        (-1.00, 0.80, 0.16, 0.54, 0.54, 0.66),  # rear bumper bottom
        (-0.96, 0.91, 0.14, 0.63, 0.65, 0.74),  # rear bumper lip
        (-0.84, 0.96, 0.13, 0.67, 0.72, 0.76),  # ducktail spoiler
        (-0.62, 0.99, 0.12, 0.67, 0.88, 0.72),  # rear fastback glass
        (-0.36, 1.00, 0.12, 0.66, 0.99, 0.67),  # rear roof crown
        (-0.04, 1.00, 0.12, 0.64, 1.00, 0.67),  # roof peak
        (0.24, 0.99, 0.12, 0.61, 0.93, 0.68),   # windshield top
        (0.50, 0.97, 0.12, 0.57, 0.59, 0.77),   # cowl / hood base
        (0.80, 0.92, 0.13, 0.49, 0.49, 0.76),   # curved front hood
        (0.96, 0.82, 0.15, 0.39, 0.39, 0.66),   # nose cone
        (1.00, 0.64, 0.18, 0.30, 0.30, 0.52),   # front lip
    ]
    stations = _catmull_rom_sample(ctrl, samples_per_seg=3)  # ~31 smooth longitudinal rings!

    # Cross-section profile angles from right skirt (-Y) over the curved roof to left skirt (+Y).
    # 13 points around each cross-section ring for silky-smooth shoulders & curved roof!
    def _ring_points(st: tuple[float, ...]) -> list[tuple[float, float, float]]:
        xf, wf, zbf, zmf, ztf, rwf = st
        x = xf * hl
        w = wf * hw
        rw = rwf * hw
        zb = zbf * height
        zm = zmf * height
        zt = max(zm, ztf * height)
        crown = (zt - zm) * 0.08 + 0.015  # subtle domed roof/hood curvature
        return [
            (x, -w * 0.88, zb),                         # 0: right lower rocker
            (x, -w * 0.98, zb + (zm - zb) * 0.35),      # 1: right lower door curve
            (x, -w * 1.00, zb + (zm - zb) * 0.72),      # 2: right door shoulder
            (x, -w * 0.96, zm),                         # 3: right beltline
            (x, -(w * 0.56 + rw * 0.44), zm + (zt - zm) * 0.55),  # 4: right side window mid
            (x, -rw * 1.00, zt - crown * 0.2),          # 5: right roof rail
            (x, 0.0, zt + crown),                       # 6: center crown
            (x, rw * 1.00, zt - crown * 0.2),           # 7: left roof rail
            (x, (w * 0.56 + rw * 0.44), zm + (zt - zm) * 0.55),   # 8: left side window mid
            (x, w * 0.96, zm),                          # 9: left beltline
            (x, w * 1.00, zb + (zm - zb) * 0.72),       # 10: left door shoulder
            (x, w * 0.98, zb + (zm - zb) * 0.35),       # 11: left lower door curve
            (x, w * 0.88, zb),                          # 12: left lower rocker
        ]

    rings = [_ring_points(st) for st in stations]
    ring_n = len(rings[0])
    verts_list: list[list[float]] = []
    for ring in rings:
        for pt in ring:
            verts_list.append(list(pt))

    faces_list: list[list[int]] = []
    cols_list: list[tuple[float, float, float]] = []
    fres_list: list[float] = []
    emis_list: list[float] = []

    skirt_color = (0.10, 0.11, 0.13)

    for i in range(len(rings) - 1):
        xf_mid = 0.5 * (stations[i][0] + stations[i + 1][0])
        in_cabin_x = -0.80 <= xf_mid <= 0.48
        in_glass_top = (-0.80 <= xf_mid <= -0.34) or (0.16 <= xf_mid <= 0.48) or (-0.34 < xf_mid < 0.16)
        # Model 3/Y has a full-length tinted panoramic glass roof from windshield to rear fastback!
        for j in range(ring_n - 1):
            v00 = i * ring_n + j
            v10 = (i + 1) * ring_n + j
            v11 = (i + 1) * ring_n + (j + 1)
            v01 = i * ring_n + (j + 1)
            faces_list.append([v00, v10, v11])
            faces_list.append([v00, v11, v01])

            if j in (0, ring_n - 2):
                col, fr = skirt_color, 0.15
            elif in_cabin_x and j in (3, 4, 7, 8):
                col, fr = glass_color, 0.85
            elif in_glass_top and j in (5, 6):
                # Panoramic tinted glass roof + windshield + fastback rear glass
                col, fr = glass_color, 0.90
            else:
                col, fr = body_color, 0.55

            cols_list.extend([col, col])
            fres_list.extend([fr, fr])
            emis_list.extend([0.0, 0.0])

    # Cap rear and front end rings cleanly around their center point
    def _cap_ring(ring_idx: int, flip: bool, col: tuple[float, float, float]) -> None:
        pts = rings[ring_idx]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        c_idx = len(verts_list)
        verts_list.append([cx, cy, cz])
        for j in range(ring_n - 1):
            a = ring_idx * ring_n + j
            b = ring_idx * ring_n + j + 1
            faces_list.append([c_idx, b, a] if flip else [c_idx, a, b])
            cols_list.append(col)
            fres_list.append(0.40)
            emis_list.append(0.0)

    _cap_ring(0, flip=True, col=body_color)
    _cap_ring(len(rings) - 1, flip=False, col=body_color)

    # Smooth vertex normals for the body shell (before adding crisp hard-edged details)
    n_body_verts = len(verts_list)
    n_body_faces = len(faces_list)

    def _add_detail_box(
        cx: float,
        cy: float,
        cz: float,
        lx: float,
        ly: float,
        lz: float,
        col: tuple[float, float, float],
        fres: float = 0.25,
        emis: float = 0.0,
    ) -> None:
        x0, x1 = cx - lx * 0.5, cx + lx * 0.5
        y0, y1 = cy - ly * 0.5, cy + ly * 0.5
        z0, z1 = cz - lz * 0.5, cz + lz * 0.5
        quads = [
            ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),
            ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)),
            ((x1, y0, z0), (x0, y0, z0), (x0, y0, z1), (x1, y0, z1)),
            ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)),
            ((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)),
        ]
        for q in quads:
            b = len(verts_list)
            verts_list.extend([list(p) for p in q])
            faces_list.append([b, b + 1, b + 2])
            faces_list.append([b, b + 2, b + 3])
            cols_list.extend([col, col])
            fres_list.extend([fres, fres])
            emis_list.extend([emis, emis])

    def _add_wheel(wx: float, wy: float, radius: float = 0.33, tire_w: float = 0.21, segs: int = 16) -> None:
        y_in = wy - tire_w * 0.5
        y_out = wy + tire_w * 0.5
        sign = 1.0 if wy > 0 else -1.0
        y_outer = y_out if sign > 0 else y_in
        tire_col = (0.06, 0.06, 0.07)
        rim_col = (0.34, 0.37, 0.42)
        hub_col = (0.14, 0.15, 0.18)

        base_in = len(verts_list)
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            verts_list.append([wx + radius * math.cos(ang), y_in, radius + radius * math.sin(ang)])
        base_out = len(verts_list)
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            verts_list.append([wx + radius * math.cos(ang), y_out, radius + radius * math.sin(ang)])
        for s in range(segs):
            sn = (s + 1) % segs
            faces_list.append([base_in + s, base_out + s, base_out + sn])
            faces_list.append([base_in + s, base_out + sn, base_in + sn])
            cols_list.extend([tire_col, tire_col])
            fres_list.extend([0.08, 0.08])
            emis_list.extend([0.0, 0.0])

        # Outer dark aero alloy rim disc + darker center hub
        c_rim = len(verts_list)
        verts_list.append([wx, y_outer + sign * 0.008, radius])
        r_rim = radius * 0.64
        r_hub = radius * 0.22
        ring_rim = len(verts_list)
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            verts_list.append([wx + r_rim * math.cos(ang), y_outer + sign * 0.008, radius + r_rim * math.sin(ang)])
        for s in range(segs):
            sn = (s + 1) % segs
            faces_list.append([c_rim, ring_rim + s, ring_rim + sn])
            cols_list.append(rim_col)
            fres_list.append(0.55)
            emis_list.append(0.0)

        c_hub = len(verts_list)
        verts_list.append([wx, y_outer + sign * 0.012, radius])
        ring_hub = len(verts_list)
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            verts_list.append([wx + r_hub * math.cos(ang), y_outer + sign * 0.012, radius + r_hub * math.sin(ang)])
        for s in range(segs):
            sn = (s + 1) % segs
            faces_list.append([c_hub, ring_hub + s, ring_hub + sn])
            cols_list.append(hub_col)
            fres_list.append(0.30)
            emis_list.append(0.0)

    # 4 rounded wheels tucked flush inside the wheel arches
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            _add_wheel(sx * length * 0.30, sy * (hw - 0.11))

    # Sleek aerodynamic side mirrors tucked tightly to the A-pillar shoulder
    for sy in (-1.0, 1.0):
        _add_detail_box(length * 0.16, sy * (hw + 0.025), height * 0.61, 0.15, 0.11, 0.065, body_color, 0.60)
        _add_detail_box(length * 0.16, sy * (hw + 0.005), height * 0.585, 0.07, 0.07, 0.035, (0.12, 0.13, 0.15), 0.20)

    # Flush recessed automotive LED taillights (strictly inside tapered rear bumper width!)
    _add_detail_box(-hl + 0.015, width * 0.24, height * 0.545, 0.07, width * 0.145, 0.052, (0.78, 0.06, 0.06), 0.50, 0.55)
    _add_detail_box(-hl + 0.015, -width * 0.24, height * 0.545, 0.07, width * 0.145, 0.052, (0.78, 0.06, 0.06), 0.50, 0.55)
    _add_detail_box(-hl + 0.012, 0.0, height * 0.545, 0.055, width * 0.36, 0.020, (0.60, 0.04, 0.05), 0.55, 0.35)
    # Recessed lower rear diffuser & subtle license plate
    _add_detail_box(-hl + 0.015, 0.0, height * 0.21, 0.06, width * 0.62, 0.12, (0.09, 0.10, 0.11), 0.15, 0.0)
    _add_detail_box(-hl + 0.008, 0.0, height * 0.37, 0.04, 0.30, 0.11, (0.83, 0.85, 0.88), 0.15, 0.0)

    # Front LED headlights (recessed cleanly below sloping front hood line!)
    for sy in (-1.0, 1.0):
        _add_detail_box(hl * 0.90, sy * width * 0.26, height * 0.32, 0.16, 0.22, 0.045, (0.96, 0.96, 0.92), 0.50, 0.65)

    verts = torch.tensor(verts_list, dtype=torch.float32)
    faces = torch.tensor(faces_list, dtype=torch.long)
    colors = torch.tensor(cols_list, dtype=torch.float32)
    fresnel = torch.tensor(fres_list, dtype=torch.float32)
    emissive = torch.tensor(emis_list, dtype=torch.float32)

    # Compute smooth area-weighted per-vertex normals for the sculpted body shell,
    # and sharp face normals for detail attachments.
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_n = torch.linalg.cross(v1 - v0, v2 - v0, dim=-1)
    # Orient outward away from car center (0, 0, height*0.42)
    car_ctr = torch.tensor([0.0, 0.0, height * 0.42])
    tri_ctr = (v0 + v1 + v2) / 3.0 - car_ctr
    flip = (face_n * tri_ctr).sum(dim=-1) < 0
    face_n = torch.where(flip[:, None], -face_n, face_n)

    vert_normals = torch.zeros_like(verts)
    for corner in range(3):
        vert_normals.index_add_(0, faces[:n_body_faces, corner], face_n[:n_body_faces])
    # Detail vertices get their own face normals
    if faces.shape[0] > n_body_faces:
        for corner in range(3):
            vert_normals.index_add_(0, faces[n_body_faces:, corner], face_n[n_body_faces:])
    _ = n_body_verts
    vert_normals = F.normalize(vert_normals, dim=-1)

    return verts, vert_normals, faces, colors, fresnel, emissive


class EgoVehicleCompositor(nn.Module):
    """Composites a smoothly curved 3D fastback sedan with real NuRec skybox IBL reflections."""

    def __init__(
        self,
        ego_box_lwh: tuple[float, float, float] = (4.72, 1.85, 1.44),
        body_color: tuple[float, float, float] = (0.87, 0.89, 0.92),
        glass_color: tuple[float, float, float] = (0.07, 0.09, 0.13),
        points_per_meter: int = 22,
        splat_radius_px: int = 1,
    ) -> None:
        super().__init__()
        _ = (points_per_meter, splat_radius_px)
        self.ego_box_lwh = ego_box_lwh
        verts, vnorms, faces, face_colors, face_fresnel, face_emissive = _build_smooth_sedan_mesh(
            ego_box_lwh=ego_box_lwh,
            body_color=body_color,
            glass_color=glass_color,
        )
        self.register_buffer("ego_points", verts, persistent=False)
        self.register_buffer("ego_vert_normals", vnorms, persistent=False)
        self.register_buffer("ego_faces", faces, persistent=False)
        self.register_buffer("ego_face_colors", face_colors, persistent=False)
        self.register_buffer("ego_face_fresnel", face_fresnel, persistent=False)
        self.register_buffer("ego_face_emissive", face_emissive, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        projection: ProjectionResult,
        tgt_intrinsics: torch.Tensor,
        tgt_c2e: torch.Tensor,
        *,
        value_range: tuple[float, float] = (0.0, 1.0),
        skybox_textures: torch.Tensor | None = None,
        T_rig_world: torch.Tensor | None = None,
    ) -> ProjectionResult:
        """Rasterize smooth Phong + skybox-IBL ego sedan & dual-layer road shadow into ``projection``."""
        coarse_rgb = projection.coarse_rgb.clone()
        hole_mask = projection.hole_mask.clone()
        target_depth = projection.target_depth.clone()

        bsz, _, h_tgt, w_tgt = coarse_rgb.shape
        device = coarse_rgb.device
        dtype = coarse_rgb.dtype

        tgt_e2c = torch.linalg.inv(tgt_c2e.float())
        verts_ego = self.ego_points.to(device=device, dtype=torch.float32)
        vnorms_ego = self.ego_vert_normals.to(device=device, dtype=torch.float32)
        faces = self.ego_faces.to(device=device)
        face_cols = self.ego_face_colors.to(device=device, dtype=torch.float32)
        face_fres = self.ego_face_fresnel.to(device=device, dtype=torch.float32)
        face_emis = self.ego_face_emissive.to(device=device, dtype=torch.float32)

        ego_center_xy = verts_ego[:, :2].mean(dim=0)
        hl_s = self.ego_box_lwh[0] * 0.54
        hw_s = self.ego_box_lwh[1] * 0.56

        sun_dir = F.normalize(torch.tensor([0.35, -0.35, 0.86], device=device), dim=0)
        r_rig_to_world = (
            T_rig_world[:3, :3].to(device=device, dtype=torch.float32)
            if T_rig_world is not None
            else torch.eye(3, device=device, dtype=torch.float32)
        )

        for b in range(bsz):
            k = tgt_intrinsics[b].float()
            rot = tgt_e2c[b, :3, :3]
            trans = tgt_e2c[b, :3, 3]
            cam_pos_e = tgt_c2e[b, :3, 3].float()

            v_cam = verts_ego @ rot.T + trans[None, :]
            if (v_cam[:, 2] <= 0.25).all():
                continue

            u_v = k[0, 0] * (v_cam[:, 0] / v_cam[:, 2].clamp_min(0.25)) + k[0, 2]
            v_v = k[1, 1] * (v_cam[:, 1] / v_cam[:, 2].clamp_min(0.25)) + k[1, 2]

            x_min = max(0, int(torch.floor(u_v.min()).item()) - 14)
            x_max = min(w_tgt, int(torch.ceil(u_v.max()).item()) + 15)
            y_min = max(0, int(torch.floor(v_v.min()).item()) - 10)
            y_max = min(h_tgt, int(torch.ceil(v_v.max()).item()) + 18)
            if x_max <= x_min or y_max <= y_min:
                continue

            py = torch.arange(y_min, y_max, device=device, dtype=torch.float32) + 0.5
            px = torch.arange(x_min, x_max, device=device, dtype=torch.float32) + 0.5
            gy, gx = torch.meshgrid(py, px, indexing="ij")

            # 1. Dual-layer soft ambient occlusion contact shadow on z = 0 road plane
            ray_c = torch.stack([(gx - k[0, 2]) / k[0, 0], (gy - k[1, 2]) / k[1, 1], torch.ones_like(gx)], dim=-1)
            ray_e = ray_c @ tgt_c2e[b, :3, :3].float().T
            hit_g = ray_e[..., 2] < -1e-4
            t_g = torch.where(hit_g, -cam_pos_e[2] / ray_e[..., 2].clamp(max=-1e-4), torch.full_like(gx, 1e6))
            gx_e = cam_pos_e[0] + t_g * ray_e[..., 0] - ego_center_xy[0]
            gy_e = cam_pos_e[1] + t_g * ray_e[..., 1] - ego_center_xy[1]

            dx_core = (gx_e.abs() - hl_s * 0.80).clamp_min(0.0) / (hl_s * 0.14)
            dy_core = (gy_e.abs() - hw_s * 0.78).clamp_min(0.0) / (hw_s * 0.16)
            dx_soft = (gx_e.abs() - hl_s * 0.74).clamp_min(0.0) / (hl_s * 0.36)
            dy_soft = (gy_e.abs() - hw_s * 0.70).clamp_min(0.0) / (hw_s * 0.42)
            shadow = (
                0.55 * torch.exp(-3.0 * (dx_core * dx_core + dy_core * dy_core))
                + 0.25 * torch.exp(-2.2 * (dx_soft * dx_soft + dy_soft * dy_soft))
            ).clamp(0.0, 0.72)
            shadow = torch.where(hit_g & (t_g > 0.5), shadow, torch.zeros_like(shadow))

            if value_range == (-1.0, 1.0):
                patch_01 = (coarse_rgb[b, :, y_min:y_max, x_min:x_max].float() + 1.0) * 0.5
                patch_01 = patch_01 * (1.0 - shadow[None])
                coarse_rgb[b, :, y_min:y_max, x_min:x_max] = (patch_01 * 2.0 - 1.0).to(dtype=dtype)
            else:
                coarse_rgb[b, :, y_min:y_max, x_min:x_max] = (
                    coarse_rgb[b, :, y_min:y_max, x_min:x_max].float() * (1.0 - shadow[None])
                ).to(dtype=dtype)

            # 2. Vectorized barycentric rasterization with PER-PIXEL SMOOTH NORMALS & POSITIONS
            p0 = torch.stack([u_v[faces[:, 0]], v_v[faces[:, 0]]], dim=-1)
            p1 = torch.stack([u_v[faces[:, 1]], v_v[faces[:, 1]]], dim=-1)
            p2 = torch.stack([u_v[faces[:, 2]], v_v[faces[:, 2]]], dim=-1)
            z0 = v_cam[faces[:, 0], 2]
            z1 = v_cam[faces[:, 1], 2]
            z2 = v_cam[faces[:, 2], 2]

            valid_tri = (z0 > 0.25) & (z1 > 0.25) & (z2 > 0.25)
            f_idx_valid = torch.nonzero(valid_tri, as_tuple=False).squeeze(-1)
            if f_idx_valid.numel() == 0:
                continue

            p0, p1, p2 = p0[f_idx_valid], p1[f_idx_valid], p2[f_idx_valid]
            z0, z1, z2 = z0[f_idx_valid], z1[f_idx_valid], z2[f_idx_valid]
            area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
            non_deg = area.abs() > 1e-3
            if not non_deg.any():
                continue

            f_idx_valid = f_idx_valid[non_deg]
            p0, p1, p2 = p0[non_deg], p1[non_deg], p2[non_deg]
            z0, z1, z2 = z0[non_deg], z1[non_deg], z2[non_deg]
            inv_area = 1.0 / area[non_deg]

            hb, wb = y_max - y_min, x_max - x_min
            n_pix = hb * wb
            best_z = target_depth[b, 0, y_min:y_max, x_min:x_max].clone()
            best_hole = hole_mask[b, 0, y_min:y_max, x_min:x_max].clone()
            z_flat = torch.where(best_hole > 0.5, torch.full_like(best_z, 1e6), best_z).reshape(n_pix)
            hit_tri = torch.full((n_pix,), -1, device=device, dtype=torch.long)
            hit_w0 = torch.zeros(n_pix, device=device, dtype=torch.float32)
            hit_w1 = torch.zeros(n_pix, device=device, dtype=torch.float32)

            px_f = gx.reshape(1, n_pix)
            py_f = gy.reshape(1, n_pix)

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
                inv_z = w0 / z0[s:e, None] + w1 / z1[s:e, None] + w2 / z2[s:e, None]
                z_pix = torch.where(inside, 1.0 / inv_z.clamp_min(1e-6), torch.full_like(w0, 1e6))
                min_z_chunk, best_local = z_pix.min(dim=0)
                closer = min_z_chunk < z_flat
                if closer.any():
                    c_pix = torch.nonzero(closer, as_tuple=False).squeeze(-1)
                    b_loc = best_local[c_pix]
                    z_flat[c_pix] = min_z_chunk[c_pix]
                    hit_tri[c_pix] = f_idx_valid[s + b_loc]
                    hit_w0[c_pix] = w0[b_loc, c_pix]
                    hit_w1[c_pix] = w1[b_loc, c_pix]

            # 3. Shade ONLY hit pixels using smooth interpolated normals + real NuRec skybox IBL!
            hit_mask = hit_tri >= 0
            if not hit_mask.any():
                continue

            tri_ids = hit_tri[hit_mask]
            bw0 = hit_w0[hit_mask][:, None]
            bw1 = hit_w1[hit_mask][:, None]
            bw2 = (1.0 - bw0 - bw1)

            f_verts = faces[tri_ids]  # [P_hit, 3]
            pos_e = (
                bw0 * verts_ego[f_verts[:, 0]]
                + bw1 * verts_ego[f_verts[:, 1]]
                + bw2 * verts_ego[f_verts[:, 2]]
            )
            nrm_e = F.normalize(
                bw0 * vnorms_ego[f_verts[:, 0]]
                + bw1 * vnorms_ego[f_verts[:, 1]]
                + bw2 * vnorms_ego[f_verts[:, 2]],
                dim=-1,
            )
            view_e = F.normalize(cam_pos_e[None, :] - pos_e, dim=-1)
            ndotv_raw = (nrm_e * view_e).sum(dim=-1, keepdim=True)
            nrm_e = torch.where(ndotv_raw < 0.0, -nrm_e, nrm_e)
            ndotv = ndotv_raw.abs().clamp(0.0, 1.0)

            base_col = face_cols[tri_ids]
            fres_k = face_fres[tri_ids][:, None]
            emis_k = face_emis[tri_ids][:, None]

            # Mirror reflection vector in NuRec world frame for real cubemap environment lighting!
            refl_e = F.normalize(2.0 * ndotv * nrm_e - view_e, dim=-1)
            if skybox_textures is not None:
                refl_w = refl_e @ r_rig_to_world.T
                nrm_w = nrm_e @ r_rig_to_world.T
                env_spec = _sample_cubemap_dirs(skybox_textures, refl_w)
                env_diff = _sample_cubemap_dirs(skybox_textures, nrm_w)
            else:
                sky_grad = (0.5 + 0.5 * refl_e[:, 2:3]).clamp(0.0, 1.0)
                env_spec = torch.cat([0.55 + 0.15 * sky_grad, 0.62 + 0.16 * sky_grad, 0.72 + 0.18 * sky_grad], dim=-1)
                env_diff = env_spec * 0.85

            # Schlick Fresnel clearcoat + diffuse ambient + crisp sun specular
            ndotl = (nrm_e @ sun_dir).clamp(0.0, 1.0)[:, None]
            half_vec = F.normalize(sun_dir[None, :] + view_e, dim=-1)
            sun_spec = ((nrm_e * half_vec).sum(dim=-1, keepdim=True).clamp(0.0, 1.0) ** 48) * fres_k * 0.55
            schlick = (0.06 + 0.72 * ((1.0 - ndotv) ** 4)) * fres_k

            lit_rgb = (
                base_col * (0.30 + 0.45 * env_diff + 0.35 * ndotl) * (1.0 - 0.55 * schlick)
                + env_spec * schlick * 0.92
                + sun_spec
                + base_col * emis_k * 0.85
            ).clamp(0.0, 1.0)
            if value_range == (-1.0, 1.0):
                lit_rgb = lit_rgb * 2.0 - 1.0

            rgb_flat = coarse_rgb[b, :, y_min:y_max, x_min:x_max].float().reshape(3, n_pix)
            hole_flat = best_hole.reshape(n_pix)
            rgb_flat[:, hit_mask] = lit_rgb.T
            hole_flat[hit_mask] = 0.0

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
