"""Zero-compiler PyTorch + Triton 3D Gaussian Splatting tile rasterizer.

Renders millions of 3D Gaussians (e.g., 2.5M Gaussians from NVIDIA PhysicalAI-AV-NuRec
``checkpoint.ckpt``) into photorealistic RGB + metric depth + alpha maps on any CUDA GPU
without requiring ``nvcc`` or ``g++`` C++ extensions.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON_COMPILER = shutil.which("gcc") is not None or shutil.which("clang") is not None or shutil.which("cc") is not None
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON_COMPILER = False

# Standard Spherical Harmonics normalization constants (degree 0..3)
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


@dataclass
class GaussianRenderOutput:
    """Outputs from rendering a 3DGS scene into a pinhole camera."""

    rgb: torch.Tensor    # [H, W, 3] in [0, 1]
    depth: torch.Tensor  # [H, W] expected metric depth in meters
    alpha: torch.Tensor  # [H, W] accumulated opacity in [0, 1]


def quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``wxyz`` quaternions ``[..., 4]`` into ``[..., 3, 3]`` rotation matrices."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def evaluate_sh_degree3(
    sh_dc: torch.Tensor,
    sh_rest: torch.Tensor | None,
    dirs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate SH degree 0..3 RGB radiance ``[N, 3]`` for unit view directions ``dirs [N, 3]``.

    Args:
        sh_dc: ``[N, 3]`` DC spherical harmonic coefficients.
        sh_rest: Optional ``[N, 15, 3]`` (or ``[N, 45]``) higher-order SH coefficients (degrees 1..3).
        dirs: Unit view directions ``[N, 3]`` pointing from camera to Gaussian.
    """
    result = _SH_C0 * sh_dc + 0.5
    if sh_rest is None or sh_rest.numel() == 0:
        return result.clamp(0.0, 1.0)

    if sh_rest.ndim == 2:
        # Check channel layout: NuRec / 3DGRUT stores features_specular as [N, 15*3] or [N, 3*15].
        # Standard 3DGS layout is [N, 15, 3] (interleaved RGB per SH basis) or [N, 3, 15].
        sh_rest = sh_rest.view(sh_rest.shape[0], -1, 3)

    num_coeffs = sh_rest.shape[1]
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]

    if num_coeffs >= 3:
        result = result + _SH_C1 * (
            -y * sh_rest[:, 0] + z * sh_rest[:, 1] - x * sh_rest[:, 2]
        )
    if num_coeffs >= 8:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (
            result
            + _SH_C2[0] * xy * sh_rest[:, 3]
            + _SH_C2[1] * yz * sh_rest[:, 4]
            + _SH_C2[2] * (2.0 * zz - xx - yy) * sh_rest[:, 5]
            + _SH_C2[3] * xz * sh_rest[:, 6]
            + _SH_C2[4] * (xx - yy) * sh_rest[:, 7]
        )
    if num_coeffs >= 15:
        xx, yy, zz = x * x, y * y, z * z
        result = (
            result
            + _SH_C3[0] * y * (3.0 * xx - yy) * sh_rest[:, 8]
            + _SH_C3[1] * x * y * z * sh_rest[:, 9]
            + _SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_rest[:, 10]
            + _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_rest[:, 11]
            + _SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_rest[:, 12]
            + _SH_C3[5] * z * (xx - yy) * sh_rest[:, 13]
            + _SH_C3[6] * x * (xx - 3.0 * yy) * sh_rest[:, 14]
        )
    return result.clamp(0.0, 1.0)


def sample_cubemap_skybox(
    textures: torch.Tensor,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Sample a 6-face cubemap ``[6, S, S, 3]`` along every pinhole camera ray in world space."""
    device = textures.device
    if textures.ndim == 5:
        textures = textures[0]  # [6, S, S, 3]
    tex_chw = textures.permute(0, 3, 1, 2).float()  # [6, 3, S, S]

    v = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    u = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    x_c = (uu - intrinsics[0, 2]) / intrinsics[0, 0]
    y_c = (vv - intrinsics[1, 2]) / intrinsics[1, 1]
    dirs_cam = torch.stack([x_c, y_c, torch.ones_like(x_c)], dim=-1)
    dirs_w = dirs_cam @ c2w[:3, :3].float().T
    dirs_w = F.normalize(dirs_w, dim=-1)

    # NuRec world is Z-up (+X fwd, +Y left, +Z up), while NuRec skybox cubemap uses
    # Y-up OpenGL face convention: +X=-y_w, -X=+y_w, +Y=+z_w, -Y=-z_w, +Z=+x_w, -Z=-x_w.
    dx = -dirs_w[..., 1]
    dy = dirs_w[..., 2]
    dz = dirs_w[..., 0]
    ax, ay, az = dx.abs(), dy.abs(), dz.abs()

    # Standard OpenGL/nvdiffrast cubemap face ordering: +X, -X, +Y, -Y, +Z, -Z
    face_idx = torch.zeros((height, width), device=device, dtype=torch.long)
    uc = torch.zeros((height, width), device=device, dtype=torch.float32)
    vc = torch.zeros((height, width), device=device, dtype=torch.float32)

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

    sky = torch.zeros((height, width, 3), device=device, dtype=torch.float32)
    for f_id in range(6):
        mask = face_idx == f_id
        if not mask.any():
            continue
        grid = torch.stack([uc[mask], vc[mask]], dim=-1).view(1, 1, -1, 2)
        sampled = F.grid_sample(tex_chw[f_id : f_id + 1], grid, mode="bilinear", align_corners=False)
        sky[mask] = sampled[0, :, 0, :].T

    # NuRec skybox textures are stored in unclipped linear/radiance range; clamp to [0, 1].
    return sky.clamp(0.0, 1.0)


if triton is not None:

    @triton.jit
    def _rasterize_tiles_kernel(
        tile_starts_ptr,
        tile_ends_ptr,
        sorted_gids_ptr,
        means2d_ptr,
        conics_ptr,
        opacities_ptr,
        colors_ptr,
        depths_ptr,
        out_rgb_ptr,
        out_depth_ptr,
        out_alpha_ptr,
        height: tl.constexpr,
        width: tl.constexpr,
        num_tiles_x: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
    ):
        tile_x = tl.program_id(0)
        tile_y = tl.program_id(1)
        tile_id = tile_y * num_tiles_x + tile_x

        start = tl.load(tile_starts_ptr + tile_id)
        end = tl.load(tile_ends_ptr + tile_id)

        # 256 pixels in this 16x16 tile
        pix_ids = tl.arange(0, TILE_SIZE * TILE_SIZE)
        py_local = pix_ids // TILE_SIZE
        px_local = pix_ids % TILE_SIZE
        px = tile_x * TILE_SIZE + px_local
        py = tile_y * TILE_SIZE + py_local
        valid_pix = (px < width) & (py < height)

        px_f = px.to(tl.float32) + 0.5
        py_f = py.to(tl.float32) + 0.5

        acc_r = tl.zeros((TILE_SIZE * TILE_SIZE,), dtype=tl.float32)
        acc_g = tl.zeros((TILE_SIZE * TILE_SIZE,), dtype=tl.float32)
        acc_b = tl.zeros((TILE_SIZE * TILE_SIZE,), dtype=tl.float32)
        acc_d = tl.zeros((TILE_SIZE * TILE_SIZE,), dtype=tl.float32)
        T = tl.full((TILE_SIZE * TILE_SIZE,), 1.0, dtype=tl.float32)

        num_gauss = end - start
        for offset in range(0, num_gauss, CHUNK_SIZE):
            # Early exit once all pixels in this tile are saturated.
            if tl.max(T, axis=0) >= 1e-4:
                chunk_end = tl.minimum(offset + CHUNK_SIZE, num_gauss)
                for idx in range(offset, chunk_end):
                    gid = tl.load(sorted_gids_ptr + start + idx)
                    mx = tl.load(means2d_ptr + gid * 2 + 0)
                    my = tl.load(means2d_ptr + gid * 2 + 1)
                    ca = tl.load(conics_ptr + gid * 3 + 0)
                    cb = tl.load(conics_ptr + gid * 3 + 1)
                    cc = tl.load(conics_ptr + gid * 3 + 2)
                    opac = tl.load(opacities_ptr + gid)

                    dx = px_f - mx
                    dy = py_f - my
                    power = -0.5 * (ca * dx * dx + 2.0 * cb * dx * dy + cc * dy * dy)
                    alpha = tl.minimum(0.99, opac * tl.exp(power))
                    alpha = tl.where((power <= 0.0) & (alpha >= (1.0 / 255.0)), alpha, 0.0)

                    w = T * alpha
                    r = tl.load(colors_ptr + gid * 3 + 0)
                    g = tl.load(colors_ptr + gid * 3 + 1)
                    b = tl.load(colors_ptr + gid * 3 + 2)
                    d = tl.load(depths_ptr + gid)

                    acc_r += w * r
                    acc_g += w * g
                    acc_b += w * b
                    acc_d += w * d
                    T = T * (1.0 - alpha)

        alpha_out = 1.0 - T
        depth_out = tl.where(alpha_out > 1e-4, acc_d / tl.maximum(alpha_out, 1e-4), 0.0)

        out_idx = py * width + px
        tl.store(out_rgb_ptr + out_idx * 3 + 0, acc_r, mask=valid_pix)
        tl.store(out_rgb_ptr + out_idx * 3 + 1, acc_g, mask=valid_pix)
        tl.store(out_rgb_ptr + out_idx * 3 + 2, acc_b, mask=valid_pix)
        tl.store(out_depth_ptr + out_idx, depth_out, mask=valid_pix)
        tl.store(out_alpha_ptr + out_idx, alpha_out, mask=valid_pix)


@torch.no_grad()
def _rasterize_tiles_torch(
    tile_starts: torch.Tensor,
    tile_ends: torch.Tensor,
    sorted_gids: torch.Tensor,
    means2d: torch.Tensor,
    conic: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    height: int,
    width: int,
    num_tiles_x: int,
    num_tiles_y: int,
    tile_size: int = 16,
    chunk_g: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch vectorized GPU tile alpha-blending fallback (requires zero C/CUDA compiler)."""
    device = means2d.device
    num_tiles = num_tiles_x * num_tiles_y
    p_count = tile_size * tile_size

    tile_ids = torch.arange(num_tiles, device=device, dtype=torch.long)
    ty = tile_ids // num_tiles_x
    tx = tile_ids % num_tiles_x

    local_p = torch.arange(p_count, device=device, dtype=torch.long)
    ly = local_p // tile_size
    lx = local_p % tile_size

    px_all = (tx[:, None] * tile_size + lx[None, :]).float() + 0.5  # [num_tiles, P]
    py_all = (ty[:, None] * tile_size + ly[None, :]).float() + 0.5  # [num_tiles, P]

    acc_rgb = torch.zeros((num_tiles, p_count, 3), device=device, dtype=torch.float32)
    acc_depth = torch.zeros((num_tiles, p_count), device=device, dtype=torch.float32)
    trans = torch.ones((num_tiles, p_count), device=device, dtype=torch.float32)

    cursors = tile_starts.to(torch.long).clone()
    ends = tile_ends.to(torch.long)
    step_offsets = torch.arange(chunk_g, device=device, dtype=torch.long)
    max_gid_idx = max(0, sorted_gids.shape[0] - 1)

    active = torch.nonzero(cursors < ends, as_tuple=False).squeeze(-1)
    while active.numel() > 0:
        cur = cursors[active]  # [A]
        end = ends[active]  # [A]

        pos = cur[:, None] + step_offsets[None, :]  # [A, B]
        valid_g = pos < end[:, None]
        gids = sorted_gids[pos.clamp(max=max_gid_idx)].to(torch.long)  # [A, B]

        m2 = means2d[gids]  # [A, B, 2]
        con = conic[gids]  # [A, B, 3]
        op = opacities[gids]  # [A, B]
        col = colors[gids]  # [A, B, 3]
        dep = depths[gids]  # [A, B]

        dx = px_all[active][:, None, :] - m2[:, :, 0:1]  # [A, B, P]
        dy = py_all[active][:, None, :] - m2[:, :, 1:2]  # [A, B, P]

        power = -0.5 * (
            con[:, :, 0:1] * dx * dx
            + 2.0 * con[:, :, 1:2] * dx * dy
            + con[:, :, 2:3] * dy * dy
        )
        alpha = (op[:, :, None] * torch.exp(power.clamp(max=0.0))).clamp(max=0.99)
        mask = valid_g[:, :, None] & (power <= 0.0) & (alpha >= (1.0 / 255.0))
        alpha = torch.where(mask, alpha, torch.zeros_like(alpha))

        one_minus_a = 1.0 - alpha
        cum_t = torch.cumprod(one_minus_a, dim=1)  # [A, B, P]
        t_local = torch.empty_like(cum_t)
        t_local[:, 0, :] = 1.0
        t_local[:, 1:, :] = cum_t[:, :-1, :]

        t_cur = trans[active]  # [A, P]
        weights = t_cur[:, None, :] * t_local * alpha  # [A, B, P]

        acc_rgb[active] = acc_rgb[active] + torch.einsum("abp,abc->apc", weights, col)
        acc_depth[active] = acc_depth[active] + (weights * dep[:, :, None]).sum(dim=1)
        new_t = t_cur * cum_t[:, -1, :]
        trans[active] = new_t

        new_cur = (cur + chunk_g).clamp(max=end)
        cursors[active] = new_cur

        still_alive = (new_cur < end) & (new_t.max(dim=-1).values >= 1e-3)
        active = active[still_alive]

    alpha_tiles = 1.0 - trans  # [num_tiles, P]
    depth_tiles = torch.where(alpha_tiles > 1e-4, acc_depth / alpha_tiles.clamp_min(1e-4), torch.zeros_like(acc_depth))

    # Unfold [num_tiles_y, num_tiles_x, tile_size, tile_size, ...] -> [H, W, ...]
    rgb_grid = (
        acc_rgb.view(num_tiles_y, num_tiles_x, tile_size, tile_size, 3)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_tiles_y * tile_size, num_tiles_x * tile_size, 3)
    )
    depth_grid = (
        depth_tiles.view(num_tiles_y, num_tiles_x, tile_size, tile_size)
        .permute(0, 2, 1, 3)
        .reshape(num_tiles_y * tile_size, num_tiles_x * tile_size)
    )
    alpha_grid = (
        alpha_tiles.view(num_tiles_y, num_tiles_x, tile_size, tile_size)
        .permute(0, 2, 1, 3)
        .reshape(num_tiles_y * tile_size, num_tiles_x * tile_size)
    )
    return (
        rgb_grid[:height, :width].contiguous(),
        depth_grid[:height, :width].contiguous(),
        alpha_grid[:height, :width].contiguous(),
    )


@torch.no_grad()
def render_gaussians_triton(
    means: torch.Tensor,
    quats_wxyz: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh_dc: torch.Tensor,
    sh_rest: torch.Tensor | None,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    *,
    skybox_textures: torch.Tensor | None = None,
    near_plane: float = 0.25,
    far_plane: float = 400.0,
    opacity_threshold: float = 0.01,
    max_radius_px: int = 96,
    tile_size: int = 16,
) -> GaussianRenderOutput:
    """Render 3D Gaussians into a pinhole camera using pure PyTorch projection + Triton rasterization."""
    device = means.device
    c2w = c2w.to(device=device, dtype=torch.float32)
    w2c = torch.linalg.inv(c2w)
    k = intrinsics.to(device=device, dtype=torch.float32)

    rot_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]

    # 1. Transform centers to camera frame & frustum cull
    means_cam = means @ rot_w2c.T + t_w2c[None, :]
    z = means_cam[:, 2]
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]

    u = fx * (means_cam[:, 0] / z.clamp_min(1e-4)) + cx
    v = fy * (means_cam[:, 1] / z.clamp_min(1e-4)) + cy

    margin = float(max_radius_px)
    vis = (
        (z > near_plane)
        & (z < far_plane)
        & (opacities > opacity_threshold)
        & (u > -margin)
        & (u < width + margin)
        & (v > -margin)
        & (v < height + margin)
    )

    if not vis.any():
        rgb = torch.zeros((height, width, 3), device=device)
        if skybox_textures is not None:
            rgb = sample_cubemap_skybox(skybox_textures, c2w, k, height, width)
        return GaussianRenderOutput(
            rgb=rgb,
            depth=torch.zeros((height, width), device=device),
            alpha=torch.zeros((height, width), device=device),
        )

    means_v = means[vis]
    means_cam_v = means_cam[vis]
    z_v = z[vis]
    u_v = u[vis]
    v_v = v[vis]
    quats_v = quats_wxyz[vis]
    scales_v = scales[vis]
    opac_v = opacities[vis]
    sh_dc_v = sh_dc[vis]
    sh_rest_v = sh_rest[vis] if sh_rest is not None else None

    # 2. EWA 2D covariance projection: cov2d = (J W R S)(J W R S)^T + 0.3 I
    rot_g = quat_wxyz_to_rotmat(quats_v)  # [M, 3, 3]
    m_mat = rot_g * scales_v[:, None, :]  # R @ diag(s)
    wm = torch.matmul(rot_w2c[None], m_mat)  # [M, 3, 3]

    # Clamp x/z, y/z inside 1.3x frustum to stabilize peripheral EWA Jacobian
    lim_x = 1.3 * (width / (2.0 * fx))
    lim_y = 1.3 * (height / (2.0 * fy))
    tx = (means_cam_v[:, 0] / z_v).clamp(-lim_x, lim_x)
    ty = (means_cam_v[:, 1] / z_v).clamp(-lim_y, lim_y)
    inv_z = 1.0 / z_v

    # Row 0 of J: [fx/z, 0, -fx*tx/z]; Row 1 of J: [0, fy/z, -fy*ty/z]
    t0 = (fx * inv_z)[:, None] * (wm[:, 0, :] - tx[:, None] * wm[:, 2, :])  # [M, 3]
    t1 = (fy * inv_z)[:, None] * (wm[:, 1, :] - ty[:, None] * wm[:, 2, :])  # [M, 3]

    cov00 = (t0 * t0).sum(dim=-1) + 0.3
    cov01 = (t0 * t1).sum(dim=-1)
    cov11 = (t1 * t1).sum(dim=-1) + 0.3
    det = cov00 * cov11 - cov01 * cov01

    good = det > 1e-6
    if not good.any():
        return GaussianRenderOutput(
            rgb=torch.zeros((height, width, 3), device=device),
            depth=torch.zeros((height, width), device=device),
            alpha=torch.zeros((height, width), device=device),
        )

    means_v = means_v[good]
    z_v = z_v[good]
    u_v = u_v[good]
    v_v = v_v[good]
    opac_v = opac_v[good]
    sh_dc_v = sh_dc_v[good]
    sh_rest_v = sh_rest_v[good] if sh_rest_v is not None else None
    cov00, cov01, cov11, det = cov00[good], cov01[good], cov11[good], det[good]

    inv_det = 1.0 / det
    conic = torch.stack([cov11 * inv_det, -cov01 * inv_det, cov00 * inv_det], dim=-1)  # [M, 3]

    mid = 0.5 * (cov00 + cov11)
    disc = (mid * mid - det).clamp_min(0.1).sqrt()
    lam_max = mid + disc
    radii = torch.ceil(3.0 * lam_max.sqrt()).clamp(min=1.0, max=float(max_radius_px)).to(torch.int32)

    # 3. Evaluate SH degree 3 colors
    cam_pos = c2w[:3, 3]
    view_dirs = F.normalize(means_v - cam_pos[None, :], dim=-1)
    colors = evaluate_sh_degree3(sh_dc_v, sh_rest_v, view_dirs).contiguous()

    # 4. Tile binning & depth-major radix sort per tile
    num_tiles_x = (width + tile_size - 1) // tile_size
    num_tiles_y = (height + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y

    tx0 = torch.floor((u_v - radii) / tile_size).clamp(0, num_tiles_x - 1).to(torch.int32)
    tx1 = torch.floor((u_v + radii) / tile_size).clamp(0, num_tiles_x - 1).to(torch.int32)
    ty0 = torch.floor((v_v - radii) / tile_size).clamp(0, num_tiles_y - 1).to(torch.int32)
    ty1 = torch.floor((v_v + radii) / tile_size).clamp(0, num_tiles_y - 1).to(torch.int32)

    # Cull Gaussians whose bounding box is completely outside image bounds
    in_img = (u_v + radii >= 0) & (u_v - radii < width) & (v_v + radii >= 0) & (v_v - radii < height)
    if not in_img.any():
        return GaussianRenderOutput(
            rgb=torch.zeros((height, width, 3), device=device),
            depth=torch.zeros((height, width), device=device),
            alpha=torch.zeros((height, width), device=device),
        )

    means2d = torch.stack([u_v, v_v], dim=-1).contiguous()
    nx = (tx1 - tx0 + 1).clamp_min(0)
    ny = (ty1 - ty0 + 1).clamp_min(0)
    tiles_per_g = torch.where(in_img, nx * ny, torch.zeros_like(nx))

    total_isects = int(tiles_per_g.sum().item())
    if total_isects == 0:
        return GaussianRenderOutput(
            rgb=torch.zeros((height, width, 3), device=device),
            depth=torch.zeros((height, width), device=device),
            alpha=torch.zeros((height, width), device=device),
        )

    # Vectorized expansion of (gaussian_id, tile_id) intersections on GPU
    g_indices = torch.repeat_interleave(
        torch.arange(means2d.shape[0], device=device, dtype=torch.int32),
        tiles_per_g,
    )
    cum = torch.cumsum(tiles_per_g, dim=0)
    starts_g = cum - tiles_per_g
    local_idx = torch.arange(total_isects, device=device, dtype=torch.int64) - starts_g[g_indices]

    nx_g = nx[g_indices].to(torch.int64)
    tile_y = ty0[g_indices].to(torch.int64) + (local_idx // nx_g)
    tile_x = tx0[g_indices].to(torch.int64) + (local_idx % nx_g)
    tile_ids = tile_y * num_tiles_x + tile_x

    # Pack (tile_id << 32) | depth_bits (positive float32 preserves order as int32)
    depth_u32 = z_v[g_indices].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sort_keys = (tile_ids << 32) | depth_u32
    _, perm = torch.sort(sort_keys)
    sorted_gids = g_indices[perm].contiguous()
    sorted_tids = tile_ids[perm].contiguous()

    tile_grid = torch.arange(num_tiles, device=device, dtype=torch.int64)
    tile_starts = torch.searchsorted(sorted_tids, tile_grid, right=False).to(torch.int32).contiguous()
    tile_ends = torch.searchsorted(sorted_tids, tile_grid, right=True).to(torch.int32).contiguous()

    if _HAS_TRITON_COMPILER and triton is not None and device.type == "cuda":
        out_rgb = torch.empty((height, width, 3), device=device, dtype=torch.float32)
        out_depth = torch.empty((height, width), device=device, dtype=torch.float32)
        out_alpha = torch.empty((height, width), device=device, dtype=torch.float32)

        _rasterize_tiles_kernel[(num_tiles_x, num_tiles_y)](
            tile_starts,
            tile_ends,
            sorted_gids,
            means2d,
            conic.contiguous(),
            opac_v.contiguous(),
            colors,
            z_v.contiguous(),
            out_rgb,
            out_depth,
            out_alpha,
            height=height,
            width=width,
            num_tiles_x=num_tiles_x,
            TILE_SIZE=tile_size,
            CHUNK_SIZE=64,
        )
    else:
        out_rgb, out_depth, out_alpha = _rasterize_tiles_torch(
            tile_starts=tile_starts,
            tile_ends=tile_ends,
            sorted_gids=sorted_gids,
            means2d=means2d,
            conic=conic.contiguous(),
            opacities=opac_v.contiguous(),
            colors=colors,
            depths=z_v.contiguous(),
            height=height,
            width=width,
            num_tiles_x=num_tiles_x,
            num_tiles_y=num_tiles_y,
            tile_size=tile_size,
            chunk_g=16,
        )

    if skybox_textures is not None:
        sky_rgb = sample_cubemap_skybox(skybox_textures, c2w, k, height, width)
        out_rgb = out_rgb + (1.0 - out_alpha[..., None]) * sky_rgb

    return GaussianRenderOutput(
        rgb=out_rgb.clamp(0.0, 1.0),
        depth=out_depth,
        alpha=out_alpha.clamp(0.0, 1.0),
    )
