"""Multi-camera Plücker ray maps expressed in target-camera coordinates."""

from __future__ import annotations

import math

import torch
from torch import nn


def compute_plucker_rays(
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    grid_height: int,
    grid_width: int,
    *,
    orig_height: int | None = None,
    orig_width: int | None = None,
    target_w2c: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute dense 6-channel Plücker ray embeddings ``(o x d, d)``.

    Args:
        intrinsics: Camera intrinsic matrices ``[..., 3, 3]`` for the original image resolution.
        c2w: Camera-to-world (or camera-to-ego) rigid transforms ``[..., 4, 4]`` (OpenCV frame).
        grid_height: Spatial height of the output ray map (e.g., ViT patch grid height).
        grid_width: Spatial width of the output ray map (e.g., ViT patch grid width).
        orig_height: Original pixel height corresponding to ``intrinsics``. Defaults to ``grid_height``.
        orig_width: Original pixel width corresponding to ``intrinsics``. Defaults to ``grid_width``.
        target_w2c: Optional world-to-target-camera transforms broadcastable to ``[..., 4, 4]``.
            When provided, all surround-camera rays are expressed in the target chase camera frame,
            making conditioning invariant to global ego pose and enabling interactive camera orbit.
        eps: Numerical floor for ray normalization.

    Returns:
        Plücker ray tensor of shape ``[..., 6, grid_height, grid_width]`` where channels ``0:3``
        are the moment ``o x d`` and channels ``3:6`` are the unit direction ``d``.
    """
    device = intrinsics.device
    dtype = intrinsics.dtype
    batch_shape = c2w.shape[:-2]

    src_h = float(orig_height if orig_height is not None else grid_height)
    src_w = float(orig_width if orig_width is not None else grid_width)

    # Pixel coordinates at centers of each patch cell mapped back into original image pixel space.
    v_coords = (torch.arange(grid_height, device=device, dtype=dtype) + 0.5) * (src_h / grid_height)
    u_coords = (torch.arange(grid_width, device=device, dtype=dtype) + 0.5) * (src_w / grid_width)
    vv, uu = torch.meshgrid(v_coords, u_coords, indexing="ij")  # [H, W]

    fx = intrinsics[..., 0, 0][..., None, None]
    fy = intrinsics[..., 1, 1][..., None, None]
    cx = intrinsics[..., 0, 2][..., None, None]
    cy = intrinsics[..., 1, 2][..., None, None]

    x_cam = (uu - cx) / fx.clamp_min(eps)
    y_cam = (vv - cy) / fy.clamp_min(eps)
    z_cam = torch.ones_like(x_cam)
    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [..., H, W, 3]

    pose = torch.matmul(target_w2c, c2w) if target_w2c is not None else c2w
    rot = pose[..., :3, :3]  # [..., 3, 3]
    origin = pose[..., :3, 3]  # [..., 3]

    dirs = torch.einsum("...ij,...hwj->...hwi", rot, dirs_cam)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(eps)

    origins = origin[..., None, None, :].expand(*batch_shape, grid_height, grid_width, 3)
    moments = torch.linalg.cross(origins, dirs, dim=-1)

    plucker = torch.cat([moments, dirs], dim=-1)  # [..., H, W, 6]
    return plucker.movedim(-1, -3).contiguous()


class FourierPluckerEncoder(nn.Module):
    """Sinusoidal Fourier encoder mapping 6D Plücker coordinates to token embeddings."""

    def __init__(self, out_dim: int, num_freq_bands: int = 6) -> None:
        super().__init__()
        self.num_freq_bands = num_freq_bands
        in_features = 6 * (1 + 2 * num_freq_bands)
        freqs = 2.0 ** torch.arange(num_freq_bands, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        # Small init on the final projection keeps initial frozen vision features intact.
        nn.init.normal_(self.proj[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, plucker: torch.Tensor) -> torch.Tensor:
        """Encode Plücker rays of shape ``[..., 6]`` into ``[..., out_dim]``."""
        orig_dtype = plucker.dtype
        x = plucker.float()
        angles = x.unsqueeze(-1) * self.freqs  # [..., 6, F]
        encoded = torch.cat(
            [x, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)],
            dim=-1,
        )
        return self.proj(encoded.to(orig_dtype))
