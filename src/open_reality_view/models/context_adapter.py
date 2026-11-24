"""Perceiver Resampler bridging frozen Alpamayo2-Edge tokens (D=1152) to Wan2.1 (text_dim=4096)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from open_reality_view.geometry.plucker import FourierPluckerEncoder


class _CrossSelfAttnBlock(nn.Module):
    """Perceiver block: cross-attention from latents to visual tokens + self-attention + FFN."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.05) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_kv = nn.Linear(dim, dim * 2, bias=False)
        self.cross_out = nn.Linear(dim, dim)

        self.norm_self = nn.LayerNorm(dim)
        self.self_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.self_out = nn.Linear(dim, dim)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, latents: torch.Tensor, vis_tokens: torch.Tensor) -> torch.Tensor:
        bsz, n_lat, dim = latents.shape
        p_drop = self.dropout if self.training else 0.0

        # Cross-attention: latents query multi-view spatio-temporal tokens
        q = self._split_heads(self.cross_q(self.norm_q(latents)))
        k, v = self.cross_kv(self.norm_kv(vis_tokens)).chunk(2, dim=-1)
        k = self._split_heads(k)
        v = self._split_heads(v)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=p_drop)
        attn = attn.transpose(1, 2).reshape(bsz, n_lat, dim)
        latents = latents + self.cross_out(attn)

        # Self-attention over compact latent tokens
        q_s, k_s, v_s = self.self_qkv(self.norm_self(latents)).chunk(3, dim=-1)
        attn_s = F.scaled_dot_product_attention(
            self._split_heads(q_s),
            self._split_heads(k_s),
            self._split_heads(v_s),
            dropout_p=p_drop,
        )
        attn_s = attn_s.transpose(1, 2).reshape(bsz, n_lat, dim)
        latents = latents + self.self_out(attn_s)

        # Feed-forward
        latents = latents + self.ffn(self.norm_ffn(latents))
        return latents


class AlpamayoContextAdapter(nn.Module):
    """Projects multi-camera frozen vision tokens + Plücker rays into Wan2.1 cross-attention space.

    Replaces the 11 GB UMT5-XXL text encoder in Wan2.1-VACE-1.3B with a compact, driving-grounded
    semantic context tensor of shape ``[B, num_latents, 4096]``.
    """

    def __init__(
        self,
        in_dim: int = 1152,
        adapter_dim: int = 1152,
        out_dim: int = 4096,
        num_latents: int = 256,
        num_layers: int = 4,
        num_heads: int = 16,
        max_cameras: int = 8,
        max_frames: int = 64,
        plucker_freq_bands: int = 6,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.adapter_dim = adapter_dim
        self.out_dim = out_dim
        self.num_latents = num_latents

        self.in_proj = nn.Linear(in_dim, adapter_dim) if in_dim != adapter_dim else nn.Identity()
        self.plucker_encoder = FourierPluckerEncoder(
            out_dim=adapter_dim,
            num_freq_bands=plucker_freq_bands,
        )
        self.camera_embed = nn.Embedding(max_cameras, adapter_dim)
        self.time_embed = nn.Embedding(max_frames, adapter_dim)

        self.latents = nn.Parameter(torch.randn(1, num_latents, adapter_dim) / math.sqrt(adapter_dim))
        self.blocks = nn.ModuleList(
            [_CrossSelfAttnBlock(adapter_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.RMSNorm(adapter_dim)
        self.out_proj = nn.Linear(adapter_dim, out_dim)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        plucker_rays: torch.Tensor | None = None,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse surround-view tokens and rays into Wan2.1 cross-attention context.

        Args:
            vision_tokens: ``[B, T, K, H_p, W_p, D_in]`` frozen vision tokens.
            plucker_rays: Optional ``[B, T, K, 6, H_p, W_p]`` Plücker rays in target-camera frame.
            camera_ids: Optional ``[K]`` or ``[B, K]`` integer camera indices.

        Returns:
            Cross-attention context ``[B, num_latents, out_dim]`` (default ``out_dim=4096``).
        """
        bsz, num_frames, num_cams, h_p, w_p, _ = vision_tokens.shape
        device = vision_tokens.device

        x = self.in_proj(vision_tokens)  # [B, T, K, H_p, W_p, adapter_dim]

        if plucker_rays is not None:
            ray_feats = self.plucker_encoder(plucker_rays.movedim(-3, -1))
            x = x + ray_feats

        if camera_ids is None:
            cam_idx = torch.arange(num_cams, device=device)
            cam_emb = self.camera_embed(cam_idx)[None, None, :, None, None, :]
        elif camera_ids.ndim == 1:
            cam_emb = self.camera_embed(camera_ids)[None, None, :, None, None, :]
        else:
            cam_emb = self.camera_embed(camera_ids)[:, None, :, None, None, :]
        x = x + cam_emb

        t_idx = torch.arange(num_frames, device=device)
        t_emb = self.time_embed(t_idx)[None, :, None, None, None, :]
        x = x + t_emb

        vis_seq = x.reshape(bsz, num_frames * num_cams * h_p * w_p, self.adapter_dim)
        latents = self.latents.expand(bsz, -1, -1)

        for block in self.blocks:
            latents = block(latents, vis_seq)

        return self.out_proj(self.out_norm(latents))
