"""Abstract interface for pluggable shared driving/embodied vision encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class VisionEncoderOutput:
    """Spatiotemporal multi-camera patch tokens from a frozen shared vision encoder."""

    tokens: torch.Tensor  # [B, T, K, H_p, W_p, D]
    grid_hw: tuple[int, int]
    hidden_size: int


class SharedVisionEncoder(nn.Module, ABC):
    """Base class enabling drop-in ablations across Alpamayo2-Edge, DM0, and DINOv3."""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Feature dimension ``D`` of output visual tokens."""

    @property
    @abstractmethod
    def patch_stride(self) -> int:
        """Effective spatial stride (pixels per output token)."""

    @abstractmethod
    def encode_multiview(self, images: torch.Tensor) -> VisionEncoderOutput:
        """Encode surround-view videos ``[B, T, K, 3, H, W]`` into ``[B, T, K, H_p, W_p, D]``."""

    def forward(self, images: torch.Tensor) -> VisionEncoderOutput:
        return self.encode_multiview(images)
