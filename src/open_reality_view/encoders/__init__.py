"""Pluggable shared driving/embodied vision encoders (Alpamayo2-Edge, DM0, DINOv3)."""

from open_reality_view.encoders.alpamayo2_edge import Alpamayo2EdgeVisionEncoder
from open_reality_view.encoders.base import SharedVisionEncoder, VisionEncoderOutput

__all__ = [
    "Alpamayo2EdgeVisionEncoder",
    "SharedVisionEncoder",
    "VisionEncoderOutput",
]
