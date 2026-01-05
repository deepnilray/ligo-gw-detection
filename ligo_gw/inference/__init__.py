"""Inference and deployment modules."""

from .detector import GWDetector
from .streaming import StreamingDetector

__all__ = ["GWDetector", "StreamingDetector"]
