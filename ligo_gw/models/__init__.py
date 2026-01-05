"""Neural network models for GW detection."""

from .baseline_cnn import BaselineCNN
from .advanced_models import ResNet1D, TCNDetector

__all__ = ["BaselineCNN", "ResNet1D", "TCNDetector"]
