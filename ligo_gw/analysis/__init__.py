"""Analysis and evaluation modules."""

from .metrics import ROCAnalyzer, ParameterEstimator
from .visualization import plot_scalogram, plot_detection

__all__ = ["ROCAnalyzer", "ParameterEstimator", "plot_scalogram", "plot_detection"]
