"""LIGO Gravitational Wave Detection Pipeline

A neural network-based inference and analysis framework for gravitational wave detection.
"""

__version__ = "0.1.0"
__author__ = "LIGO ML Collaboration"

from . import data, models, inference, analysis

__all__ = ["data", "models", "inference", "analysis"]
