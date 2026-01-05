"""Tests for inference modules.

Week 2-3: Verify detector and streaming inference.
"""

import pytest
import torch
import numpy as np
from ligo_gw.models import BaselineCNN
from ligo_gw.inference import GWDetector


class TestGWDetector:
    """Test detection inference."""
    
    @pytest.fixture
    def model(self):
        """Create model."""
        return BaselineCNN()
    
    @pytest.fixture
    def detector(self, model):
        """Create detector."""
        return GWDetector(model, device="cpu", threshold=0.5)
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.threshold == 0.5
    
    def test_prediction(self, detector):
        """Test batch prediction."""
        # Dummy scalograms: (batch=4, channels=1, freqs=128, time=256)
        scalograms = np.random.randn(4, 1, 128, 256).astype(np.float32)
        
        detections, confidences = detector.predict(scalograms)
        
        assert detections.shape == (4,)
        assert confidences.shape == (4,)
        assert np.all((detections == 0) | (detections == 1))
        assert np.all((confidences >= 0) & (confidences <= 1))
