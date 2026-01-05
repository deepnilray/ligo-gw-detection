"""Tests for models.

Week 2: Verify CNN architecture and forward pass.
"""

import pytest
import torch
import numpy as np
from ligo_gw.models import BaselineCNN


class TestBaselineCNN:
    """Test baseline CNN model."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return BaselineCNN(
            input_channels=1,
            num_freqs=128,
            dropout_rate=0.3,
            num_classes=2
        )
    
    @pytest.fixture
    def dummy_input(self):
        """Create dummy input (batch_size=4, channels=1, freqs=128, time=256)."""
        return torch.randn(4, 1, 128, 256)
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.input_channels == 1
        assert model.num_freqs == 128
    
    def test_forward_pass(self, model, dummy_input):
        """Test forward pass."""
        logits = model(dummy_input)
        
        assert logits.shape == (4, 2)  # (batch_size, num_classes)
        assert not torch.isnan(logits).any()
    
    def test_confidence(self, model, dummy_input):
        """Test confidence computation."""
        logits = model(dummy_input)
        confidence = model.get_detection_confidence(logits)
        
        assert confidence.shape == (4,)
        assert (confidence >= 0).all() and (confidence <= 1).all()
    
    def test_parameter_count(self, model):
        """Test parameter counting."""
        num_params = model.count_parameters()
        assert num_params > 0
        print(f"Model has {num_params:,} trainable parameters")
