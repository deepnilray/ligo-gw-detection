"""Tests for data loading modules.

Week 1: Verify LIGO loader, whitening, and injection.
"""

import pytest
import numpy as np
from ligo_gw.data import LIGOStrainLoader, WaveletTransform, InjectionGenerator


class TestLIGOStrainLoader:
    """Test LIGO strain data loading."""
    
    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        return LIGOStrainLoader()
    
    def test_initialization(self, loader):
        """Test loader initialization."""
        assert loader.sample_rate == 16384.0
    
    def test_normalize(self, loader):
        """Test data normalization."""
        data = np.random.randn(100)
        normalized = loader.normalize(data)
        
        assert np.abs(np.mean(normalized)) < 1e-6
        assert np.abs(np.std(normalized) - 1.0) < 1e-2


class TestWaveletTransform:
    """Test wavelet transforms."""
    
    def test_cwt_initialization(self):
        """Test CWT initialization."""
        from ligo_gw.data.transforms import CWTMorlet
        
        cwt = CWTMorlet(min_freq=20, max_freq=2048, num_freqs=128)
        assert len(cwt.freqs) == 128
        assert cwt.freqs[0] >= 20
        assert cwt.freqs[-1] <= 2048


class TestInjectionGenerator:
    """Test GW signal injection."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return InjectionGenerator()
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.sample_rate == 16384.0
