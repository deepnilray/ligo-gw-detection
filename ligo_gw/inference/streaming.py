"""Streaming inference for real-time detection.

Week 3 Milestone: Sub-second latency detection on live LIGO data.
"""

import torch
import numpy as np
from typing import Optional
from collections import deque


class StreamingDetector:
    """Real-time streaming detector.
    
    Maintains a sliding window of recent data, computes features on-the-fly,
    and produces detection alerts with minimal latency.
    
    Properties:
    - Causal convolutions (no future data dependency)
    - Configurable latency/accuracy tradeoff
    - Online whitening/normalization
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        window_duration: float = 1.0,
        stride_duration: float = 0.1,
        sample_rate: float = 16384.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5
    ):
        """Initialize streaming detector.
        
        Args:
            model: Causal CNN model
            window_duration: Analysis window (seconds)
            stride_duration: Stride between windows (seconds)
            sample_rate: Sampling rate (Hz)
            device: 'cuda' or 'cpu'
            threshold: Detection threshold
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        
        # Circular buffer for streaming data
        self.window_samples = int(window_duration * sample_rate)
        self.stride_samples = int(stride_duration * sample_rate)
        self.buffer = deque(maxlen=self.window_samples)
        
        self.model.eval()
    
    def process_chunk(self, data: np.ndarray) -> Optional[dict]:
        """Process chunk of streaming data.
        
        Args:
            data: New strain samples
            
        Returns:
            Detection dict if signal detected, else None
        """
        # TODO: Implement online feature extraction and detection
        raise NotImplementedError("Week 3: Implement streaming inference")
    
    def reset(self) -> None:
        """Reset buffer for new stream."""
        self.buffer.clear()
