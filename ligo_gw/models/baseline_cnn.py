"""Baseline CNN for GW detection.

Week 2 Milestone: Simple but effective CNN that beats existing ML baselines.
"""

import torch
import torch.nn as nn
from typing import Optional


class BaselineCNN(nn.Module):
    """Simple CNN for binary GW presence classification.
    
    Architecture:
    - Input: time-frequency scalogram (128 x T)
    - Stack of Conv2D + BatchNorm + ReLU + MaxPool
    - Global average pooling
    - Dense classifier (2 outputs: signal/noise)
    
    Design for:
    - Fast training (Week 2)
    - Easy interpretation
    - Streaming inference (Week 3)
    - Parameter tuning baseline
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_freqs: int = 128,
        dropout_rate: float = 0.3,
        num_classes: int = 2
    ):
        """Initialize baseline CNN.
        
        Args:
            input_channels: Input channels (1 for single scalogram)
            num_freqs: Number of frequency bins
            dropout_rate: Dropout probability
            num_classes: Number of output classes (2 for binary classification)
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_freqs = num_freqs
        
        # Convolutional backbone
        self.conv_stack = nn.Sequential(
            # Block 1: (1, 128, T) -> (32, 64, T/2)
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout_rate),
            
            # Block 2: (32, 64, T/2) -> (64, 32, T/4)
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout_rate),
            
            # Block 3: (64, 32, T/4) -> (128, 16, T/8)
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout_rate),
        )
        
        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, channels, num_freqs, time_steps)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Convolutional feature extraction
        features = self.conv_stack(x)  # (batch, 128, freq_reduced, time_reduced)
        
        # Global pooling
        pooled = self.global_pool(features)  # (batch, 128, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # (batch, 128)
        
        # Classification
        logits = self.classifier(flattened)
        
        return logits
    
    def get_detection_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Get detection probability (probability of signal class).
        
        Args:
            logits: Raw output from forward()
            
        Returns:
            Probability of signal (0-1)
        """
        return torch.softmax(logits, dim=1)[:, 1]  # Class 1 = signal
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
