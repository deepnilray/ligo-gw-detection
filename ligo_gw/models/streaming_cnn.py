"""
Streaming CNN with Causal Convolutions for Real-Time GW Detection

Week 3: Low-latency inference suitable for early-warning systems.
Uses causal padding to ensure causality (no future information leakage).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D Convolution with causal padding (no future information)."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding, **kwargs
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        x = self.conv(x)
        # Remove future padding
        if self.kernel_size > 1:
            x = x[:, :, :-(self.kernel_size - 1) * self.dilation]
        return x


class StreamingCNN(nn.Module):
    """
    Causal CNN for streaming GW detection.
    
    Processes 1-second spectrograms with 256-sample windows (15.6 ms @ 16384 Hz).
    Suitable for online processing with minimal latency.
    
    Input: (batch, 1, 128, 127) — spectrogram
    Output: (batch, 2) — [P(noise), P(signal)]
    Latency: ~5 ms (STFT) + ~2 ms (inference) = ~7 ms per window
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Flatten spectrogram to sequence: (batch, 128, 127) → (batch, 128*127)
        # Then treat as temporal sequence for causal convolutions
        
        # Temporal blocks with causal convolutions
        self.temporal_conv1 = CausalConv1d(128, 64, kernel_size=3, dilation=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.temporal_conv2 = CausalConv1d(64, 64, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.temporal_conv3 = CausalConv1d(64, 128, kernel_size=3, dilation=4)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(0.3)
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: (batch, 1, 128, 127) spectrogram
        batch_size = x.shape[0]
        
        # Reshape: flatten frequency bins, keep time as sequence
        # (batch, 1, 128, 127) → (batch, 128, 127)
        x = x.squeeze(1)
        
        # Causal temporal convolutions
        x = F.relu(self.bn1(self.temporal_conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.temporal_conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.temporal_conv3(x)))
        x = self.dropout(x)
        
        # Global pooling over time
        x = self.pool(x)  # (batch, 128, 1)
        x = x.view(batch_size, -1)  # (batch, 128)
        
        # Classification
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.fc_out(x)  # (batch, 2)
        
        return x
    
    def get_detection_confidence(self, logits):
        """Extract probability of signal (class 1)."""
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]  # P(signal)
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StreamingDetector:
    """Wrapper for streaming inference with state management."""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = StreamingCNN().to(device)
        
        if model_path:
            checkpoint = torch.load(model_path, weights_only=False)
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def predict(self, X):
        """
        Batch inference on spectrograms.
        
        Args:
            X: (batch, 1, 128, 127) tensor
        
        Returns:
            detections: (batch,) binary labels
            confidences: (batch,) signal probabilities
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X)
            confidences = self.model.get_detection_confidence(logits)
            detections = (confidences > 0.5).long().cpu().numpy()
            confidences = confidences.cpu().numpy()
        
        return detections, confidences
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint)
