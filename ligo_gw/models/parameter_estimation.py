"""
Parameter Estimation CNN for Gravitational Wave Properties

Week 4: Multi-task learning combining binary classification + regression.
Estimates: m1, m2 (component masses) and SNR (signal-to-noise ratio)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ParameterEstimationCNN(nn.Module):
    """
    Multi-task CNN: simultaneous detection and parameter estimation.
    
    Shared feature extraction backbone with separate classification and regression heads.
    
    Input: (batch, 1, 128, 127) spectrogram
    Outputs:
        - classification: (batch, 2) [P(noise), P(signal)]
        - m1_pred: (batch, 1) primary mass [solar masses]
        - m2_pred: (batch, 1) secondary mass [solar masses]
        - snr_pred: (batch, 1) signal-to-noise ratio [SNR]
    
    Training: Combined loss = CE(classification) + MSE(m1) + MSE(m2) + MSE(SNR)
    """
    
    def __init__(self):
        super().__init__()
        
        # Shared convolutional backbone (same as BaselineCNN)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head (detection)
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # [P(noise), P(signal)]
        )
        
        # Regression head (m1)
        self.m1_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # Regression head (m2)
        self.m2_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # Regression head (SNR)
        self.snr_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, 128, 127) spectrogram
        
        Returns:
            logits: (batch, 2) classification logits
            m1: (batch, 1) m1 predictions [Solar masses]
            m2: (batch, 1) m2 predictions [Solar masses]
            snr: (batch, 1) SNR predictions
        """
        # Shared backbone
        features = self.conv_block1(x)  # (batch, 32, 64, 63)
        features = self.conv_block2(features)  # (batch, 64, 32, 31)
        features = self.conv_block3(features)  # (batch, 128, 16, 15)
        
        features = self.global_pool(features)  # (batch, 128, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 128)
        
        # Multi-task outputs
        logits = self.classification_head(features)  # (batch, 2)
        m1 = self.m1_head(features)  # (batch, 1)
        m2 = self.m2_head(features)  # (batch, 1)
        snr = self.snr_head(features)  # (batch, 1)
        
        return logits, m1, m2, snr
    
    def get_detection_confidence(self, logits):
        """Extract P(signal)."""
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]
    
    def count_parameters(self):
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ParameterEstimator:
    """Wrapper for parameter estimation inference."""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = ParameterEstimationCNN().to(device)
        
        if model_path:
            checkpoint = torch.load(model_path, weights_only=False)
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def predict(self, X):
        """
        Batch inference with parameter estimation.
        
        Args:
            X: (batch, 1, 128, 127) spectrogram tensor or numpy array
        
        Returns:
            dict with keys:
                - 'detections': (batch,) binary labels
                - 'signal_prob': (batch,) P(signal)
                - 'm1': (batch,) primary mass [Solar masses]
                - 'm2': (batch,) secondary mass [Solar masses]
                - 'snr': (batch,) signal-to-noise ratio
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X = X.to(self.device)
        
        with torch.no_grad():
            logits, m1, m2, snr = self.model(X)
            signal_prob = self.model.get_detection_confidence(logits)
            detections = (signal_prob > 0.5).long().cpu().numpy()
            
            return {
                'detections': detections,
                'signal_prob': signal_prob.cpu().numpy(),
                'm1': m1.squeeze(1).cpu().numpy(),
                'm2': m2.squeeze(1).cpu().numpy(),
                'snr': snr.squeeze(1).cpu().numpy()
            }
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint)


def multi_task_loss(logits, m1_pred, m2_pred, snr_pred, 
                    labels, m1_true, m2_true, snr_true,
                    weight_ce=1.0, weight_reg=0.5):
    """
    Combined loss for multi-task learning.
    
    Args:
        weight_ce: Cross-entropy weight
        weight_reg: Regression loss weight (applied to each regression task)
    
    Returns:
        Total loss
    """
    # Classification loss
    ce_loss = F.cross_entropy(logits, labels)
    
    # Regression losses (MSE)
    m1_loss = F.mse_loss(m1_pred.squeeze(), m1_true)
    m2_loss = F.mse_loss(m2_pred.squeeze(), m2_true)
    snr_loss = F.mse_loss(snr_pred.squeeze(), snr_true)
    
    # Combined
    total_loss = weight_ce * ce_loss + weight_reg * (m1_loss + m2_loss + snr_loss) / 3.0
    
    return {
        'total': total_loss,
        'ce': ce_loss.item(),
        'm1': m1_loss.item(),
        'm2': m2_loss.item(),
        'snr': snr_loss.item()
    }
