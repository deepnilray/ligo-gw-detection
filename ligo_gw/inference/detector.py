"""Offline detection inference.

Week 2-3: Detection on fixed datasets, then streaming.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class GWDetector:
    """Inference wrapper for GW detection models.
    
    Handles:
    - Model loading and GPU management
    - Batch inference
    - Post-processing (thresholding, NMS)
    - ROC curve computation
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5
    ):
        """Initialize detector.
        
        Args:
            model: Trained PyTorch model
            device: 'cuda' or 'cpu'
            threshold: Detection threshold (0-1)
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.model.eval()
    
    def predict(
        self,
        scalograms: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Detect signals in batch of scalograms.
        
        Args:
            scalograms: (batch_size, 1, num_freqs, time_steps)
            return_confidence: Also return detection confidence
            
        Returns:
            (detections, confidences) where detections are binary labels
        """
        with torch.no_grad():
            x = torch.from_numpy(scalograms).float().to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(signal)
        
        detections = (probs > self.threshold).cpu().numpy().astype(int)
        
        if return_confidence:
            return detections, probs.cpu().numpy()
        return detections
    
    def save(self, checkpoint_path: str) -> None:
        """Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        # TODO: Implement full checkpoint saving
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
