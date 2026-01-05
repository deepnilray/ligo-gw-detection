"""Advanced architectures for future weeks.

Week 2+: More sophisticated models with parameter regression.
"""

import torch
import torch.nn as nn


class ResNet1D(nn.Module):
    """1D ResNet for strain-domain processing.
    
    TODO: Implement for direct strain input (no scalogram preprocessing).
    """
    pass


class TCNDetector(nn.Module):
    """Temporal Convolutional Network for streaming inference.
    
    TODO: Implement causal convolutions for Week 3 latency benchmarks.
    """
    pass
