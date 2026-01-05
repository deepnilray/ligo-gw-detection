# Quick Start

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ligo-gw-detection.git
cd ligo-gw-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

## Training

```bash
# Quick test (100 samples, 5 epochs)
python scripts/train.py --num-samples 100 --epochs 5

# Production training (1000 samples, 20 epochs, realistic LIGO noise)
python scripts/train.py --num-samples 1000 --epochs 20 --use-real-ligo-noise

# Specify other parameters
python scripts/train.py \
  --num-samples 5000 \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --signal-duration 1.0 \
  --use-real-ligo-noise
```

## Inference

```python
import torch
from ligo_gw.models.baseline_cnn import BaselineCNN
from ligo_gw.inference.detector import GWDetector

# Load trained model
detector = GWDetector(model_path='checkpoints/best_model.pt', device='cpu')

# Predict on spectrogram (batch_size, 1, 128, 127)
predictions, confidences = detector.predict(X_test)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=ligo_gw --cov-report=html
```

## Documentation

- **Methods**: See [METHODS_PAPER.md](papers/METHODS_PAPER.md)
- **Development Plan**: See [COMMIT_PLAN.md](COMMIT_PLAN.md)
- **Full Details**: See [README.md](README.md)

## Citation

```bibtex
@article{ligo-gw-detection,
  title={Fast Detection of Gravitational Waves with Convolutional Neural Networks},
  author={LIGO ML Collaboration},
  journal={arXiv},
  year={2026}
}
```
