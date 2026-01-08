# LIGO Gravitational Wave Detection Pipeline

Neural network framework for gravitational wave detection and parameter estimation. Designed for speed

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Vision

- **Real-time detection** on LIGO data streams with <1s latency
- **Parameter estimation** for source characterization (m1, m2, SNR)
- **Open benchmark** enabling reproducible GW ML research
- **Deployment-ready** code for LIGO analysis infrastructure

---

## Repository Structure

```
ligo-gw-detection/
├── ligo_gw/                 # Main package
│   ├── data/                # Data loaders, transforms, injections
│   │   ├── loaders.py       # FITS loading, whitening
│   │   ├── transforms.py    # Wavelet (CWT), STFT
│   │   └── injection.py     # Synthetic GW signal generation
│   ├── models/              # Neural network architectures
│   │   ├── baseline_cnn.py  # Simple CNN (Week 2)
│   │   └── advanced_models.py # ResNet, TCN (future)
│   ├── inference/           # Detection and inference
│   │   ├── detector.py      # Offline detection
│   │   └── streaming.py     # Real-time streaming (Week 3)
│   └── analysis/            # Evaluation metrics
│       ├── metrics.py       # ROC, parameter regression
│       └── visualization.py # Plotting for paper
├── scripts/                 # Executable scripts
│   ├── train.py            # Training script (Week 2)
│   ├── inference.py        # Batch inference (Week 2-3)
│   └── benchmark.py        # Comparison with baselines (future)
├── notebooks/              # Jupyter notebooks (analysis, demos)
├── tests/                  # Unit tests
└── README.md              # This file
```

---

## Milestones & Timeline

### **Week 1: Infrastructure**
- [ ] LIGO strain loader (FITS parsing, whitening)
- [ ] Wavelet transform pipeline (CWT with Morlet)
- [ ] Injection generator (chirp + burst signals)
- [ ] First commit: working data pipeline

### **Week 2: Baseline Detection**
- [ ] Train baseline CNN on injected signals
- [ ] Validate accuracy > existing ML baselines
- [ ] ROC analysis on test set
- [ ] Paper: Methods section (CNN architecture)

### **Week 3: Real-time Inference**
- [ ] Streaming detector with causal convolutions
- [ ] Latency benchmarks on simulated data
- [ ] Demo: GW170817 early-warning simulation
- [ ] Paper: Results + streaming latency plots

### **Week 4: Parameter Regression & Release**
- [ ] Parameter estimator (m1, m2, SNR regression)
- [ ] Draft arXiv paper
- [ ] Open-source release + benchmark dataset
- [ ] Grant-ready documentation (NSF/NASA/ERC)

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/...
cd ligo-gw-detection
pip install -e ".[dev]"
```

### Training a Detector

```bash
python scripts/train.py \
    --data-dir ./data/scalograms \
    --output-dir ./checkpoints \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-3
```

### Running Inference

```bash
python scripts/inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data-dir ./data/test \
    --output-file ./detections.json
```

---

## Key Design Decisions

1. **Time-Frequency Input**: CWT scalograms as input (natural for GW signals)
2. **Lightweight Architecture**: Baseline CNN is fast to train, easy to interpret
3. **Modular Design**: Data / Models / Inference cleanly separated
4. **Test Coverage**: Unit tests for each module (runs on dummy data)
5. **Paper-Ready**: Visualization and metrics tools included

---

## Dependencies

### Core
- `torch` - Model training and inference
- `numpy`, `scipy` - Numerical computing
- `scikit-learn` - Metrics (ROC, confusion matrix)
- `astropy` - FITS file I/O
- `matplotlib` - Plotting

### Development
- `pytest`, `pytest-cov` - Testing
- `jupyter` - Notebooks
- `black`, `flake8` - Code formatting

See `pyproject.toml` for full dependency list.

---

## Usage Examples

### Week 1: Load and Transform Data

```python
from ligo_gw.data import LIGOStrainLoader, CWTMorlet

# Load LIGO strain
loader = LIGOStrainLoader()
strain = loader.load_fits("GW150914.fits")
strain = loader.whiten(strain)

# Apply wavelet transform
cwt = CWTMorlet(min_freq=20, max_freq=2048, num_freqs=128)
scalogram = cwt.transform(strain)
```

### Week 2: Train Detector

```python
from ligo_gw.models import BaselineCNN
from ligo_gw.inference import GWDetector

# Create and train model
model = BaselineCNN()
# ... training loop ...

# Inference
detector = GWDetector(model, threshold=0.5)
detections, confidences = detector.predict(X_test)
```

### Week 3: Stream Real-time Data

```python
from ligo_gw.inference import StreamingDetector

detector = StreamingDetector(
    model,
    window_duration=1.0,
    stride_duration=0.1
)

# Process live data chunk by chunk
while data_available:
    chunk = get_next_chunk()  # ~1.6M samples/sec
    alert = detector.process_chunk(chunk)
    if alert:
        print(f"Detection at {alert['gps_time']}")
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ligo_gw --cov-report=html

# Run specific test file
pytest tests/test_data.py -v
```

---

## Next Steps

- [ ] Implement Week 1 data loaders
- [ ] Collect/generate training data (or use public dataset like GW Open Science Center)
- [ ] Train baseline CNN (Week 2)
- [ ] Benchmark against existing ML approaches (PyCBC, GstLAL)
- [ ] Publish preliminary results

---

## References

- LIGO Open Science Center: https://www.gwopenscience.org/
- PyCBC: https://github.com/gwastro/pycbc
- LALSuite: https://git.ligo.org/lalsuite/lalsuite
- Zenodo LIGO datasets: https://zenodo.org/communities/ligo_virgo

---

## Authors

See CONTRIBUTORS.md

## License

MIT License - See LICENSE file

---

## Authors & Contributors

- **Deepnil Ray** (NoRCEL) - Project lead, architecture design, implementation, experiments

### Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Citation

```bibtex
@article{Ray2026,
  title={Fast Detection of Gravitational Waves with Convolutional Neural Networks},
  author={Ray, Deepnil}}
```
