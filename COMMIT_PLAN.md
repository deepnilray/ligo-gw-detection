# LIGO GW Detection: First Commit Plan

## Initial Commit Strategy

This outlines the **first 4-5 commits** to establish infrastructure for Week 1-2 work.

---

## Commit 1: Project Structure & Scaffolding

**Message:** `Initial project setup with package structure`

**Contents:**
- `pyproject.toml` - Build config, dependencies
- `.gitignore` - Python exclusions
- `README.md` - Project overview and milestones
- `ligo_gw/__init__.py` - Package root
- Empty module directories (placeholders)

**Why first:** Establishes baseline CI/CD ready structure. Can build/install package immediately.

---

## Commit 2: Data Loading Infrastructure

**Message:** `Week 1: LIGO strain loader and whitening pipeline`

**Contents:**
- `ligo_gw/data/loaders.py` - `LIGOStrainLoader` class (interface only)
- `ligo_gw/data/__init__.py` - Export public API
- `tests/test_data.py` - Unit tests for loader interface

**Key methods (stubbed with NotImplementedError):**
- `load_fits()` - Parse FITS files
- `whiten()` - PSD normalization
- `normalize()` - Unit variance scaling
- `get_window()` - Extract time windows

**Why:** Defines data input contract. Tests run (on dummy data). Ready for implementation.

---

## Commit 3: Wavelet Transform Pipeline

**Message:** `Week 1: CWT and STFT scalogram transforms`

**Contents:**
- `ligo_gw/data/transforms.py` - `CWTMorlet`, `STFTSpectrogram` classes
- Extension of `tests/test_data.py` with wavelet tests

**Key methods:**
- `CWTMorlet.transform()` - Continuous wavelet transform
- `CWTMorlet.to_db()` - dB scaling
- `STFTSpectrogram.transform()` - STFT alternative

**Output shape:** (num_freqs, num_time_samples) for CNN input

**Why:** Defines feature extraction pipeline. Interface locked down before implementation.

---

## Commit 4: Signal Injection & Data Generation

**Message:** `Week 1: Synthetic GW injection for augmentation and testing`

**Contents:**
- `ligo_gw/data/injection.py` - `InjectionGenerator`, `GWSignal` dataclass
- Tests for signal generation

**Key methods:**
- `generate_chirp()` - BBH merger waveforms
- `generate_burst()` - Sine-Gaussian bursts
- `inject_signal()` - Noise + waveform mixing at target SNR
- `generate_batch()` - Random parameter sampling

**Why:** Core for training data generation. Decouples synthetic signal creation from FITS loading.

---

## Commit 5: Baseline CNN Model

**Message:** `Week 2: Baseline CNN for signal classification`

**Contents:**
- `ligo_gw/models/baseline_cnn.py` - `BaselineCNN` class (fully implemented)
- `ligo_gw/models/__init__.py` - Export API
- `tests/test_models.py` - Unit tests (working tests with real forward passes)

**Architecture:**
- 3 conv blocks (Conv2D + BatchNorm + ReLU + MaxPool)
- Global average pooling
- 2-layer MLP classifier (128 -> 64 -> 2)
- ~50k parameters (fast iteration)

**Key methods:**
- `forward()` - Forward pass
- `get_detection_confidence()` - Softmax probability extraction
- `count_parameters()` - Model size

**Why:** First trainable model. Not implementing yet, but code is production-ready.

---

## Commit 6: Inference & Detection

**Message:** `Week 2-3: Offline detector and streaming inference stubs`

**Contents:**
- `ligo_gw/inference/detector.py` - `GWDetector` class (interface mostly stubbed)
- `ligo_gw/inference/streaming.py` - `StreamingDetector` (Week 3 stub)
- `tests/test_inference.py` - Working tests for detector

**Key methods:**
- `GWDetector.predict()` - Batch inference
- `GWDetector.save()` / `load()` - Checkpoint persistence
- `StreamingDetector.process_chunk()` - Real-time inference (Week 3)

**Why:** Locks inference API. Can write training script that uses this interface immediately.

---

## Commit 7: Analysis & Metrics

**Message:** `Evaluation tools: ROC analysis and parameter estimation stubs`

**Contents:**
- `ligo_gw/analysis/metrics.py` - `ROCAnalyzer`, `ParameterEstimator`
- `ligo_gw/analysis/visualization.py` - Plot functions (`plot_scalogram`, `plot_detection`)
- Tests for metrics

**Key classes:**
- `ROCAnalyzer` - AUC computation, confusion matrix, benchmark metrics
- `ParameterEstimator` - Stub for Week 4 parameter regression

**Why:** Paper-ready analysis pipeline. Metrics computed directly from model outputs.

---

## Commit 8: Training & Inference Scripts

**Message:** `Week 2: Training loop and batch inference scripts`

**Contents:**
- `scripts/train.py` - Full training loop with dummy data
- `scripts/inference.py` - Inference on test set, JSON output
- `scripts/benchmark.py` - Comparison framework (stub)

**Key features:**
- Argparse CLI interface
- Early stopping + LR scheduling
- Checkpoint saving/loading
- JSON output for downstream analysis

**Why:** Users can run `python train.py` immediately and see training work (with dummy data).

---

## Commit 9: Tests & CI Setup

**Message:** `Test suite and CI configuration`

**Contents:**
- `.github/workflows/tests.yml` - GitHub Actions CI
- Complete `tests/` directory with all unit tests passing
- `pytest.ini` configuration

**CI checks:**
- Unit test suite (runs on dummy data)
- Code linting (flake8, black)
- Type checking (mypy)
- Coverage report

**Why:** All commits after this point automatically verified. Safe for parallel work.

---

## Commit 10: Notebooks Template

**Message:** `Analysis notebooks and demo templates`

**Contents:**
- `notebooks/01_data_exploration.ipynb` - Load and visualize scalograms
- `notebooks/02_model_architecture.ipynb` - Visualize CNN layers
- `notebooks/03_training_results.ipynb` - Plot training curves

**Why:** Entry point for paper figures. Week 2-4 results plots here.

---

## Summary: Commit Dependencies

```
1. Project setup
   └─> 2. Loaders
   └─> 3. Transforms
   └─> 4. Injection
   └─> 5. CNN Model
   └─> 6. Inference
   └─> 7. Analysis
   └─> 8. Scripts
       └─> 9. CI/Tests
           └─> 10. Notebooks
```

**Key property:** Commits 2-8 are **mostly interface definitions** (NotImplementedError stubs). Tests run on dummy data. This allows:
- Parallel Week 1 implementation work (assign different modules to different people)
- Week 2 training script ready to use as soon as data loaders are implemented
- Clear API contracts that won't change mid-project

---

## After Commit 10: Week 1-2 Workflow

```
Branch: week-1-implementation
  ├─ dev/data-loaders (implements ligo_gw/data/loaders.py + tests)
  ├─ dev/transforms (implements ligo_gw/data/transforms.py + tests)
  ├─ dev/injection (implements ligo_gw/data/injection.py + tests)
  └─ Merge → main: Commit 11: "Week 1 complete: data pipeline implemented"

Branch: week-2-training
  ├─ Merge week-1-implementation
  ├─ Implement model.forward() fully (currently has working stub)
  ├─ Train on real/synthetic data
  ├─ → Commit 12: "Week 2: Baseline CNN beats ML baselines"
  └─ Draft methods paper section
```

---

## File Changes by Commit

| Commit | Files | LOC | Purpose |
|--------|-------|-----|---------|
| 1 | 4 | 150 | Setup |
| 2 | 4 | 200 | Loaders |
| 3 | 2 | 180 | Transforms |
| 4 | 2 | 200 | Injection |
| 5 | 3 | 300 | CNN |
| 6 | 3 | 200 | Inference |
| 7 | 3 | 250 | Analysis |
| 8 | 3 | 600 | Scripts |
| 9 | 5 | 200 | Tests/CI |
| 10 | 3 | 500 | Notebooks |
| **Total** | **30** | **~2700** | **Production-ready scaffold** |

All 10 commits ready to go now.
