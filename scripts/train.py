"""Training script template.

Week 2 Milestone: Training baseline CNN to exceed existing ML baselines.

Usage:
    python train.py \
        --output-dir ./checkpoints \
        --epochs 50 \
        --batch-size 32 \
        --learning-rate 1e-3
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from ligo_gw.models import BaselineCNN
from ligo_gw.inference import GWDetector
from ligo_gw.analysis import ROCAnalyzer
from ligo_gw.data import InjectionGenerator, CWTMorlet


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    num_samples: int = 2000,
    signal_duration: float = 1.0,
    sample_rate: float = 16384.0,
    use_real_ligo_noise: bool = True
) -> tuple:
    """Generate synthetic dataset with injected GW signals and noise.
    
    Args:
        num_samples: Total number of samples
        signal_duration: Duration of each sample (seconds)
        sample_rate: Sampling rate (Hz)
        use_real_ligo_noise: Use real LIGO noise instead of Gaussian
        
    Returns:
        (X_scalograms, y_labels) - scalograms and binary labels
    """
    logger.info(f"Generating {num_samples} synthetic signals...")
    if use_real_ligo_noise:
        logger.info("  (using realistic LIGO detector noise)")
    
    # Generate strains with injected signals
    injector = InjectionGenerator(sample_rate=sample_rate)
    strains, labels = injector.generate_batch(
        num_samples,
        background_duration=signal_duration,
        mass_range=(10.0, 60.0),
        snr_range=(8.0, 50.0),
        signal_type='chirp',
        use_real_ligo_noise=use_real_ligo_noise
    )
    
    # Convert to time-frequency representation using STFT (faster than CWT on CPU)
    logger.info("Computing STFT spectrograms...")
    from ligo_gw.data import STFTSpectrogram
    
    stft = STFTSpectrogram(
        window='hann',
        nperseg=256,
        sample_rate=sample_rate
    )
    
    scalograms = []
    for strain in strains:
        # Compute STFT
        freqs, times, spectrogram = stft.transform(np.array(strain, dtype=np.float64))
        
        # Resample to fixed frequency grid (20-2048 Hz, 128 bins)
        from scipy.interpolate import interp1d
        
        freq_target = np.logspace(np.log10(20), np.log10(2048), 128)
        
        # Interpolate each time frame
        spec_resampled = np.zeros((len(freq_target), spectrogram.shape[1]))
        for t in range(spectrogram.shape[1]):
            if np.max(np.abs(spectrogram[:, t])) > 0:
                interp_func = interp1d(
                    freqs,
                    spectrogram[:, t],
                    kind='linear',
                    bounds_error=False,
                    fill_value=0
                )
                spec_resampled[:, t] = interp_func(freq_target)
        
        # Convert to dB
        spec_db = 10 * np.log10(spec_resampled + 1e-10)
        scalograms.append(spec_db)
    
    # Stack into numpy array
    X = np.stack(scalograms, axis=0)  # (num_samples, num_freqs, num_times)
    X = np.expand_dims(X, axis=1)  # Add channel dimension: (num_samples, 1, num_freqs, num_times)
    
    y = np.array(labels, dtype=np.int64)
    
    logger.info(f"Dataset shape: {X.shape}, Labels: {np.bincount(y)}")
    
    return X.astype(np.float32), y


def main(args):
    """Training loop."""
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Generate synthetic data
    logger.info("\n=== Data Generation ===")
    X_all, y_all = generate_synthetic_dataset(
        num_samples=args.num_samples,
        signal_duration=args.signal_duration,
        use_real_ligo_noise=args.use_real_ligo_noise
    )
    
    # Split into train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Training set: {X_train.shape}, {np.mean(y_train):.1%} signals")
    logger.info(f"Validation set: {X_val.shape}, {np.mean(y_val):.1%} signals")
    logger.info(f"Test set: {X_test.shape}, {np.mean(y_test):.1%} signals")
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).long()
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model setup
    logger.info("\n=== Model Setup ===")
    model = BaselineCNN(input_channels=1, num_freqs=128)
    model.to(device)
    
    logger.info(f"Model has {model.count_parameters():,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Training loop
    logger.info("\n=== Training ===")
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_analyzer = ROCAnalyzer()
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]
                
                val_analyzer.add_batch(y.cpu().numpy(), probs.cpu().numpy())
        
        _, _, val_auc = val_analyzer.compute_roc()
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = output_dir / f"best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  -> Saved checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    # Test evaluation
    logger.info("\n=== Test Set Evaluation ===")
    
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    detector = GWDetector(model, device=device, threshold=0.5)
    
    detections, confidences = detector.predict(X_test)
    
    test_analyzer = ROCAnalyzer()
    test_analyzer.add_batch(y_test, confidences)
    fpr, tpr, test_auc = test_analyzer.compute_roc()
    
    metrics = test_analyzer.compute_metrics_at_threshold(0.5)
    
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    logger.info("\n=== Training Complete ===")
    logger.info(f"Best model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline GW detector")
    
    parser.add_argument("--output-dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--signal-duration", type=float, default=1.0, help="Duration of each signal (seconds)")
    parser.add_argument("--use-real-ligo-noise", action="store_true", help="Use realistic LIGO noise (no GB download)")
    
    args = parser.parse_args()
    main(args)
