#!/usr/bin/env python
"""
Streaming CNN Training: Real-time Causal Inference

Week 3: Trains StreamingCNN for online gravitational wave detection.
Causal convolutions ensure no future information is used during inference.

Usage:
    python scripts/train_streaming.py --num-samples 500 --epochs 20
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

# Add package to path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from ligo_gw.data.injection import InjectionGenerator
from ligo_gw.models.streaming_cnn import StreamingCNN
from ligo_gw.analysis.metrics import compute_roc_auc, compute_f1


def prepare_streaming_dataset(num_samples=500, signal_duration=1.0, use_real_ligo_noise=True):
    """
    Generate synthetic dataset for streaming inference.
    
    Instead of 2D spectrograms, use temporal 1D sequences.
    Each sample is a 1-second strain with 4096 samples at 4kHz.
    
    Returns:
        X: (num_samples, 1, 4096) 1D strain sequences
        y: (num_samples,) binary labels {0: noise, 1: signal}
    """
    gen = InjectionGenerator()
    
    X_list = []
    y_list = []
    
    # Generate equal numbers of signal and noise
    n_signal = num_samples // 2
    n_noise = num_samples - n_signal
    
    print(f"[Data] Generating {n_signal} signal samples...")
    
    # Signal samples
    for i in range(n_signal):
        # Random parameters
        m1 = np.random.uniform(10, 60)
        m2 = np.random.uniform(10, min(m1, 60))
        
        # Generate chirp
        strain = gen.generate_chirp(
            duration=signal_duration,
            m1=m1, m2=m2,
            sampling_rate=4096
        )
        
        X_list.append(strain)
        y_list.append(1)  # Signal
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_signal}]")
    
    print(f"[Data] Generating {n_noise} noise samples...")
    
    # Noise samples
    for i in range(n_noise):
        # Generate pure noise
        noise = gen._generate_colored_noise(
            duration=signal_duration,
            sampling_rate=4096,
            use_real_ligo_noise=use_real_ligo_noise
        )
        
        X_list.append(noise)
        y_list.append(0)  # Noise
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_noise}]")
    
    # Stack into tensors (need to reshape for 1D input)
    X = np.stack(X_list, axis=0)  # (num_samples, 4096)
    # Reshape to (num_samples, 1, 4096) - 1D temporal input
    # Downsample to match StreamingCNN expected input (1, 128, 127) by using STFT-like resolution
    # Actually, stream model takes (batch, 1, seq_len) temporal inputs
    X = X[:, np.newaxis, :]  # (num_samples, 1, 4096)
    
    y = np.array(y_list)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    
    print(f"[Data] Dataset shape: {X.shape}")
    print(f"[Data] Signal/Noise ratio: {(y==1).sum()}/{(y==0).sum()}")
    
    return X, y


def train_streaming(model, train_loader, val_loader, epochs=20, lr=0.001, 
                   patience=5, device='cpu'):
    """
    Streaming CNN training loop.
    
    Args:
        patience: Early stopping patience on validation AUC
    
    Returns:
        training_history: dict with epoch-wise metrics
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_f1': [],
        'best_epoch': 0,
        'best_val_auc': 0
    }
    
    patience_counter = 0
    best_val_auc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.long().to(device)
            
            # Forward
            logits = model(X_batch)
            
            # Loss
            loss = criterion(logits, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_y_true = []
        val_y_prob = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.long().to(device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_losses.append(loss.item())
                
                val_y_true.append(y_batch.cpu().numpy())
                val_y_prob.append(
                    torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                )
        
        # Aggregate metrics
        val_y_true = np.concatenate(val_y_true)
        val_y_prob = np.concatenate(val_y_prob)
        
        val_auc = roc_auc_score(val_y_true, val_y_prob)
        val_f1 = f1_score(val_y_true, (val_y_prob > 0.5).astype(int))
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            history['best_epoch'] = epoch
            history['best_val_auc'] = val_auc
        else:
            patience_counter += 1
        
        # Print
        print(
            f"[Epoch {epoch+1:2d}/{epochs}] "
            f"Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Val Loss: {history['val_loss'][-1]:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )
        
        scheduler.step()
        
        if patience_counter >= patience:
            print(f"\n[Early Stopping] Patience exceeded at epoch {epoch+1}")
            break
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Streaming GW detection training")
    parser.add_argument('--num-samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--signal-duration', type=float, default=1.0, help='Signal duration in seconds')
    parser.add_argument('--use-real-ligo-noise', action='store_true', help='Use realistic LIGO noise')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n[Config] Device: {device}")
    print(f"[Config] Samples: {args.num_samples}, Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    # Data
    print("\n[Stage 1/4] Preparing dataset...")
    X, y = prepare_streaming_dataset(
        num_samples=args.num_samples,
        signal_duration=args.signal_duration,
        use_real_ligo_noise=args.use_real_ligo_noise
    )
    
    # Train/val/test split (64/16/20)
    n_train = int(0.64 * len(X))
    n_val = int(0.16 * len(X))
    
    X_train = torch.tensor(X[:n_train], dtype=torch.float32)
    y_train = torch.tensor(y[:n_train], dtype=torch.long)
    
    X_val = torch.tensor(X[n_train:n_train+n_val], dtype=torch.float32)
    y_val = torch.tensor(y[n_train:n_train+n_val], dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    print("\n[Stage 2/4] Initializing model...")
    model = StreamingCNN().to(device)
    print(f"[Model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n[Stage 3/4] Training...")
    start_time = datetime.now()
    
    history = train_streaming(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        patience=args.patience,
        device=device
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[Time] Training completed in {elapsed:.1f}s")
    
    # Save
    print("\n[Stage 4/4] Saving model and history...")
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    model_path = checkpoint_dir / 'streaming_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"[Saved] Model checkpoint: {model_path}")
    
    history_path = checkpoint_dir / 'streaming_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[Saved] Training history: {history_path}")
    
    # Summary
    print("\n" + "="*60)
    print("[Summary]")
    print(f"  Best Epoch: {history['best_epoch'] + 1}")
    print(f"  Best Val AUC: {history['best_val_auc']:.4f}")
    print(f"  Final Val F1: {history['val_f1'][-1]:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
