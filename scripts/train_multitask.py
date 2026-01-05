#!/usr/bin/env python
"""
Multi-task Training: Simultaneous Detection + Parameter Estimation

Week 3-4 integration. Trains ParameterEstimationCNN to jointly:
1. Detect gravitational wave signals (binary classification)
2. Estimate component masses (m1, m2) in solar masses
3. Estimate signal-to-noise ratio (SNR)

Usage:
    python scripts/train_multitask.py --num-samples 500 --epochs 20
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
from scipy import signal
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

# Add package to path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from ligo_gw.data.injection import InjectionGenerator
from ligo_gw.data.transforms import STFTSpectrogram
from ligo_gw.models.parameter_estimation import ParameterEstimationCNN, multi_task_loss
from ligo_gw.analysis.metrics import compute_roc_auc, compute_f1


def prepare_multitask_dataset(num_samples=500, signal_duration=1.0, use_real_ligo_noise=True):
    """
    Generate synthetic dataset with parameter labels.
    
    Returns:
        X: (num_samples, 1, 128, 127) spectrogram
        y_label: (num_samples,) binary labels {0: noise, 1: signal}
        y_m1: (num_samples,) true m1 masses
        y_m2: (num_samples,) true m2 masses
        y_snr: (num_samples,) true SNR values
    """
    gen = InjectionGenerator()
    spectrogram_transform = STFTSpectrogram(
        freq_min=20, freq_max=2048, n_freqs=128,
        n_times=127, window_duration=1.0
    )
    
    X_list = []
    y_label_list = []
    y_m1_list = []
    y_m2_list = []
    y_snr_list = []
    
    # Generate equal numbers of signal and noise
    n_signal = num_samples // 2
    n_noise = num_samples - n_signal
    
    print(f"[Data] Generating {n_signal} signal samples...")
    
    # Signal samples
    for i in range(n_signal):
        # Random parameters
        m1 = np.random.uniform(10, 60)
        m2 = np.random.uniform(10, min(m1, 60))
        snr = np.random.uniform(8, 50)
        
        # Generate chirp
        strain = gen.generate_chirp(
            duration=signal_duration,
            m1=m1, m2=m2,
            sampling_rate=4096
        )
        
        # Get spectrogram
        spec = spectrogram_transform(strain)
        X_list.append(spec)
        y_label_list.append(1)  # Signal
        y_m1_list.append(m1)
        y_m2_list.append(m2)
        y_snr_list.append(snr)
        
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
        
        # Get spectrogram
        spec = spectrogram_transform(noise)
        X_list.append(spec)
        y_label_list.append(0)  # Noise
        y_m1_list.append(0.0)  # Placeholder (noise has no mass)
        y_m2_list.append(0.0)
        y_snr_list.append(0.0)
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_noise}]")
    
    # Stack into tensors
    X = np.stack(X_list, axis=0)  # (num_samples, 1, 128, 127)
    y_label = np.array(y_label_list)
    y_m1 = np.array(y_m1_list)
    y_m2 = np.array(y_m2_list)
    y_snr = np.array(y_snr_list)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y_label = y_label[idx]
    y_m1 = y_m1[idx]
    y_m2 = y_m2[idx]
    y_snr = y_snr[idx]
    
    print(f"[Data] Dataset shape: {X.shape}")
    print(f"[Data] Signal/Noise ratio: {(y_label==1).sum()}/{(y_label==0).sum()}")
    
    return X, y_label, y_m1, y_m2, y_snr


def train_multitask(model, train_loader, val_loader, epochs=20, lr=0.001, 
                   weight_ce=1.0, weight_reg=0.5, patience=5, device='cpu'):
    """
    Multi-task training loop.
    
    Args:
        weight_ce: Classification loss weight
        weight_reg: Regression loss weight per task
        patience: Early stopping patience on validation AUC
    
    Returns:
        training_history: dict with epoch-wise metrics
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [],
        'train_ce': [],
        'train_m1': [],
        'train_m2': [],
        'train_snr': [],
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
        train_losses = {'total': [], 'ce': [], 'm1': [], 'm2': [], 'snr': []}
        
        for X_batch, y_label_batch, y_m1_batch, y_m2_batch, y_snr_batch in train_loader:
            X_batch = X_batch.to(device)
            y_label_batch = y_label_batch.long().to(device)
            y_m1_batch = y_m1_batch.float().to(device)
            y_m2_batch = y_m2_batch.float().to(device)
            y_snr_batch = y_snr_batch.float().to(device)
            
            # Forward
            logits, m1, m2, snr = model(X_batch)
            
            # Loss
            loss_dict = multi_task_loss(
                logits, m1, m2, snr,
                y_label_batch, y_m1_batch, y_m2_batch, y_snr_batch,
                weight_ce=weight_ce, weight_reg=weight_reg
            )
            
            loss = loss_dict['total']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Record
            train_losses['total'].append(loss.item())
            train_losses['ce'].append(loss_dict['ce'])
            train_losses['m1'].append(loss_dict['m1'])
            train_losses['m2'].append(loss_dict['m2'])
            train_losses['snr'].append(loss_dict['snr'])
        
        # Validation
        model.eval()
        val_losses = {'total': [], 'ce': [], 'm1': [], 'm2': [], 'snr': []}
        val_y_true = []
        val_y_prob = []
        
        with torch.no_grad():
            for X_batch, y_label_batch, y_m1_batch, y_m2_batch, y_snr_batch in val_loader:
                X_batch = X_batch.to(device)
                y_label_batch = y_label_batch.long().to(device)
                y_m1_batch = y_m1_batch.float().to(device)
                y_m2_batch = y_m2_batch.float().to(device)
                y_snr_batch = y_snr_batch.float().to(device)
                
                logits, m1, m2, snr = model(X_batch)
                
                loss_dict = multi_task_loss(
                    logits, m1, m2, snr,
                    y_label_batch, y_m1_batch, y_m2_batch, y_snr_batch,
                    weight_ce=weight_ce, weight_reg=weight_reg
                )
                
                val_losses['total'].append(loss_dict['total'].item())
                val_losses['ce'].append(loss_dict['ce'])
                val_losses['m1'].append(loss_dict['m1'])
                val_losses['m2'].append(loss_dict['m2'])
                val_losses['snr'].append(loss_dict['snr'])
                
                # Metrics
                val_y_true.append(y_label_batch.cpu().numpy())
                val_y_prob.append(
                    torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                )
        
        # Aggregate metrics
        val_y_true = np.concatenate(val_y_true)
        val_y_prob = np.concatenate(val_y_prob)
        
        val_auc = roc_auc_score(val_y_true, val_y_prob)
        val_f1 = f1_score(val_y_true, (val_y_prob > 0.5).astype(int))
        
        history['train_loss'].append(np.mean(train_losses['total']))
        history['train_ce'].append(np.mean(train_losses['ce']))
        history['train_m1'].append(np.mean(train_losses['m1']))
        history['train_m2'].append(np.mean(train_losses['m2']))
        history['train_snr'].append(np.mean(train_losses['snr']))
        history['val_loss'].append(np.mean(val_losses['total']))
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
    parser = argparse.ArgumentParser(description="Multi-task GW detection training")
    parser.add_argument('--num-samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-ce', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--weight-reg', type=float, default=0.5, help='Regression loss weight')
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
    X, y_label, y_m1, y_m2, y_snr = prepare_multitask_dataset(
        num_samples=args.num_samples,
        signal_duration=args.signal_duration,
        use_real_ligo_noise=args.use_real_ligo_noise
    )
    
    # Train/val/test split (64/16/20)
    n_train = int(0.64 * len(X))
    n_val = int(0.16 * len(X))
    
    X_train = torch.tensor(X[:n_train], dtype=torch.float32)
    y_train_label = torch.tensor(y_label[:n_train], dtype=torch.long)
    y_train_m1 = torch.tensor(y_m1[:n_train], dtype=torch.float32)
    y_train_m2 = torch.tensor(y_m2[:n_train], dtype=torch.float32)
    y_train_snr = torch.tensor(y_snr[:n_train], dtype=torch.float32)
    
    X_val = torch.tensor(X[n_train:n_train+n_val], dtype=torch.float32)
    y_val_label = torch.tensor(y_label[n_train:n_train+n_val], dtype=torch.long)
    y_val_m1 = torch.tensor(y_m1[n_train:n_train+n_val], dtype=torch.float32)
    y_val_m2 = torch.tensor(y_m2[n_train:n_train+n_val], dtype=torch.float32)
    y_val_snr = torch.tensor(y_snr[n_train:n_train+n_val], dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, y_train_label, y_train_m1, y_train_m2, y_train_snr)
    val_dataset = TensorDataset(X_val, y_val_label, y_val_m1, y_val_m2, y_val_snr)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    print("\n[Stage 2/4] Initializing model...")
    model = ParameterEstimationCNN().to(device)
    print(f"[Model] Total parameters: {model.count_parameters():,}")
    
    # Train
    print("\n[Stage 3/4] Training...")
    start_time = datetime.now()
    
    history = train_multitask(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        weight_ce=args.weight_ce,
        weight_reg=args.weight_reg,
        patience=args.patience,
        device=device
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[Time] Training completed in {elapsed:.1f}s")
    
    # Save
    print("\n[Stage 4/4] Saving model and history...")
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    model_path = checkpoint_dir / 'multitask_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"[Saved] Model checkpoint: {model_path}")
    
    history_path = checkpoint_dir / 'multitask_history.json'
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
