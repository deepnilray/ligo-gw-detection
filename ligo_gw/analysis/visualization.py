"""Visualization utilities.

Plotting for paper figures, demos, and debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_scalogram(
    scalogram: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    title: str = "Scalogram",
    vmin: float = -100,
    vmax: float = 0,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot wavelet scalogram.
    
    Args:
        scalogram: Time-frequency power (num_freqs, num_times)
        freqs: Frequency axis (Hz)
        times: Time axis (seconds)
        title: Plot title
        vmin, vmax: Color scale limits (dB)
        ax: Matplotlib axes
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.pcolormesh(times, freqs, scalogram, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_yscale("log")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Power (dB)")
    
    return ax


def plot_detection(
    scalogram: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    detection_confidence: float,
    parameters: Optional[dict] = None,
    title: str = "Detection Result"
) -> plt.Figure:
    """Plot detection with overlaid information.
    
    Args:
        scalogram: Time-frequency scalogram
        freqs: Frequency axis
        times: Time axis
        detection_confidence: Detection probability (0-1)
        parameters: Optional parameter estimates {m1, m2, snr}
        title: Plot title
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot scalogram
    plot_scalogram(scalogram, freqs, times, ax=ax)
    
    # Add detection info
    status = "SIGNAL DETECTED" if detection_confidence > 0.5 else "NOISE"
    color = "red" if status == "SIGNAL DETECTED" else "gray"
    ax.text(
        0.5, 0.95, f"{status} (confidence: {detection_confidence:.2f})",
        transform=ax.transAxes, fontsize=14, color=color,
        ha="center", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )
    
    # Add parameter info if available
    if parameters:
        param_text = f"m1={parameters.get('m1', 0):.1f}M⊙, m2={parameters.get('m2', 0):.1f}M⊙"
        ax.text(
            0.5, 0.05, param_text, transform=ax.transAxes,
            fontsize=12, ha="center", va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5)
        )
    
    fig.suptitle(title)
    return fig
