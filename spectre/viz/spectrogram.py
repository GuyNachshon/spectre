"""Spectrogram visualizations."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from spectre.core.features import TensorFeatures


def plot_weight_spectrogram(
    tensor_features: TensorFeatures,
    array: np.ndarray,
    output_path: Path,
    window: tuple = (16, 16)
):
    """
    Plot weight spectrogram.
    
    Args:
        tensor_features: TensorFeatures instance
        array: Weight tensor
        output_path: Path to save figure
        window: Window size for STFT
    """
    if array.ndim < 2:
        return
    
    # Ensure 2D
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    
    m, n = array.shape
    
    if m < window[0] or n < window[1]:
        return
    
    try:
        # Compute 2D FFT as spectrogram
        fft_2d = np.fft.fft2(array)
        spectrogram = np.abs(fft_2d)
        
        # Shift zero frequency to center
        spectrogram = np.fft.fftshift(spectrogram)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(
            np.log(spectrogram + 1e-10),
            aspect="auto",
            cmap="viridis",
            origin="lower"
        )
        plt.colorbar(label="Log Magnitude")
        plt.xlabel("Frequency (Column)")
        plt.ylabel("Frequency (Row)")
        plt.title(f"Weight Spectrogram: {tensor_features.name[:50]}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

