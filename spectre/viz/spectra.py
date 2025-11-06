"""Spectral visualizations."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from spectre.core.features import TensorFeatures


def plot_spectral_spectrum(
    tensor_features: TensorFeatures,
    array: np.ndarray,
    output_path: Path,
    top_k: int = 50
):
    """
    Plot spectral spectrum (singular values).
    
    Args:
        tensor_features: TensorFeatures instance
        array: Weight tensor
        output_path: Path to save figure
        top_k: Number of top singular values to plot
    """
    if array.ndim < 2:
        return
    
    # Flatten to 2D if needed
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    
    try:
        # Compute SVD
        _, s, _ = np.linalg.svd(array, full_matrices=False)
        
        # Plot top-k singular values
        k = min(top_k, len(s))
        s_top = s[:k]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, k + 1), s_top, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel("Singular Value Index")
        plt.ylabel("Singular Value")
        plt.title(f"Spectral Spectrum: {tensor_features.name[:50]}")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def plot_mp_fit(
    tensor_features: TensorFeatures,
    array: np.ndarray,
    output_path: Path
):
    """
    Plot Marchenko-Pastur distribution fit.
    
    Args:
        tensor_features: TensorFeatures instance
        array: Weight tensor
        output_path: Path to save figure
    """
    if array.ndim < 2:
        return
    
    # Flatten to 2D if needed
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    
    m, n = array.shape
    
    if min(m, n) < 10:
        return
    
    try:
        # Compute sample covariance
        array_centered = array - np.mean(array, axis=0, keepdims=True)
        
        if m < n:
            C = np.dot(array_centered, array_centered.T) / n
        else:
            C = np.dot(array_centered.T, array_centered) / m
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvalsh(C)
        eigenvals = np.sort(eigenvals)[::-1]
        eigenvals = eigenvals[eigenvals > 0]
        
        if len(eigenvals) < 5:
            return
        
        # MP parameters
        c = min(m, n) / max(m, n)
        sigma_sq = np.var(array_centered)
        lambda_plus = sigma_sq * (1 + np.sqrt(c)) ** 2
        lambda_minus = sigma_sq * (1 - np.sqrt(c)) ** 2
        
        # Plot histogram and MP PDF
        plt.figure(figsize=(10, 6))
        
        # Histogram of eigenvalues
        plt.hist(eigenvals, bins=50, density=True, alpha=0.7, label="Empirical")
        
        # MP PDF
        x = np.linspace(lambda_minus * 0.5, lambda_plus * 1.5, 1000)
        pdf = np.zeros_like(x)
        mask = (x >= lambda_minus) & (x <= lambda_plus)
        pdf[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / (2 * np.pi * x[mask])
        pdf = pdf / (np.sum(pdf) * (x[1] - x[0]) + 1e-10)
        
        plt.plot(x, pdf, 'r-', linewidth=2, label="Marchenko-Pastur")
        plt.axvline(lambda_plus, color='g', linestyle='--', label=f"λ+ = {lambda_plus:.4f}")
        plt.axvline(lambda_minus, color='g', linestyle='--', label=f"λ- = {lambda_minus:.4f}")
        
        plt.xlabel("Eigenvalue")
        plt.ylabel("Density")
        plt.title(f"MP Fit: {tensor_features.name[:50]}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

