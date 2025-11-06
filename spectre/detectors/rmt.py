"""D7: Random Matrix Theory (RMT) Deviations detector."""

from typing import Dict, Optional

import numpy as np
from scipy import stats

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class RmtDetector:
    """D7: Random Matrix Theory deviations detector using Marchenko-Pastur distribution."""
    
    def __init__(self, config: Config):
        """
        Initialize RMT detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.use_cuda = config.get("device.use_cuda", True) and TORCH_AVAILABLE
        self.device = self._get_device()
    
    def _get_device(self):
        """Get compute device (CUDA if available, else CPU)."""
        if self.use_cuda and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device_idx = self.config.get("device.cuda_device", 0)
                return torch.device(f"cuda:{device_idx}")
        return torch.device("cpu") if TORCH_AVAILABLE else None
    
    def _marchenko_pastur_pdf(self, x: np.ndarray, lambda_plus: float, lambda_minus: float) -> np.ndarray:
        """
        Compute Marchenko-Pastur probability density function.
        
        Args:
            x: Eigenvalue values
            lambda_plus: Upper edge of MP distribution
            lambda_minus: Lower edge of MP distribution
            
        Returns:
            PDF values
        """
        pdf = np.zeros_like(x)
        mask = (x >= lambda_minus) & (x <= lambda_plus)
        pdf[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / (2 * np.pi * x[mask])
        return pdf
    
    def extract(
        self, 
        array: np.ndarray, 
        name: str, 
        role: ParameterRole, 
        layer_idx: Optional[int]
    ) -> Dict[str, float]:
        """
        Extract RMT features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of RMT features
        """
        if array.ndim < 2:
            return {}
        
        # Flatten to 2D if needed
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        
        m, n = array.shape
        
        if min(m, n) < 10:
            return {}
        
        features = {}
        
        try:
            # Compute sample covariance matrix
            # Center the data
            array_centered = array - np.mean(array, axis=0, keepdims=True)
            
            # Sample covariance: C = (1/n) * X^T * X
            # For MP distribution, we need eigenvalues of C
            if m < n:
                # Use X * X^T instead (smaller matrix)
                C = np.dot(array_centered, array_centered.T) / n
                use_transpose = True
            else:
                C = np.dot(array_centered.T, array_centered) / m
                use_transpose = False
            
            # Compute eigenvalues (use GPU if available)
            if self.device and self.device.type == "cuda" and TORCH_AVAILABLE:
                try:
                    C_torch = torch.from_numpy(C).float().to(self.device)
                    eigenvals = torch.linalg.eigvalsh(C_torch).cpu().numpy()
                    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
                except Exception:
                    eigenvals = np.linalg.eigvalsh(C)
                    eigenvals = np.sort(eigenvals)[::-1]
            else:
                eigenvals = np.linalg.eigvalsh(C)
                eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
            
            # Remove negative eigenvalues (numerical errors)
            eigenvals = eigenvals[eigenvals > 0]
            
            if len(eigenvals) < 5:
                return {}
            
            # Marchenko-Pastur parameters
            # For matrix X (m x n), ratio c = min(m,n) / max(m,n)
            c = min(m, n) / max(m, n)
            sigma_sq = np.var(array_centered)
            
            # MP distribution edges
            lambda_plus = sigma_sq * (1 + np.sqrt(c)) ** 2
            lambda_minus = sigma_sq * (1 - np.sqrt(c)) ** 2
            
            features["rmt.lambda_plus"] = float(lambda_plus)
            features["rmt.lambda_minus"] = float(lambda_minus)
            features["rmt.c_ratio"] = float(c)
            features["rmt.sigma_sq"] = float(sigma_sq)
            
            # Count spikes above bulk edge
            spikes_above = np.sum(eigenvals > lambda_plus)
            features["rmt.spikes_above_bulk"] = float(spikes_above)
            features["rmt.spike_fraction"] = float(spikes_above / len(eigenvals))
            
            # Largest eigenvalue
            if len(eigenvals) > 0:
                features["rmt.largest_eigenval"] = float(eigenvals[0])
                
                # Deviation from bulk edge
                if lambda_plus > 0:
                    deviation = (eigenvals[0] - lambda_plus) / lambda_plus
                    features["rmt.largest_deviation"] = float(deviation)
            
            # Fit MP distribution to bulk eigenvalues
            bulk_eigenvals = eigenvals[eigenvals <= lambda_plus]
            if len(bulk_eigenvals) > 10:
                # Compute empirical CDF
                empirical_cdf = np.arange(1, len(bulk_eigenvals) + 1) / len(bulk_eigenvals)
                
                # Theoretical MP CDF (approximate)
                # Use Kolmogorov-Smirnov test
                try:
                    # Create histogram for comparison
                    hist, bins = np.histogram(bulk_eigenvals, bins=50, density=True)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    
                    # Theoretical PDF
                    mp_pdf = self._marchenko_pastur_pdf(bin_centers, lambda_plus, lambda_minus)
                    mp_pdf = mp_pdf / (np.sum(mp_pdf) + 1e-10)
                    
                    # Normalize hist
                    hist = hist / (np.sum(hist) + 1e-10)
                    
                    # KL divergence
                    kl_div = np.sum(hist * np.log((hist + 1e-10) / (mp_pdf + 1e-10)))
                    features["rmt.bulk_kl_divergence"] = float(kl_div)
                except Exception:
                    pass
            
            # Small sigma anomaly: check if variance is unusually small
            if sigma_sq > 0:
                # Compare to expected variance for random matrix
                expected_var = np.mean(eigenvals)
                if expected_var > 0:
                    var_ratio = sigma_sq / expected_var
                    features["rmt.variance_ratio"] = float(var_ratio)
            
        except Exception:
            pass
        
        return features

