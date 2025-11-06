"""D3: Weight Distribution Divergence detector."""

from typing import Dict, Optional

import numpy as np
from scipy import stats
from scipy.stats import entropy, kurtosis, skew

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class DistributionDetector:
    """D3: Weight distribution divergence detector."""
    
    def __init__(self, config: Config):
        """
        Initialize distribution detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
    
    def extract(
        self, 
        array: np.ndarray, 
        name: str, 
        role: ParameterRole, 
        layer_idx: Optional[int]
    ) -> Dict[str, float]:
        """
        Extract distribution features from tensor.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of distribution features
        """
        array_flat = array.flatten()
        
        if len(array_flat) < 10:
            return {}
        
        # Handle complex arrays by taking real part
        if np.iscomplexobj(array_flat):
            array_flat = np.real(array_flat)
        
        features = {}
        
        # Basic statistics
        mean = np.mean(array_flat)
        std = np.std(array_flat)
        features["distribution.mean"] = float(mean)
        features["distribution.std"] = float(std)
        
        # Skewness and kurtosis
        if std > 0:
            features["distribution.skew"] = float(skew(array_flat))
            features["distribution.kurtosis"] = float(kurtosis(array_flat))
        
        # Entropy (discretized)
        # array_flat is already real (handled at top of function)
        hist, _ = np.histogram(array_flat, bins=64, density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        hist = hist[hist > 0]
        if len(hist) > 0:
            features["distribution.entropy"] = float(entropy(hist))
        
        # KLD to Gaussian
        if std > 0:
            # Handle complex arrays (already handled at top of function)
            gaussian_samples = np.random.normal(mean, std, size=len(array_flat))
            hist_data, _ = np.histogram(array_flat, bins=64, density=True)
            hist_gauss, _ = np.histogram(gaussian_samples, bins=64, density=True)
            
            hist_data = hist_data / (np.sum(hist_data) + 1e-10)
            hist_gauss = hist_gauss / (np.sum(hist_gauss) + 1e-10)
            
            kld_gauss = entropy(hist_data, hist_gauss)
            features["distribution.kld_gaussian"] = float(kld_gauss)
        
        # KLD to Laplace
        if std > 0:
            # Handle complex arrays (already handled at top of function)
            laplace_samples = np.random.laplace(mean, std / np.sqrt(2), size=len(array_flat))
            hist_data, _ = np.histogram(array_flat, bins=64, density=True)
            hist_laplace, _ = np.histogram(laplace_samples, bins=64, density=True)
            
            hist_data = hist_data / (np.sum(hist_data) + 1e-10)
            hist_laplace = hist_laplace / (np.sum(hist_laplace) + 1e-10)
            
            kld_laplace = entropy(hist_data, hist_laplace)
            features["distribution.kld_laplace"] = float(kld_laplace)
        
        # Chebyshev tail fraction: fraction of values beyond k*std
        if std > 0:
            for k in [2, 3, 4]:
                threshold = k * std
                tail_fraction = np.mean(np.abs(array_flat - mean) > threshold)
                features[f"distribution.tail_fraction_{k}sigma"] = float(tail_fraction)
        
        # Normality tests (p-values)
        if len(array_flat) > 3:
            try:
                _, p_value = stats.normaltest(array_flat)
                features["distribution.normality_pvalue"] = float(p_value)
            except Exception:
                pass
        
        return features

