"""D4: Robust Outlier Tests detector (MAD, Dixon's Q)."""

from typing import Dict, Optional

import numpy as np
from scipy import stats

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class RobustDetector:
    """D4: Robust outlier tests using median/MAD and Dixon's Q-test."""
    
    def __init__(self, config: Config):
        """
        Initialize robust detector.
        
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
        Extract robust outlier features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of robust outlier features
        """
        array_flat = array.flatten()
        
        if len(array_flat) < 3:
            return {}
        
        features = {}
        
        # Median and MAD (Median Absolute Deviation)
        median = np.median(array_flat)
        mad = np.median(np.abs(array_flat - median))
        
        features["robust.median"] = float(median)
        features["robust.mad"] = float(mad)
        
        # MAD-based z-scores
        if mad > 0:
            mad_z_scores = np.abs((array_flat - median) / (mad * 1.4826))  # 1.4826 makes MAD consistent with std for normal
            max_mad_z = np.max(mad_z_scores)
            features["robust.max_mad_z"] = float(max_mad_z)
            features["robust.mean_mad_z"] = float(np.mean(mad_z_scores))
            
            # Fraction of outliers (beyond 3 MAD)
            outlier_fraction = np.mean(mad_z_scores > 3.0)
            features["robust.outlier_fraction"] = float(outlier_fraction)
        
        # Dixon's Q-test for small vectors (3-30 elements)
        if 3 <= len(array_flat) <= 30:
            sorted_array = np.sort(array_flat)
            
            # Test for outlier at high end
            if len(sorted_array) > 1:
                range_val = sorted_array[-1] - sorted_array[0]
                if range_val > 0:
                    q_high = (sorted_array[-1] - sorted_array[-2]) / range_val
                    features["robust.dixon_q_high"] = float(q_high)
            
            # Test for outlier at low end
            if len(sorted_array) > 1:
                range_val = sorted_array[-1] - sorted_array[0]
                if range_val > 0:
                    q_low = (sorted_array[1] - sorted_array[0]) / range_val
                    features["robust.dixon_q_low"] = float(q_low)
        
        # IQR-based outlier detection
        q25, q75 = np.percentile(array_flat, [25, 75])
        iqr = q75 - q25
        
        features["robust.iqr"] = float(iqr)
        
        if iqr > 0:
            # Outliers beyond 1.5 * IQR
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_count = np.sum((array_flat < lower_bound) | (array_flat > upper_bound))
            features["robust.iqr_outlier_count"] = float(outlier_count)
            features["robust.iqr_outlier_fraction"] = float(outlier_count / len(array_flat))
        
        # Robust coefficient of variation (MAD/median)
        if median != 0:
            robust_cv = mad / abs(median)
            features["robust.robust_cv"] = float(robust_cv)
        
        return features

