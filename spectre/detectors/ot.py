"""D12: Optimal Transport Drift detector."""

from typing import Dict, Optional

import numpy as np

try:
    from ot import wasserstein_1d, wasserstein_2d
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class OtDetector:
    """D12: Optimal Transport drift detector using Wasserstein distances."""
    
    def __init__(self, config: Config):
        """
        Initialize OT detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self._previous_distributions: Dict[ParameterRole, np.ndarray] = {}
    
    def extract(
        self, 
        array: np.ndarray, 
        name: str, 
        role: ParameterRole, 
        layer_idx: Optional[int]
    ) -> Dict[str, float]:
        """
        Extract optimal transport features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of OT features
        """
        if not POT_AVAILABLE:
            return {}
        
        features = {}
        
        array_flat = array.flatten()
        
        if len(array_flat) < 10:
            return {}
        
        # Compare with previous layer of same role
        if role in self._previous_distributions:
            prev_dist = self._previous_distributions[role]
            
            try:
                # 1D Wasserstein distance
                # Sample for efficiency
                n_samples = min(1000, len(array_flat), len(prev_dist))
                current_samples = np.random.choice(array_flat, size=n_samples, replace=False)
                prev_samples = np.random.choice(prev_dist, size=n_samples, replace=False)
                
                # Compute 1D Wasserstein distance
                wasserstein_1d_dist = wasserstein_1d(current_samples, prev_samples)
                features["ot.wasserstein_1d"] = float(wasserstein_1d_dist)
                
                # Sliced Wasserstein distance (for 2D+ arrays)
                if array.ndim >= 2:
                    # Flatten to 2D if needed
                    if array.ndim > 2:
                        array_2d = array.reshape(array.shape[0], -1)
                    else:
                        array_2d = array
                    
                    # Sample slices
                    n_slices = min(10, array_2d.shape[0])
                    slice_indices = np.random.choice(array_2d.shape[0], size=n_slices, replace=False)
                    
                    sliced_distances = []
                    for idx in slice_indices:
                        current_slice = array_2d[idx, :]
                        # Compare with previous (if available)
                        if len(prev_dist) >= len(current_slice):
                            prev_slice = prev_dist[:len(current_slice)]
                            try:
                                w_dist = wasserstein_1d(current_slice, prev_slice)
                                sliced_distances.append(w_dist)
                            except Exception:
                                pass
                    
                    if sliced_distances:
                        features["ot.sliced_wasserstein_mean"] = float(np.mean(sliced_distances))
                        features["ot.sliced_wasserstein_std"] = float(np.std(sliced_distances))
                
            except Exception:
                pass
        
        # Store current distribution for next iteration
        self._previous_distributions[role] = array_flat
        
        # Limit history
        if len(self._previous_distributions) > 20:
            # Keep only most recent
            roles_to_keep = list(self._previous_distributions.keys())[-10:]
            self._previous_distributions = {r: self._previous_distributions[r] for r in roles_to_keep}
        
        return features

