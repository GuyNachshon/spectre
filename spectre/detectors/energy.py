"""D5: Energy-Norm Anomaly detector."""

from typing import Dict, Optional

import numpy as np

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class EnergyDetector:
    """D5: Energy-norm anomaly detector."""
    
    def __init__(self, config: Config):
        """
        Initialize energy detector.
        
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
        Extract energy-norm features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of energy features
        """
        features = {}
        
        # L1, L2 norms (work for any dimension)
        l1_norm = np.linalg.norm(array, ord=1)
        l2_norm = np.linalg.norm(array, ord=2)
        
        features["energy.l1"] = float(l1_norm)
        features["energy.l2"] = float(l2_norm)
        
        # Frobenius norm (only for 2D+ arrays, same as L2 for 1D)
        if array.ndim >= 2:
            frobenius_norm = np.linalg.norm(array, ord='fro')
            features["energy.frobenius"] = float(frobenius_norm)
        else:
            # For 1D arrays, Frobenius norm is the same as L2 norm
            frobenius_norm = l2_norm
            features["energy.frobenius"] = float(l2_norm)
        
        # Normalized norms (per element)
        num_elements = array.size
        if num_elements > 0:
            features["energy.l1_normalized"] = float(l1_norm / num_elements)
            features["energy.l2_normalized"] = float(l2_norm / num_elements)
            features["energy.frobenius_normalized"] = float(frobenius_norm / num_elements)
        
        # Row and column energy (for 2D matrices)
        if array.ndim == 2:
            # Row energy (L2 norm of each row)
            row_energies = np.linalg.norm(array, axis=1, ord=2)
            features["energy.row_mean"] = float(np.mean(row_energies))
            features["energy.row_std"] = float(np.std(row_energies))
            features["energy.row_max"] = float(np.max(row_energies))
            features["energy.row_min"] = float(np.min(row_energies))
            
            # Column energy (L2 norm of each column)
            col_energies = np.linalg.norm(array, axis=0, ord=2)
            features["energy.col_mean"] = float(np.mean(col_energies))
            features["energy.col_std"] = float(np.std(col_energies))
            features["energy.col_max"] = float(np.max(col_energies))
            features["energy.col_min"] = float(np.min(col_energies))
            
            # Energy concentration: max row/col energy / mean
            if features["energy.row_mean"] > 0:
                features["energy.row_concentration"] = float(
                    features["energy.row_max"] / features["energy.row_mean"]
                )
            if features["energy.col_mean"] > 0:
                features["energy.col_concentration"] = float(
                    features["energy.col_max"] / features["energy.col_mean"]
                )
        
        # Energy ratio: L2 / L1 (sparsity indicator)
        if l1_norm > 0:
            features["energy.l2_l1_ratio"] = float(l2_norm / l1_norm)
        
        return features

