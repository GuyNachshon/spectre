"""D11: Sequence Change-Point & Matrix-Profile detector."""

from typing import Dict, List, Optional

import numpy as np

try:
    import stumpy
    STUMPY_AVAILABLE = True
except ImportError:
    STUMPY_AVAILABLE = False

from scipy import stats
from scipy.signal import find_peaks

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class SequenceDetector:
    """D11: Sequence change-point and matrix-profile detector."""
    
    def __init__(self, config: Config):
        """
        Initialize sequence detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self._feature_sequences: Dict[str, List[float]] = {}
    
    def extract(
        self, 
        array: np.ndarray, 
        name: str, 
        role: ParameterRole, 
        layer_idx: Optional[int]
    ) -> Dict[str, float]:
        """
        Extract sequence features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of sequence features
        """
        if layer_idx is None:
            return {}
        
        features = {}
        
        # Extract basic features for sequence analysis
        array_flat = array.flatten()
        
        # Compute features that will be tracked across layers
        feature_values = {
            "entropy": self._compute_entropy(array_flat),
            "stable_rank": self._compute_stable_rank(array),
            "mean": np.mean(array_flat),
            "std": np.std(array_flat),
        }
        
        # Store in sequences
        for feature_name, value in feature_values.items():
            key = f"{role.value}.{feature_name}"
            if key not in self._feature_sequences:
                self._feature_sequences[key] = []
            self._feature_sequences[key].append(value)
        
        # Analyze sequences if we have enough data
        for key, sequence in self._feature_sequences.items():
            if len(sequence) >= 5:
                # Change-point detection using CUSUM
                change_points = self._detect_change_points(sequence)
                if change_points:
                    features[f"sequence.{key}.change_points"] = float(len(change_points))
                    features[f"sequence.{key}.last_change"] = float(change_points[-1] / len(sequence))
                
                # Matrix profile (if stumpy available)
                if STUMPY_AVAILABLE and len(sequence) >= 10:
                    try:
                        mp = stumpy.stump(sequence, m=min(5, len(sequence) // 2))
                        if len(mp) > 0:
                            # Discord (anomaly) score
                            discord_idx = np.argmax(mp[:, 0])
                            discord_score = mp[discord_idx, 0]
                            features[f"sequence.{key}.discord_score"] = float(discord_score)
                            features[f"sequence.{key}.discord_position"] = float(discord_idx / len(sequence))
                    except Exception:
                        pass
        
        return features
    
    def _compute_entropy(self, array: np.ndarray) -> float:
        """Compute entropy of array."""
        hist, _ = np.histogram(array, bins=64, density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        hist = hist[hist > 0]
        if len(hist) > 0:
            from scipy.stats import entropy
            return float(entropy(hist))
        return 0.0
    
    def _compute_stable_rank(self, array: np.ndarray) -> float:
        """Compute stable rank of array."""
        if array.ndim < 2:
            return 0.0
        try:
            if array.ndim > 2:
                array = array.reshape(array.shape[0], -1)
            _, s, _ = np.linalg.svd(array, full_matrices=False)
            if len(s) > 0 and s[0] > 0:
                return float(np.sum(s ** 2) / (s[0] ** 2))
        except Exception:
            pass
        return 0.0
    
    def _detect_change_points(self, sequence: List[float]) -> List[int]:
        """
        Detect change points in sequence using CUSUM.
        
        Args:
            sequence: Sequence of values
            
        Returns:
            List of change point indices
        """
        if len(sequence) < 5:
            return []
        
        sequence = np.array(sequence)
        
        # CUSUM test
        mean = np.mean(sequence)
        cumsum = np.cumsum(sequence - mean)
        
        # Find peaks in cumulative sum (change points)
        peaks, _ = find_peaks(np.abs(cumsum), height=np.std(cumsum))
        
        return peaks.tolist()

