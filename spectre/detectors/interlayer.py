"""D2: Inter-Layer Correlation Drift detector."""

from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class InterlayerDetector:
    """D2: Inter-layer correlation drift detector."""
    
    def __init__(self, config: Config):
        """
        Initialize interlayer detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.spectral_hist_bins = config.get("interlayer.spectral_hist_bins", 64)
        self.use_cuda = config.get("device.use_cuda", True) and TORCH_AVAILABLE
        self.device = self._get_device()
        self._previous_tensors: Dict[ParameterRole, List[np.ndarray]] = {}
        self._previous_spectra: Dict[ParameterRole, List[np.ndarray]] = {}
    
    def _get_device(self):
        """Get compute device (CUDA if available, else CPU)."""
        if self.use_cuda and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device_idx = self.config.get("device.cuda_device", 0)
                return torch.device(f"cuda:{device_idx}")
        return torch.device("cpu") if TORCH_AVAILABLE else None
    
    def extract(
        self, 
        array: np.ndarray, 
        name: str, 
        role: ParameterRole, 
        layer_idx: Optional[int]
    ) -> Dict[str, float]:
        """
        Extract inter-layer correlation features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of inter-layer features
        """
        if array.ndim < 2:
            return {}
        
        # Flatten to 1D for correlation
        array_flat = array.flatten()
        
        features = {}
        
        # Cosine similarity with previous layers of same role
        if role in self._previous_tensors and len(self._previous_tensors[role]) > 0:
            prev_flat = self._previous_tensors[role][-1]
            
            # Normalize for cosine similarity
            norm_current = np.linalg.norm(array_flat)
            norm_prev = np.linalg.norm(prev_flat)
            
            if norm_current > 0 and norm_prev > 0:
                cosine_sim = np.dot(array_flat, prev_flat) / (norm_current * norm_prev)
                features["interlayer.cosine_sim"] = float(cosine_sim)
                
                # Correlation coefficient
                if len(array_flat) == len(prev_flat):
                    corr = np.corrcoef(array_flat, prev_flat)[0, 1]
                    if not np.isnan(corr):
                        features["interlayer.correlation"] = float(corr)
        
        # Spectral histogram JSD distance
        if array.ndim >= 2:
            # Compute SVD for spectral histogram
            if array.ndim > 2:
                array_2d = array.reshape(array.shape[0], -1)
            else:
                array_2d = array
            
            try:
                # Use GPU SVD if available
                if self.device and self.device.type == "cuda" and TORCH_AVAILABLE:
                    try:
                        array_torch = torch.from_numpy(array_2d).float().to(self.device)
                        _, s, _ = torch.linalg.svd(array_torch, full_matrices=False)
                        s = s.cpu().numpy()
                    except Exception:
                        _, s, _ = np.linalg.svd(array_2d, full_matrices=False)
                else:
                    _, s, _ = np.linalg.svd(array_2d, full_matrices=False)
                s_normalized = s / (np.sum(s) + 1e-10)
                
                # Create histogram
                hist, _ = np.histogram(
                    s_normalized, 
                    bins=self.spectral_hist_bins, 
                    range=(0, 1),
                    density=True
                )
                hist = hist / (np.sum(hist) + 1e-10)
                
                if role in self._previous_spectra and len(self._previous_spectra[role]) > 0:
                    prev_hist = self._previous_spectra[role][-1]
                    
                    # Jensen-Shannon divergence
                    jsd = jensenshannon(hist, prev_hist)
                    features["interlayer.spectral_jsd"] = float(jsd)
                    
                    # KL divergence (symmetric)
                    kl_forward = entropy(hist, prev_hist)
                    kl_backward = entropy(prev_hist, hist)
                    features["interlayer.spectral_kl"] = float((kl_forward + kl_backward) / 2)
                
                # Store for next iteration
                if role not in self._previous_spectra:
                    self._previous_spectra[role] = []
                self._previous_spectra[role].append(hist)
                
            except Exception:
                pass
        
        # Store current tensor for next iteration
        if role not in self._previous_tensors:
            self._previous_tensors[role] = []
        self._previous_tensors[role].append(array_flat)
        
        # Limit history to avoid memory issues
        if len(self._previous_tensors[role]) > 10:
            self._previous_tensors[role] = self._previous_tensors[role][-10:]
        if role in self._previous_spectra and len(self._previous_spectra[role]) > 10:
            self._previous_spectra[role] = self._previous_spectra[role][-10:]
        
        return features

