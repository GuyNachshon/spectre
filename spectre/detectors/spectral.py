"""D1: Spectral Signatures detector using SVD."""

from typing import Dict, Optional

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class SpectralDetector:
    """D1: Spectral signatures detector using truncated/randomized SVD."""
    
    def __init__(self, config: Config):
        """
        Initialize spectral detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.rank = config.get("svd.rank", 96)
        self.power_iters = config.get("svd.power_iters", 2)
        self.use_cuda = config.get("svd.use_cuda", True) and TORCH_AVAILABLE
        self.device = self._get_device()
    
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
        Extract spectral features using SVD.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of spectral features
        """
        # Skip very large tensors (>10M elements) to avoid extremely slow SVD
        if array.size > 10_000_000:
            return {}
        
        if array.ndim < 2:
            # Skip 1D tensors (biases, layer norms)
            return {}
        
        # Flatten to 2D if needed
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        
        m, n = array.shape
        k = min(self.rank, min(m, n) - 1)
        
        if k < 1:
            return {}
        
        features = {}
        
        try:
            # Try GPU-accelerated SVD if available
            if self.device and self.device.type == "cuda" and TORCH_AVAILABLE:
                try:
                    # Convert to torch tensor and move to GPU
                    array_torch = torch.from_numpy(array).float().to(self.device)
                    
                    # Use torch SVD (faster on GPU for large matrices)
                    U, s, Vt = torch.linalg.svd(array_torch, full_matrices=False)
                    
                    # Take top-k
                    s = s[:k].cpu().numpy()
                    U = U[:, :k].cpu().numpy()
                    Vt = Vt[:k, :].cpu().numpy()
                except Exception:
                    # Fallback to CPU
                    U, s, Vt = randomized_svd(
                        array, 
                        n_components=k, 
                        n_iter=self.power_iters,
                        random_state=42
                    )
            else:
                # Use randomized SVD for efficiency (CPU)
                U, s, Vt = randomized_svd(
                    array, 
                    n_components=k, 
                    n_iter=self.power_iters,
                    random_state=42
                )
            
            # Top-k singular values
            for i in range(min(10, len(s))):
                features[f"spectral.top{i+1}"] = float(s[i])
            
            # Stable rank: sum(s^2) / max(s)^2
            if len(s) > 0 and s[0] > 0:
                stable_rank = np.sum(s ** 2) / (s[0] ** 2)
                features["spectral.stable_rank"] = float(stable_rank)
            
            # Spectral decay: ratio of top-1 to top-k
            if len(s) > 1 and s[0] > 0:
                spectral_decay = s[-1] / s[0]
                features["spectral.decay"] = float(spectral_decay)
            
            # Tail mass: sum of tail singular values / sum of all
            if len(s) > 1:
                tail_size = max(1, len(s) // 4)
                tail_mass = np.sum(s[-tail_size:] ** 2) / np.sum(s ** 2)
                features["spectral.tail_mass"] = float(tail_mass)
            
            # Effective rank (Shannon entropy of normalized singular values)
            if len(s) > 0:
                s_normalized = s ** 2 / np.sum(s ** 2)
                s_normalized = s_normalized[s_normalized > 0]
                if len(s_normalized) > 0:
                    effective_rank = -np.sum(s_normalized * np.log2(s_normalized))
                    features["spectral.effective_rank"] = float(effective_rank)
            
        except Exception:
            # Fallback to full SVD for small matrices
            try:
                U, s, Vt = np.linalg.svd(array, full_matrices=False)
                k = min(self.rank, len(s))
                s = s[:k]
                
                if len(s) > 0:
                    features["spectral.top1"] = float(s[0])
                    if len(s) > 1:
                        features["spectral.top2"] = float(s[1])
                    if s[0] > 0:
                        stable_rank = np.sum(s ** 2) / (s[0] ** 2)
                        features["spectral.stable_rank"] = float(stable_rank)
            except Exception:
                pass
        
        return features

