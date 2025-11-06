"""D10: Topological Data Analysis (TDA) detector."""

from typing import Dict, Optional

import numpy as np

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class TdaDetector:
    """D10: Topological Data Analysis detector using persistent homology."""
    
    def __init__(self, config: Config):
        """
        Initialize TDA detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        if not GUDHI_AVAILABLE:
            print("Warning: gudhi not available, TDA detector will be disabled")
    
    def extract(
        self, 
        array: np.ndarray, 
        name: str, 
        role: ParameterRole, 
        layer_idx: Optional[int]
    ) -> Dict[str, float]:
        """
        Extract TDA features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of TDA features
        """
        if not GUDHI_AVAILABLE:
            return {}
        
        if array.ndim < 2:
            return {}
        
        features = {}
        
        # Sample patches for TDA
        # For large arrays, sample patches
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        
        m, n = array.shape
        
        if min(m, n) < 10:
            return {}
        
        try:
            # Sample patches from the array
            patch_size = min(20, min(m, n) // 2)
            num_patches = min(50, (m - patch_size) * (n - patch_size))
            
            if num_patches < 10:
                return {}
            
            # Extract patches
            patches = []
            for _ in range(num_patches):
                i = np.random.randint(0, m - patch_size)
                j = np.random.randint(0, n - patch_size)
                patch = array[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())
            
            patches = np.array(patches)
            
            # Compute distance matrix
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(patches))
            
            # Create Rips complex
            rips_complex = gudhi.RipsComplex(distance_matrix=distances, max_edge_length=1.0)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            
            # Compute persistent homology
            persistence = simplex_tree.persistence()
            
            # Extract H0 and H1 features
            h0_lifetimes = []
            h1_lifetimes = []
            
            for dim, (birth, death) in persistence:
                if death == float('inf'):
                    death = birth + 1  # Handle infinite persistence
                
                lifetime = death - birth
                
                if dim == 0:
                    h0_lifetimes.append(lifetime)
                elif dim == 1:
                    h1_lifetimes.append(lifetime)
            
            # Barcode entropy
            if h0_lifetimes:
                h0_normalized = np.array(h0_lifetimes) / (np.sum(h0_lifetimes) + 1e-10)
                h0_normalized = h0_normalized[h0_normalized > 0]
                if len(h0_normalized) > 0:
                    from scipy.stats import entropy
                    h0_entropy = entropy(h0_normalized)
                    features["tda.h0_entropy"] = float(h0_entropy)
                    features["tda.h0_total_persistence"] = float(np.sum(h0_lifetimes))
                    features["tda.h0_longest_lifetime"] = float(np.max(h0_lifetimes)) if h0_lifetimes else 0.0
            
            if h1_lifetimes:
                h1_normalized = np.array(h1_lifetimes) / (np.sum(h1_lifetimes) + 1e-10)
                h1_normalized = h1_normalized[h1_normalized > 0]
                if len(h1_normalized) > 0:
                    from scipy.stats import entropy
                    h1_entropy = entropy(h1_normalized)
                    features["tda.h1_entropy"] = float(h1_entropy)
                    features["tda.h1_total_persistence"] = float(np.sum(h1_lifetimes))
                    features["tda.h1_longest_lifetime"] = float(np.max(h1_lifetimes)) if h1_lifetimes else 0.0
            
            # Number of features
            features["tda.h0_count"] = float(len(h0_lifetimes))
            features["tda.h1_count"] = float(len(h1_lifetimes))
            
        except Exception:
            pass
        
        return features

