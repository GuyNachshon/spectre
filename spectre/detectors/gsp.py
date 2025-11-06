"""D9: Graph-Signal Processing (Neuron Graph) detector."""

from typing import Dict, Optional

import numpy as np
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class GspDetector:
    """D9: Graph-signal processing detector for neuron graphs."""
    
    def __init__(self, config: Config):
        """
        Initialize GSP detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.embed_dim = config.get("gsp.embed_dim", 16)
    
    def extract(
        self, 
        array: np.ndarray, 
        name: str, 
        role: ParameterRole, 
        layer_idx: Optional[int]
    ) -> Dict[str, float]:
        """
        Extract graph-signal processing features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of GSP features
        """
        if array.ndim < 2:
            return {}
        
        features = {}
        
        # Ensure 2D
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        
        m, n = array.shape
        
        if min(m, n) < 5:
            return {}
        
        try:
            # Build graph from rows (neurons)
            # Use cosine similarity as edge weights
            if m >= 2:
                # Compute pairwise cosine similarity
                row_norms = np.linalg.norm(array, axis=1, keepdims=True)
                row_norms[row_norms == 0] = 1  # Avoid division by zero
                array_normalized = array / row_norms
                
                # Cosine similarity matrix
                similarity_matrix = np.dot(array_normalized, array_normalized.T)
                
                # Threshold to create sparse graph
                threshold = np.percentile(similarity_matrix.flatten(), 75)
                adjacency = (similarity_matrix > threshold).astype(float)
                adjacency = adjacency - np.eye(m)  # Remove self-loops
                
                # Graph Laplacian
                degree = np.sum(adjacency, axis=1)
                laplacian = np.diag(degree) - adjacency
                
                # Graph Fourier Transform: eigendecomposition of Laplacian
                eigenvals, eigenvecs = np.linalg.eigh(laplacian)
                
                # Sort eigenvalues
                idx = np.argsort(eigenvals)
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
                
                # Bandpass ratios: energy in different frequency bands
                if len(eigenvals) > 0:
                    # Low-pass: bottom 25%
                    low_idx = int(len(eigenvals) * 0.25)
                    low_energy = np.sum(eigenvals[:low_idx])
                    
                    # High-pass: top 25%
                    high_idx = int(len(eigenvals) * 0.75)
                    high_energy = np.sum(eigenvals[high_idx:])
                    
                    total_energy = np.sum(eigenvals)
                    if total_energy > 0:
                        features["gsp.lowpass_ratio"] = float(low_energy / total_energy)
                        features["gsp.highpass_ratio"] = float(high_energy / total_energy)
                    
                    # Graph connectivity
                    features["gsp.avg_degree"] = float(np.mean(degree))
                    features["gsp.max_degree"] = float(np.max(degree))
                    features["gsp.min_degree"] = float(np.min(degree))
                    
                    # Spectral gap
                    if len(eigenvals) > 1:
                        spectral_gap = eigenvals[1] - eigenvals[0]
                        features["gsp.spectral_gap"] = float(spectral_gap)
            
            # Build graph from columns
            if n >= 2:
                col_norms = np.linalg.norm(array, axis=0, keepdims=True)
                col_norms[col_norms == 0] = 1
                array_normalized = array / col_norms
                
                similarity_matrix = np.dot(array_normalized.T, array_normalized)
                threshold = np.percentile(similarity_matrix.flatten(), 75)
                adjacency = (similarity_matrix > threshold).astype(float)
                adjacency = adjacency - np.eye(n)
                
                degree = np.sum(adjacency, axis=1)
                
                features["gsp.col_avg_degree"] = float(np.mean(degree))
                features["gsp.col_max_degree"] = float(np.max(degree))
                features["gsp.col_min_degree"] = float(np.min(degree))
            
        except Exception:
            pass
        
        return features

