"""D6: Graph Structural Outlier detector (Layer Graph)."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class LayerGraphDetector:
    """D6: Graph structural outlier detector using layer similarity graph."""
    
    def __init__(self, config: Config):
        """
        Initialize layer graph detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self._layer_embeddings: Dict[Tuple[int, ParameterRole], np.ndarray] = {}
        self._layer_features: List[Tuple[int, ParameterRole, np.ndarray]] = []
    
    def extract(
        self, 
        array: np.ndarray, 
        name: str, 
        role: ParameterRole, 
        layer_idx: Optional[int]
    ) -> Dict[str, float]:
        """
        Extract layer graph features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of layer graph features
        """
        if layer_idx is None:
            return {}
        
        features = {}
        
        # Create embedding for this layer-role pair
        # Use flattened array with summary statistics
        array_flat = array.flatten()
        
        # Create feature vector: statistics + spectral features
        embedding = []
        
        # Basic statistics
        embedding.extend([
            np.mean(array_flat),
            np.std(array_flat),
            np.median(array_flat),
            np.percentile(array_flat, 25),
            np.percentile(array_flat, 75),
        ])
        
        # Spectral features (top singular values)
        if array.ndim >= 2:
            try:
                if array.ndim > 2:
                    array_2d = array.reshape(array.shape[0], -1)
                else:
                    array_2d = array
                
                _, s, _ = np.linalg.svd(array_2d, full_matrices=False)
                # Take top 5 singular values
                for i in range(min(5, len(s))):
                    embedding.append(s[i])
                # Pad if needed
                while len(embedding) < 10:
                    embedding.append(0.0)
            except Exception:
                # Pad if SVD fails
                while len(embedding) < 10:
                    embedding.append(0.0)
        else:
            # Pad for 1D tensors
            while len(embedding) < 10:
                embedding.append(0.0)
        
        embedding = np.array(embedding)
        
        # Store for graph construction
        key = (layer_idx, role)
        self._layer_embeddings[key] = embedding
        self._layer_features.append((layer_idx, role, embedding))
        
        # Compute LOF score if we have enough neighbors
        if len(self._layer_features) >= 5:
            try:
                # Get embeddings for same role
                same_role_embeddings = [
                    emb for l, r, emb in self._layer_features
                    if r == role
                ]
                
                if len(same_role_embeddings) >= 5:
                    same_role_embeddings = np.array(same_role_embeddings)
                    
                    # Local Outlier Factor
                    n_neighbors = min(5, len(same_role_embeddings) - 1)
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
                    lof_scores = lof.fit_predict(same_role_embeddings)
                    lof_outlier_scores = lof.negative_outlier_factor_
                    
                    # Find index of current layer
                    current_idx = len(same_role_embeddings) - 1
                    features["layer_graph.lof_score"] = float(-lof_outlier_scores[current_idx])
                    features["layer_graph.is_outlier"] = float(lof_scores[current_idx] == -1)
                    
                    # kNN distance
                    nn = NearestNeighbors(n_neighbors=min(3, len(same_role_embeddings) - 1))
                    nn.fit(same_role_embeddings[:-1])
                    distances, _ = nn.kneighbors([embedding])
                    features["layer_graph.knn_mean_distance"] = float(np.mean(distances))
                    features["layer_graph.knn_min_distance"] = float(np.min(distances))
            except Exception:
                pass
        
        return features
    
    def finalize(self) -> Dict[str, float]:
        """
        Finalize graph analysis after all layers processed.
        
        Returns:
            Dictionary of global graph features
        """
        # This can be called after all tensors are processed
        # to compute global graph metrics
        return {}

