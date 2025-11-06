"""D14: Multi-View Adversarial Subspace Search detector."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class MultiviewDetector:
    """D14: Multi-view adversarial subspace search detector."""
    
    def __init__(self, config: Config):
        """
        Initialize multiview detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self._feature_views: Dict[ParameterRole, List[np.ndarray]] = {}
        self._subspaces: Dict[ParameterRole, np.ndarray] = {}
    
    def extract(
        self, 
        array: np.ndarray, 
        name: str, 
        role: ParameterRole, 
        layer_idx: Optional[int]
    ) -> Dict[str, float]:
        """
        Extract multiview features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of multiview features
        """
        if array.ndim < 2:
            return {}
        
        features = {}
        
        # Create feature vector from multiple views
        array_flat = array.flatten()
        
        # View 1: Statistical features
        view1 = np.array([
            np.mean(array_flat),
            np.std(array_flat),
            np.median(array_flat),
            np.percentile(array_flat, 25),
            np.percentile(array_flat, 75),
            np.min(array_flat),
            np.max(array_flat),
        ])
        
        # View 2: Spectral features
        view2 = np.array([])
        if array.ndim >= 2:
            try:
                if array.ndim > 2:
                    array_2d = array.reshape(array.shape[0], -1)
                else:
                    array_2d = array
                
                _, s, _ = np.linalg.svd(array_2d, full_matrices=False)
                # Take top 5 singular values
                view2 = s[:5]
                # Pad if needed
                if len(view2) < 5:
                    view2 = np.pad(view2, (0, 5 - len(view2)), mode='constant')
            except Exception:
                view2 = np.zeros(5)
        else:
            view2 = np.zeros(5)
        
        # View 3: Distribution features
        hist, _ = np.histogram(array_flat, bins=10, density=True)
        view3 = hist / (np.sum(hist) + 1e-10)
        
        # Combine views
        feature_vector = np.concatenate([view1, view2, view3])
        
        # Store for subspace learning
        if role not in self._feature_views:
            self._feature_views[role] = []
        self._feature_views[role].append(feature_vector)
        
        # Learn subspace if we have enough samples
        if len(self._feature_views[role]) >= 10:
            try:
                # Stack all feature vectors
                X = np.array(self._feature_views[role])
                
                # Standardize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Learn PCA subspace
                n_components = min(5, X_scaled.shape[1] - 1)
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                
                # Project current sample
                current_scaled = scaler.transform([feature_vector])
                current_pca = pca.transform(current_scaled)
                
                # Anomaly score: distance from subspace
                # Use reconstruction error
                reconstructed = pca.inverse_transform(current_pca)
                reconstruction_error = np.linalg.norm(current_scaled - reconstructed)
                features["multiview.reconstruction_error"] = float(reconstruction_error)
                
                # Distance to centroid in PCA space
                centroid = np.mean(X_pca, axis=0)
                distance_to_centroid = np.linalg.norm(current_pca[0] - centroid)
                features["multiview.distance_to_centroid"] = float(distance_to_centroid)
                
                # Explained variance ratio
                features["multiview.explained_variance_ratio"] = float(np.sum(pca.explained_variance_ratio_))
                
                # Store subspace for future use
                self._subspaces[role] = {
                    "pca": pca,
                    "scaler": scaler,
                    "centroid": centroid,
                }
                
            except Exception:
                pass
        
        # Limit history
        if role in self._feature_views and len(self._feature_views[role]) > 50:
            self._feature_views[role] = self._feature_views[role][-50:]
        
        return features
    
    def finalize(self) -> Dict[str, float]:
        """
        Finalize multiview analysis after all layers processed.
        
        Returns:
            Dictionary of global multiview features
        """
        # This can be called after all tensors are processed
        # to compute global multiview metrics
        return {}

