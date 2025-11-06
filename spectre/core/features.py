"""Feature extraction and storage."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectre.io.name_mapper import ParameterRole


@dataclass
class TensorFeatures:
    """Unified feature storage structure per PRD spec."""
    
    name: str
    role: ParameterRole
    layer_idx: Optional[int]
    shape: Tuple[int, ...]
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "role": self.role.value,
            "layer_idx": self.layer_idx,
            "shape": list(self.shape),
            "features": self.features,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorFeatures":
        """Create from dictionary."""
        from spectre.io.name_mapper import ParameterRole
        
        return cls(
            name=data["name"],
            role=ParameterRole(data["role"]),
            layer_idx=data.get("layer_idx"),
            shape=tuple(data["shape"]),
            features=data.get("features", {}),
        )


class FeatureStore:
    """Store and manage features for all tensors."""
    
    def __init__(self):
        """Initialize feature store."""
        self.features: List[TensorFeatures] = []
        self._by_name: Dict[str, TensorFeatures] = {}
        self._by_role: Dict[ParameterRole, List[TensorFeatures]] = {}
        self._by_layer: Dict[int, List[TensorFeatures]] = {}
    
    def add(self, tensor_features: TensorFeatures):
        """
        Add tensor features to store.
        
        Args:
            tensor_features: TensorFeatures instance
        """
        self.features.append(tensor_features)
        self._by_name[tensor_features.name] = tensor_features
        
        # Index by role
        role = tensor_features.role
        if role not in self._by_role:
            self._by_role[role] = []
        self._by_role[role].append(tensor_features)
        
        # Index by layer
        if tensor_features.layer_idx is not None:
            layer_idx = tensor_features.layer_idx
            if layer_idx not in self._by_layer:
                self._by_layer[layer_idx] = []
            self._by_layer[layer_idx].append(tensor_features)
    
    def get(self, name: str) -> Optional[TensorFeatures]:
        """
        Get features by tensor name.
        
        Args:
            name: Tensor name
            
        Returns:
            TensorFeatures or None if not found
        """
        return self._by_name.get(name)
    
    def get_by_role(self, role: ParameterRole) -> List[TensorFeatures]:
        """
        Get all features for a given role.
        
        Args:
            role: ParameterRole
            
        Returns:
            List of TensorFeatures
        """
        return self._by_role.get(role, [])
    
    def get_by_layer(self, layer_idx: int) -> List[TensorFeatures]:
        """
        Get all features for a given layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            List of TensorFeatures
        """
        return self._by_layer.get(layer_idx, [])
    
    def get_feature_matrix(
        self, 
        feature_name: str, 
        role: Optional[ParameterRole] = None
    ) -> np.ndarray:
        """
        Get feature values as matrix for standardization.
        
        Args:
            feature_name: Name of feature (e.g., "spectral.top1")
            role: Optional role filter
            
        Returns:
            Array of feature values
        """
        features_list = self._by_role.get(role) if role else self.features
        
        values = []
        for tf in features_list:
            if feature_name in tf.features:
                values.append(tf.features[feature_name])
        
        return np.array(values) if values else np.array([])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "per_tensor": [tf.to_dict() for tf in self.features],
            "summary": {
                "total_tensors": len(self.features),
                "by_role": {role.value: len(tensors) for role, tensors in self._by_role.items()},
                "by_layer": {str(layer): len(tensors) for layer, tensors in self._by_layer.items()},
            },
        }
    
    def get_all_feature_names(self) -> List[str]:
        """
        Get all unique feature names across all tensors.
        
        Returns:
            List of feature names
        """
        feature_names = set()
        for tf in self.features:
            feature_names.update(tf.features.keys())
        return sorted(list(feature_names))

