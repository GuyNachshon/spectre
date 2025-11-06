"""Scoring system: standardization, ensemble scoring, thresholding."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from spectre.core.features import FeatureStore, TensorFeatures
from spectre.io.name_mapper import ParameterRole


# Default detector weights from PRD
DEFAULT_WEIGHTS = {
    "spectral": 0.25,
    "interlayer": 0.20,
    "distribution": 0.15,
    "robust": 0.08,
    "energy": 0.07,
    "layer_graph": 0.05,
    "rmt": 0.10,
    "spectrogram": 0.05,
    "gsp": 0.03,
    "tda": 0.03,
    "sequence_cp": 0.03,
    "ot": 0.04,
    "multiview": 0.02,
}

# Thresholds from PRD
THRESHOLDS = {
    "GREEN": 2.0,
    "AMBER": 3.0,
    "RED": 3.0,
    "HARD_RED": 4.5,
}


class Scorer:
    """Scoring system for feature standardization and ensemble scoring."""
    
    def __init__(self, feature_store: FeatureStore, weights: Optional[Dict[str, float]] = None):
        """
        Initialize scorer.
        
        Args:
            feature_store: FeatureStore instance
            weights: Optional detector weights (defaults to PRD weights)
        """
        self.feature_store = feature_store
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.detector_scores: Dict[str, Dict[str, float]] = {}
        self.ensemble_scores: Dict[str, float] = {}
        self.z_scores: Dict[str, Dict[str, float]] = {}
    
    def standardize_features(self, use_robust: bool = True):
        """
        Standardize features using role-based cohort standardization.
        
        Args:
            use_robust: Use robust z-scores (median/MAD) instead of mean/std
        """
        all_feature_names = self.feature_store.get_all_feature_names()
        
        # Group features by detector
        detector_features: Dict[str, List[str]] = {}
        for feature_name in all_feature_names:
            detector = feature_name.split(".")[0]
            if detector not in detector_features:
                detector_features[detector] = []
            detector_features[detector].append(feature_name)
        
        # Standardize per role
        for role in ParameterRole:
            role_features = self.feature_store.get_by_role(role)
            
            if len(role_features) < 2:
                continue
            
            # Standardize each feature within this role
            for feature_name in all_feature_names:
                feature_values = self.feature_store.get_feature_matrix(feature_name, role)
                
                if len(feature_values) < 2:
                    continue
                
                # Compute z-scores
                if use_robust:
                    # Robust z-score using median and MAD
                    median = np.median(feature_values)
                    mad = np.median(np.abs(feature_values - median))
                    if mad > 0:
                        z_scores = (feature_values - median) / (mad * 1.4826)
                    else:
                        z_scores = np.zeros_like(feature_values)
                else:
                    # Standard z-score
                    mean = np.mean(feature_values)
                    std = np.std(feature_values)
                    if std > 0:
                        z_scores = (feature_values - mean) / std
                    else:
                        z_scores = np.zeros_like(feature_values)
                
                # Store z-scores
                for i, tf in enumerate(role_features):
                    if feature_name in tf.features:
                        if tf.name not in self.z_scores:
                            self.z_scores[tf.name] = {}
                        self.z_scores[tf.name][feature_name] = float(z_scores[i])
    
    def compute_detector_scores(self, max_z: float = 5.0):
        """
        Compute per-detector z-scores.
        
        Args:
            max_z: Maximum z-score to clip to
        """
        # Group features by detector
        detector_features: Dict[str, List[str]] = {}
        for feature_name in self.feature_store.get_all_feature_names():
            detector = feature_name.split(".")[0]
            if detector not in detector_features:
                detector_features[detector] = []
            detector_features[detector].append(feature_name)
        
        # Compute detector scores for each tensor
        for tf in self.feature_store.features:
            tensor_name = tf.name
            
            if tensor_name not in self.detector_scores:
                self.detector_scores[tensor_name] = {}
            
            # Compute score for each detector
            for detector, feature_names in detector_features.items():
                # Get z-scores for this detector's features
                z_scores = []
                if tensor_name in self.z_scores:
                    for feature_name in feature_names:
                        if feature_name in self.z_scores[tensor_name]:
                            z_scores.append(self.z_scores[tensor_name][feature_name])
                
                if z_scores:
                    # Use max absolute z-score as detector score
                    detector_z = np.max(np.abs(z_scores))
                    detector_z = np.clip(detector_z, -max_z, max_z)
                    self.detector_scores[tensor_name][detector] = float(detector_z)
                else:
                    self.detector_scores[tensor_name][detector] = 0.0
    
    def compute_ensemble_score(self, max_z: float = 5.0):
        """
        Compute ensemble score as weighted sum of detector scores.
        
        Args:
            max_z: Maximum z-score to clip to
        """
        for tensor_name, detector_scores in self.detector_scores.items():
            ensemble = 0.0
            total_weight = 0.0
            
            for detector, z_score in detector_scores.items():
                weight = self.weights.get(detector, 0.0)
                ensemble += weight * np.clip(z_score, -max_z, max_z)
                total_weight += weight
            
            if total_weight > 0:
                ensemble = ensemble / total_weight
            
            self.ensemble_scores[tensor_name] = float(ensemble)
    
    def get_flag(self, tensor_name: str) -> str:
        """
        Get risk flag for tensor.
        
        Args:
            tensor_name: Tensor name
            
        Returns:
            Risk flag: GREEN, AMBER, RED, or HARD_RED
        """
        ensemble = self.ensemble_scores.get(tensor_name, 0.0)
        
        # Check for HARD_RED (any detector >= 4.5)
        if tensor_name in self.detector_scores:
            max_detector = max(
                abs(z) for z in self.detector_scores[tensor_name].values()
            )
            if max_detector >= THRESHOLDS["HARD_RED"]:
                return "HARD_RED"
        
        # Check ensemble thresholds
        if ensemble >= THRESHOLDS["RED"]:
            return "RED"
        elif ensemble >= THRESHOLDS["AMBER"]:
            return "AMBER"
        else:
            return "GREEN"
    
    def get_suspects(self, top_k: int = 10) -> List[Dict]:
        """
        Get top-k suspect tensors.
        
        Args:
            top_k: Number of suspects to return
            
        Returns:
            List of suspect dictionaries
        """
        suspects = []
        
        for tensor_name, ensemble in self.ensemble_scores.items():
            flag = self.get_flag(tensor_name)
            tf = self.feature_store.get(tensor_name)
            
            if tf:
                suspects.append({
                    "name": tensor_name,
                    "role": tf.role.value,
                    "layer_idx": tf.layer_idx,
                    "ensemble_score": ensemble,
                    "flag": flag,
                    "detector_scores": self.detector_scores.get(tensor_name, {}),
                })
        
        # Sort by ensemble score (descending)
        suspects.sort(key=lambda x: x["ensemble_score"], reverse=True)
        
        return suspects[:top_k]
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics.
        
        Returns:
            Summary dictionary
        """
        if not self.ensemble_scores:
            return {
                "ensemble_sigma": 0.0,
                "flag": "GREEN",
                "suspects": [],
            }
        
        ensemble_values = list(self.ensemble_scores.values())
        ensemble_sigma = np.mean(ensemble_values)
        
        # Overall flag: worst flag across all tensors
        flags = [self.get_flag(name) for name in self.ensemble_scores.keys()]
        if "HARD_RED" in flags:
            overall_flag = "HARD_RED"
        elif "RED" in flags:
            overall_flag = "RED"
        elif "AMBER" in flags:
            overall_flag = "AMBER"
        else:
            overall_flag = "GREEN"
        
        return {
            "ensemble_sigma": float(ensemble_sigma),
            "flag": overall_flag,
            "suspects": self.get_suspects(top_k=10),
            "total_tensors": len(self.ensemble_scores),
            "flag_counts": {
                "GREEN": sum(1 for f in flags if f == "GREEN"),
                "AMBER": sum(1 for f in flags if f == "AMBER"),
                "RED": sum(1 for f in flags if f == "RED"),
                "HARD_RED": sum(1 for f in flags if f == "HARD_RED"),
            },
        }

