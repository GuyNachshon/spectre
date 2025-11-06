"""Main orchestration pipeline."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from spectre.core.config import Config
from spectre.core.conformal import ConformalCalibrator
from spectre.core.features import FeatureStore, TensorFeatures
from spectre.core.scoring import Scorer
from spectre.io.loader import get_checkpoint_info, load_checkpoint
from spectre.io.name_mapper import NameMapper, ParameterRole

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Pipeline:
    """Main pipeline for model scanning."""
    
    def __init__(self, config: Config):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.feature_store = FeatureStore()
        self.name_mapper = NameMapper(ruleset=self.config.get("model.ruleset", "gpt_like"))
        self.detectors = self._initialize_detectors()
        self.model_info: Optional[Dict] = None
        self.scorer: Optional[Scorer] = None
        self.calibrator: Optional[ConformalCalibrator] = None
        self.device = self._get_device()
    
    def _get_device(self):
        """Get compute device (CUDA if available, else CPU)."""
        use_cuda = self.config.get("device.use_cuda", True)
        if use_cuda and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device_idx = self.config.get("device.cuda_device", 0)
                device = torch.device(f"cuda:{device_idx}")
                print(f"Using CUDA device: {device}")
                return device
        if TORCH_AVAILABLE:
            print("Using CPU (CUDA not available or disabled)")
        return None
    
    def _initialize_detectors(self) -> Dict:
        """Initialize enabled detectors."""
        detectors = {}
        
        # Import detectors lazily to avoid circular imports
        # Map config name -> (module_path, class_name)
        detector_modules = {
            "svd": ("spectre.detectors.spectral", "SpectralDetector"),
            "spectral": ("spectre.detectors.spectral", "SpectralDetector"),
            "interlayer": ("spectre.detectors.interlayer", "InterlayerDetector"),
            "distribution": ("spectre.detectors.distribution", "DistributionDetector"),
            "robust": ("spectre.detectors.robust", "RobustDetector"),
            "energy": ("spectre.detectors.energy", "EnergyDetector"),
            "layer_graph": ("spectre.detectors.layer_graph", "LayerGraphDetector"),
            "rmt": ("spectre.detectors.rmt", "RmtDetector"),
            "spectrogram": ("spectre.detectors.spectrogram", "SpectrogramDetector"),
            "gsp": ("spectre.detectors.gsp", "GspDetector"),
            "tda": ("spectre.detectors.tda", "TdaDetector"),
            "sequence_cp": ("spectre.detectors.sequence", "SequenceDetector"),
            "ot": ("spectre.detectors.ot", "OtDetector"),
            "multiview": ("spectre.detectors.multiview", "MultiviewDetector"),
        }
        
        for name, (module_path, class_name) in detector_modules.items():
            # Check both the config name and aliases
            if self.config.is_enabled(name) or (name == "svd" and self.config.is_enabled("spectral")):
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    detector_class = getattr(module, class_name, None)
                    if detector_class:
                        # Use "spectral" as the key for both "svd" and "spectral"
                        key = "spectral" if name == "svd" else name
                        detectors[key] = detector_class(self.config)
                        print(f"✓ Loaded detector: {key} ({class_name})")
                except (ImportError, AttributeError) as e:
                    # Log error for debugging
                    print(f"Warning: Failed to load detector {name} ({class_name}): {e}")
                    pass
        
        if not detectors:
            print("Warning: No detectors were loaded! Check your configuration.")
        
        return detectors
    
    def run(self, checkpoint_path: Optional[Path] = None) -> Dict:
        """
        Run full pipeline: load → map → detect → score → output.
        
        Args:
            checkpoint_path: Optional path override
            
        Returns:
            Dictionary with results
        """
        # Load checkpoint
        checkpoint_path = checkpoint_path or Path(self.config.get("model.path"))
        if checkpoint_path is None:
            raise ValueError("Checkpoint path not specified")
        
        # Get model info
        self.model_info = get_checkpoint_info(checkpoint_path)
        
        # Load and map parameters
        tensors = self._load_and_map(checkpoint_path)
        
        # Extract features with detectors
        self._extract_features(tensors)
        
        # Score features
        self._score_features()
        
        # Generate outputs
        results = self._generate_results()
        
        # Save outputs to disk
        output_dir = Path(self.config.get("output.dir", "./scan_out"))
        save_visuals = self.config.get("output.save_visuals", True)
        topk_visuals = self.config.get("output.topk_visuals", 20)
        
        from spectre.core.output import generate_all_outputs
        generate_all_outputs(self, output_dir, save_visuals=save_visuals, topk_visuals=topk_visuals)
        
        return results
    
    def _load_and_map(self, checkpoint_path: Path) -> List[Dict]:
        """
        Load checkpoint and map parameters to roles.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            List of tensor dictionaries with name, array, role, layer_idx
        """
        tensors = []
        
        include_patterns = self.config.get("model.include", [])
        exclude_patterns = self.config.get("model.exclude", [])
        
        for name, array in tqdm(load_checkpoint(checkpoint_path), desc="Loading tensors"):
            # Filter by include/exclude patterns
            if include_patterns and not any(pattern in name for pattern in include_patterns):
                continue
            if exclude_patterns and any(pattern in name for pattern in exclude_patterns):
                continue
            
            # Map to role
            layer_idx = self.name_mapper.get_layer_index(name)
            role = self.name_mapper.map_name(name, array.shape, layer_idx)
            
            tensors.append({
                "name": name,
                "array": array,
                "role": role,
                "layer_idx": layer_idx,
                "shape": array.shape,
            })
        
        return tensors
    
    def _extract_features(self, tensors: List[Dict]):
        """
        Extract features using all enabled detectors.
        
        Args:
            tensors: List of tensor dictionaries
        """
        if not tensors:
            print("Warning: No tensors to process")
            return
        
        print(f"Processing {len(tensors)} tensors with {len(self.detectors)} detectors")
        for tensor_dict in tqdm(tensors, desc="Extracting features"):
            name = tensor_dict["name"]
            array = tensor_dict["array"]
            role = tensor_dict["role"]
            layer_idx = tensor_dict["layer_idx"]
            shape = tensor_dict["shape"]
            
            # Create TensorFeatures
            tf = TensorFeatures(
                name=name,
                role=role,
                layer_idx=layer_idx,
                shape=shape,
            )
            
            # Run all detectors
            for detector_name, detector in self.detectors.items():
                try:
                    features = detector.extract(array, name, role, layer_idx)
                    if features:
                        tf.features.update(features)
                except Exception as e:
                    # Log error but continue
                    print(f"Warning: Detector {detector_name} failed for {name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Always add tensor features, even if no features were extracted
            # This ensures we have a record of all tensors
            self.feature_store.add(tf)
    
    def _score_features(self):
        """Score features using scorer and calibrator."""
        # Initialize scorer
        self.scorer = Scorer(self.feature_store)
        
        # Standardize features
        self.scorer.standardize_features(use_robust=True)
        
        # Compute detector scores
        self.scorer.compute_detector_scores()
        
        # Compute ensemble scores
        self.scorer.compute_ensemble_score()
        
        # Calibrate if enabled
        if self.config.is_enabled("conformal"):
            coverage = self.config.get("conformal.coverage", 0.95)
            self.calibrator = ConformalCalibrator(self.feature_store, coverage=coverage)
            self.calibrator.calibrate(self.scorer)
    
    def _generate_results(self) -> Dict:
        """
        Generate final results dictionary.
        
        Returns:
            Results dictionary
        """
        summary = self.scorer.get_summary() if self.scorer else {
            "ensemble_sigma": 0.0,
            "flag": "GREEN",
            "suspects": [],
        }
        
        return {
            "model": self.model_info or {},
            "summary": summary,
            "per_tensor": [tf.to_dict() for tf in self.feature_store.features],
            "detectors": {
                name: {
                    "enabled": True,
                    "scores": {
                        tensor_name: scores.get(name, 0.0)
                        for tensor_name, scores in (self.scorer.detector_scores.items() if self.scorer else {})
                    }
                }
                for name in self.detectors.keys()
            },
        }

