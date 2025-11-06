"""Configuration management with YAML support."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for Spectre."""
    
    def __init__(self, config_path: Optional[Path] = None, config_dict: Optional[Dict] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Dictionary with configuration (overrides file)
        """
        if config_dict:
            self.config = config_dict
        elif config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self._validate()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration matching PRD spec."""
        return {
            "model": {
                "path": None,
                "ruleset": "gpt_like",
                "include": [],  # Empty = include all
                "exclude": [],
            },
            "svd": {
                "enabled": True,
                "rank": 96,
                "power_iters": 2,
                "use_cuda": True,  # Use CUDA if available
            },
            "interlayer": {
                "enabled": True,
                "spectral_hist_bins": 64,
            },
            "distribution": {
                "enabled": True,
            },
            "robust": {
                "enabled": True,
            },
            "energy": {
                "enabled": True,
            },
            "rmt": {
                "enabled": True,
            },
            "spectrogram": {
                "enabled": True,
                "window": [16, 16],
                "stride": [16, 16],
                "method": "stft",
            },
            "gsp": {
                "enabled": True,
                "embed_dim": 16,
            },
            "tda": {
                "enabled": True,
            },
            "sequence_cp": {
                "enabled": True,
            },
            "ot": {
                "enabled": True,
            },
            "multiview": {
                "enabled": False,
            },
            "conformal": {
                "enabled": True,
                "coverage": 0.95,
            },
            "output": {
                "dir": "./scan_out",
                "save_visuals": True,
                "topk_visuals": 20,
            },
            "device": {
                "use_cuda": True,  # Use CUDA if available
                "cuda_device": 0,  # CUDA device index
            },
        }
    
    def _validate(self):
        """Validate configuration."""
        required_sections = ["model", "output"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        if self.config["model"].get("path") is None:
            # Allow None for testing, but warn
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Dot-separated key (e.g., "model.path")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Dot-separated key (e.g., "model.path")
            value: Value to set
        """
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def is_enabled(self, detector_name: str) -> bool:
        """
        Check if a detector is enabled.
        
        Args:
            detector_name: Name of detector (e.g., "svd", "interlayer")
            
        Returns:
            True if enabled, False otherwise
        """
        detector_config = self.config.get(detector_name, {})
        if isinstance(detector_config, dict):
            return detector_config.get("enabled", False)
        return False
    
    def save(self, path: Path):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save configuration
        """
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

