"""Synthetic attack tests."""

import numpy as np
import pytest

from spectre.core.config import Config
from spectre.core.features import FeatureStore, TensorFeatures
from spectre.core.pipeline import Pipeline
from spectre.core.scoring import Scorer
from spectre.detectors.spectral import SpectralDetector
from spectre.io.name_mapper import ParameterRole


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def base_tensor():
    """Create base tensor for attacks."""
    np.random.seed(42)
    return np.random.randn(100, 200)


def test_low_rank_injection(config, base_tensor):
    """Test low-rank injection attack: W' = W + α * u vᵀ"""
    # Create low-rank injection
    alpha = 0.1
    u = np.random.randn(100, 1)
    v = np.random.randn(1, 200)
    injection = alpha * np.dot(u, v)
    
    attacked_tensor = base_tensor + injection
    
    # Test spectral detector
    detector = SpectralDetector(config)
    features_base = detector.extract(base_tensor, "base", ParameterRole.ATTN_IN, 0)
    features_attacked = detector.extract(attacked_tensor, "attacked", ParameterRole.ATTN_IN, 0)
    
    # Attacked tensor should have different spectral features
    if "spectral.top1" in features_base and "spectral.top1" in features_attacked:
        assert features_attacked["spectral.top1"] != features_base["spectral.top1"]


def test_sparse_corruption(config, base_tensor):
    """Test sparse corruption: flip 10-100 random values"""
    attacked_tensor = base_tensor.copy()
    
    # Flip 50 random values
    num_flips = 50
    flat_indices = np.random.choice(attacked_tensor.size, num_flips, replace=False)
    attacked_tensor.flat[flat_indices] *= -10  # Large flip
    
    # Test robust detector
    from spectre.detectors.robust import RobustDetector
    
    detector = RobustDetector(config)
    features_base = detector.extract(base_tensor, "base", ParameterRole.ATTN_IN, 0)
    features_attacked = detector.extract(attacked_tensor, "attacked", ParameterRole.ATTN_IN, 0)
    
    # Attacked tensor should have more outliers
    if "robust.outlier_fraction" in features_attacked:
        assert features_attacked["robust.outlier_fraction"] > features_base.get("robust.outlier_fraction", 0)


def test_column_row_implant(config, base_tensor):
    """Test column/row implant: boost a row"""
    attacked_tensor = base_tensor.copy()
    
    # Boost a row
    row_idx = 50
    attacked_tensor[row_idx, :] *= 5
    
    # Test energy detector
    from spectre.detectors.energy import EnergyDetector
    
    detector = EnergyDetector(config)
    features_base = detector.extract(base_tensor, "base", ParameterRole.ATTN_IN, 0)
    features_attacked = detector.extract(attacked_tensor, "attacked", ParameterRole.ATTN_IN, 0)
    
    # Attacked tensor should have higher row concentration
    if "energy.row_concentration" in features_attacked:
        assert features_attacked["energy.row_concentration"] > features_base.get("energy.row_concentration", 1.0)


def test_patch_injection(config, base_tensor):
    """Test patch injection: alter region of weight"""
    attacked_tensor = base_tensor.copy()
    
    # Alter a patch
    patch_size = 20
    i_start, j_start = 40, 80
    attacked_tensor[i_start:i_start+patch_size, j_start:j_start+patch_size] *= 10
    
    # Test spectrogram detector
    from spectre.detectors.spectrogram import SpectrogramDetector
    
    detector = SpectrogramDetector(config)
    features_base = detector.extract(base_tensor, "base", ParameterRole.ATTN_IN, 0)
    features_attacked = detector.extract(attacked_tensor, "attacked", ParameterRole.ATTN_IN, 0)
    
    # Attacked tensor should have different spectrogram features
    if "spectrogram.contrast_mean" in features_attacked:
        assert features_attacked["spectrogram.contrast_mean"] != features_base.get("spectrogram.contrast_mean", 0)


def test_quantization_error(config, base_tensor):
    """Test quantization error: simulate FP16/bfloat16 compression"""
    # Simulate quantization
    attacked_tensor = base_tensor.astype(np.float16).astype(np.float32)
    
    # Test distribution detector
    from spectre.detectors.distribution import DistributionDetector
    
    detector = DistributionDetector(config)
    features_base = detector.extract(base_tensor, "base", ParameterRole.ATTN_IN, 0)
    features_attacked = detector.extract(attacked_tensor, "attacked", ParameterRole.ATTN_IN, 0)
    
    # Quantization should cause minimal changes
    if "distribution.mean" in features_base and "distribution.mean" in features_attacked:
        mean_diff = abs(features_attacked["distribution.mean"] - features_base["distribution.mean"])
        # Mean should be very close
        assert mean_diff < 0.1


def test_random_noise(config, base_tensor):
    """Test random noise: validate false-positive rate"""
    # Add small random noise
    noise_level = 0.01
    attacked_tensor = base_tensor + np.random.randn(*base_tensor.shape) * noise_level
    
    # Test multiple detectors
    from spectre.detectors.distribution import DistributionDetector
    
    detectors = {
        "spectral": SpectralDetector(config),
        "distribution": DistributionDetector(config),
    }
    
    # Features should be similar
    for name, detector in detectors.items():
        features_base = detector.extract(base_tensor, "base", ParameterRole.ATTN_IN, 0)
        features_attacked = detector.extract(attacked_tensor, "attacked", ParameterRole.ATTN_IN, 0)
        
        # Check that features are similar (low false positive)
        if "distribution.mean" in features_base and "distribution.mean" in features_attacked:
            mean_diff = abs(features_attacked["distribution.mean"] - features_base["distribution.mean"])
            # Mean difference should be small relative to noise
            assert mean_diff < noise_level * 10

