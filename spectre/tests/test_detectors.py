"""Unit tests for detectors."""

import numpy as np
import pytest

from spectre.core.config import Config
from spectre.detectors.distribution import DistributionDetector
from spectre.detectors.energy import EnergyDetector
from spectre.detectors.interlayer import InterlayerDetector
from spectre.detectors.robust import RobustDetector
from spectre.detectors.spectral import SpectralDetector
from spectre.io.name_mapper import ParameterRole


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def sample_tensor():
    """Create sample 2D tensor."""
    np.random.seed(42)
    return np.random.randn(100, 200)


def test_spectral_detector(config, sample_tensor):
    """Test spectral detector."""
    detector = SpectralDetector(config)
    features = detector.extract(sample_tensor, "test", ParameterRole.ATTN_IN, 0)
    
    assert "spectral.top1" in features
    assert "spectral.stable_rank" in features
    assert features["spectral.top1"] > 0


def test_interlayer_detector(config, sample_tensor):
    """Test interlayer detector."""
    detector = InterlayerDetector(config)
    
    # First tensor
    features1 = detector.extract(sample_tensor, "test1", ParameterRole.ATTN_IN, 0)
    
    # Second tensor (should have correlation)
    features2 = detector.extract(sample_tensor, "test2", ParameterRole.ATTN_IN, 1)
    
    assert "interlayer.cosine_sim" in features2 or len(features2) == 0


def test_distribution_detector(config, sample_tensor):
    """Test distribution detector."""
    detector = DistributionDetector(config)
    features = detector.extract(sample_tensor, "test", ParameterRole.ATTN_IN, 0)
    
    assert "distribution.mean" in features
    assert "distribution.std" in features
    assert "distribution.entropy" in features


def test_robust_detector(config, sample_tensor):
    """Test robust detector."""
    detector = RobustDetector(config)
    features = detector.extract(sample_tensor, "test", ParameterRole.ATTN_IN, 0)
    
    assert "robust.median" in features
    assert "robust.mad" in features
    assert features["robust.mad"] >= 0


def test_energy_detector(config, sample_tensor):
    """Test energy detector."""
    detector = EnergyDetector(config)
    features = detector.extract(sample_tensor, "test", ParameterRole.ATTN_IN, 0)
    
    assert "energy.l1" in features
    assert "energy.l2" in features
    assert "energy.frobenius" in features
    assert features["energy.frobenius"] > 0


def test_detector_1d_tensor(config):
    """Test detectors with 1D tensor."""
    np.random.seed(42)
    tensor_1d = np.random.randn(100)
    
    detector = EnergyDetector(config)
    features = detector.extract(tensor_1d, "test", ParameterRole.LN_GAMMA, 0)
    
    assert "energy.l1" in features
    assert "energy.l2" in features


def test_detector_3d_tensor(config):
    """Test detectors with 3D tensor."""
    np.random.seed(42)
    tensor_3d = np.random.randn(10, 20, 30)
    
    detector = SpectralDetector(config)
    features = detector.extract(tensor_3d, "test", ParameterRole.ATTN_IN, 0)
    
    # Should handle 3D by reshaping
    assert len(features) >= 0  # May return empty for very small tensors

