"""D8: Spectrogram & Signal-View Analysis detector."""

from typing import Dict, Optional

import numpy as np
from scipy import signal
from scipy.stats import entropy

from spectre.core.config import Config
from spectre.io.name_mapper import ParameterRole


class SpectrogramDetector:
    """D8: Spectrogram and signal-view analysis detector."""
    
    def __init__(self, config: Config):
        """
        Initialize spectrogram detector.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.window = tuple(config.get("spectrogram.window", [16, 16]))
        self.stride = tuple(config.get("spectrogram.stride", [16, 16]))
        self.method = config.get("spectrogram.method", "stft")
    
    def extract(
        self, 
        array: np.ndarray, 
        name: str, 
        role: ParameterRole, 
        layer_idx: Optional[int]
    ) -> Dict[str, float]:
        """
        Extract spectrogram features.
        
        Args:
            array: Weight tensor
            name: Parameter name
            role: Parameter role
            layer_idx: Layer index
            
        Returns:
            Dictionary of spectrogram features
        """
        if array.ndim < 2:
            return {}
        
        features = {}
        
        # Ensure 2D for spectrogram
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        
        m, n = array.shape
        
        if m < self.window[0] or n < self.window[1]:
            return {}
        
        try:
            if self.method == "stft":
                # 2D Short-Time Fourier Transform
                # Compute STFT for each row
                f, t, Zxx = signal.stft(
                    array[0], 
                    nperseg=min(self.window[0], m // 4),
                    noverlap=min(self.window[0] // 2, m // 8)
                )
                
                # Average across rows
                spectrogram = np.zeros_like(Zxx)
                for i in range(min(10, m)):  # Sample rows to avoid memory issues
                    _, _, Zxx_row = signal.stft(
                        array[i],
                        nperseg=min(self.window[0], m // 4),
                        noverlap=min(self.window[0] // 2, m // 8)
                    )
                    spectrogram += np.abs(Zxx_row)
                spectrogram = spectrogram / min(10, m)
                
            else:
                # Use 2D FFT as spectrogram
                fft_2d = np.fft.fft2(array)
                spectrogram = np.abs(fft_2d)
            
            # Texture features from spectrogram
            # Entropy
            hist, _ = np.histogram(spectrogram.flatten(), bins=64, density=True)
            hist = hist / (np.sum(hist) + 1e-10)
            hist = hist[hist > 0]
            if len(hist) > 0:
                spectrogram_entropy = entropy(hist)
                features["spectrogram.entropy"] = float(spectrogram_entropy)
            
            # Anisotropy: measure of directional structure
            # Compute variance along rows vs columns
            row_variance = np.var(spectrogram, axis=1)
            col_variance = np.var(spectrogram, axis=0)
            
            if np.mean(row_variance) > 0 and np.mean(col_variance) > 0:
                anisotropy = np.abs(np.mean(row_variance) - np.mean(col_variance)) / (
                    np.mean(row_variance) + np.mean(col_variance) + 1e-10
                )
                features["spectrogram.anisotropy"] = float(anisotropy)
            
            # Contrast: measure of local variation
            # Compute local contrast using gradient
            grad_x = np.gradient(spectrogram, axis=1)
            grad_y = np.gradient(spectrogram, axis=0)
            contrast = np.sqrt(np.real(grad_x) ** 2 + np.real(grad_y) ** 2)
            features["spectrogram.contrast_mean"] = float(np.mean(contrast))
            features["spectrogram.contrast_std"] = float(np.std(contrast))
            
            # Energy distribution
            spectrogram_real = np.real(spectrogram)
            total_energy = np.sum(spectrogram_real ** 2)
            if total_energy > 0:
                # Energy concentration in top frequencies
                sorted_energy = np.sort(spectrogram_real.flatten() ** 2)[::-1]
                top_10_percent = int(len(sorted_energy) * 0.1)
                if top_10_percent > 0:
                    energy_concentration = np.sum(sorted_energy[:top_10_percent]) / total_energy
                    features["spectrogram.energy_concentration"] = float(energy_concentration)
            
            # Dominant frequency
            if spectrogram.size > 0:
                max_idx = np.unravel_index(np.argmax(spectrogram), spectrogram.shape)
                features["spectrogram.dominant_freq_row"] = float(max_idx[0] / spectrogram.shape[0])
                features["spectrogram.dominant_freq_col"] = float(max_idx[1] / spectrogram.shape[1])
            
        except Exception:
            pass
        
        return features

