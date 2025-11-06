"""D13: Conformal Outlier Calibration detector."""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from spectre.core.features import FeatureStore
from spectre.core.scoring import Scorer


class ConformalCalibrator:
    """D13: Conformal outlier calibration for distribution-free thresholding."""
    
    def __init__(self, feature_store: FeatureStore, coverage: float = 0.95):
        """
        Initialize conformal calibrator.
        
        Args:
            feature_store: FeatureStore instance
            coverage: Desired coverage (default 0.95)
        """
        self.feature_store = feature_store
        self.coverage = coverage
        self.calibrated_thresholds: Dict[str, float] = {}
    
    def calibrate(self, scorer: Scorer):
        """
        Calibrate thresholds using conformal prediction.
        
        Args:
            scorer: Scorer instance with computed scores
        """
        # Get all ensemble scores
        ensemble_scores = list(scorer.ensemble_scores.values())
        
        if len(ensemble_scores) < 10:
            # Not enough data for calibration
            return
        
        # Compute quantile for desired coverage
        # For 95% coverage, we want 95% of scores to be below threshold
        quantile = self.coverage
        threshold = np.quantile(ensemble_scores, quantile)
        
        self.calibrated_thresholds["ensemble"] = float(threshold)
        
        # Calibrate per-detector thresholds
        for detector in scorer.weights.keys():
            detector_scores = []
            for tensor_name, scores in scorer.detector_scores.items():
                if detector in scores:
                    detector_scores.append(abs(scores[detector]))
            
            if len(detector_scores) >= 10:
                threshold = np.quantile(detector_scores, quantile)
                self.calibrated_thresholds[detector] = float(threshold)
    
    def get_calibrated_flag(self, ensemble_score: float, detector_scores: Dict[str, float]) -> str:
        """
        Get calibrated risk flag.
        
        Args:
            ensemble_score: Ensemble score
            detector_scores: Dictionary of detector scores
            
        Returns:
            Calibrated risk flag
        """
        # Check for HARD_RED using calibrated threshold
        if "ensemble" in self.calibrated_thresholds:
            hard_red_threshold = np.quantile(
                [abs(s) for s in detector_scores.values()] if detector_scores else [0],
                0.99  # Top 1% for HARD_RED
            )
            
            max_detector = max(abs(s) for s in detector_scores.values()) if detector_scores else 0.0
            if max_detector >= hard_red_threshold:
                return "HARD_RED"
        
        # Use calibrated thresholds
        if "ensemble" in self.calibrated_thresholds:
            threshold = self.calibrated_thresholds["ensemble"]
            
            # Define thresholds relative to calibrated threshold
            red_threshold = threshold
            amber_threshold = threshold * 0.67  # 2/3 of red threshold
            
            if ensemble_score >= red_threshold:
                return "RED"
            elif ensemble_score >= amber_threshold:
                return "AMBER"
            else:
                return "GREEN"
        
        # Fallback to default thresholds
        if ensemble_score >= 3.0:
            return "RED"
        elif ensemble_score >= 2.0:
            return "AMBER"
        else:
            return "GREEN"
    
    def update_scorer_flags(self, scorer):
        """
        Update scorer flags using calibrated thresholds.
        
        Args:
            scorer: Scorer instance
        """
        # This would modify the scorer's get_flag method
        # For now, we'll create a wrapper
        pass

