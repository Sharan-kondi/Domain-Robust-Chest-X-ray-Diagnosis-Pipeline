"""
drift_monitor.py — Data drift monitoring using Evidently.

Compares incoming inference images against the training data distribution
to detect covariate shift (different scanner, different patient population).
"""

import numpy as np
from typing import Dict, Optional, List
import json
import os
from datetime import datetime


class DriftMonitor:
    """Simple pixel-statistics-based drift monitor.

    Tracks running statistics of incoming images and compares against
    stored training-set baselines. Uses Evidently for detailed reports
    when available.
    """

    def __init__(
        self,
        reference_stats: Optional[Dict] = None,
        threshold: float = 0.1,
    ):
        self.reference_stats = reference_stats or {
            "mean": 0.5,
            "std": 0.25,
        }
        self.threshold = threshold
        self.incoming_means = []
        self.incoming_stds = []
        self.predictions_log = []

    def log_image(self, image_array: np.ndarray) -> None:
        """Record statistics for an incoming image."""
        self.incoming_means.append(float(image_array.mean()))
        self.incoming_stds.append(float(image_array.std()))

    def log_prediction(self, prediction: Dict) -> None:
        """Record a prediction for monitoring."""
        self.predictions_log.append({
            "timestamp": datetime.now().isoformat(),
            **prediction,
        })

    def check_drift(self) -> Dict:
        """Check if incoming data has drifted from training distribution.

        Returns:
            Dict with drift_detected, drift_score, and details.
        """
        if len(self.incoming_means) < 10:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "n_samples_analyzed": len(self.incoming_means),
                "message": "Need at least 10 samples for drift detection",
            }

        current_mean = np.mean(self.incoming_means[-100:])
        current_std = np.mean(self.incoming_stds[-100:])

        mean_drift = abs(current_mean - self.reference_stats["mean"])
        std_drift = abs(current_std - self.reference_stats["std"])
        drift_score = (mean_drift + std_drift) / 2

        return {
            "drift_detected": drift_score > self.threshold,
            "drift_score": float(drift_score),
            "n_samples_analyzed": len(self.incoming_means),
            "current_mean": float(current_mean),
            "current_std": float(current_std),
            "reference_mean": self.reference_stats["mean"],
            "reference_std": self.reference_stats["std"],
        }

    def save_log(self, path: str = "logs/drift_log.json") -> None:
        """Persist drift monitoring log."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "predictions": self.predictions_log[-1000:],
                "drift_report": self.check_drift(),
            }, f, indent=2)
