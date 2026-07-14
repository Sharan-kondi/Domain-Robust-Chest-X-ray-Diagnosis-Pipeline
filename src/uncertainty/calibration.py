"""
calibration.py — Model calibration assessment and reliability diagrams.

Computes Expected Calibration Error (ECE) and plots reliability diagrams
to assess whether predicted probabilities match observed frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import os


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, Dict]:
    """Compute Expected Calibration Error (ECE).

    Args:
        y_true: (N,) binary ground truth labels.
        y_prob: (N,) predicted probabilities.
        n_bins: Number of bins for calibration.

    Returns:
        (ece_value, bin_details) where bin_details contains per-bin info.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= low) & (y_prob < high)
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)

        count = mask.sum()
        bin_counts.append(count)

        if count > 0:
            accuracy = y_true[mask].mean()
            confidence = y_prob[mask].mean()
        else:
            accuracy = 0.0
            confidence = 0.0

        bin_accuracies.append(accuracy)
        bin_confidences.append(confidence)

    bin_counts = np.array(bin_counts)
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)

    total = bin_counts.sum()
    if total == 0:
        return 0.0, {}

    ece = np.sum(bin_counts / total * np.abs(bin_accuracies - bin_confidences))

    return float(ece), {
        "bin_boundaries": bin_boundaries,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    label: str = "Model",
    output_dir: str = "./figures",
    filename: str = "reliability_diagram.png",
) -> str:
    """Plot a reliability (calibration) diagram.

    Args:
        y_true: (N,) binary ground truth.
        y_prob: (N,) predicted probabilities.
        n_bins: Number of bins.
        label: Legend label.
        output_dir: Where to save.
        filename: Output filename.

    Returns:
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    ece, bins = expected_calibration_error(y_true, y_prob, n_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    bin_mids = (bins["bin_boundaries"][:-1] + bins["bin_boundaries"][1:]) / 2
    ax1.bar(bin_mids, bins["bin_accuracies"], width=1.0 / n_bins, alpha=0.7,
            edgecolor="black", label=f"{label} (ECE={ece:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Histogram of predictions
    ax2.bar(bin_mids, bins["bin_counts"], width=1.0 / n_bins, alpha=0.7, color="gray")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[calibration] Saved reliability diagram -> {path}")
    return path
