"""
pointing_game.py — Quantitative explainability evaluation.

The pointing game metric: does the maximum activation point of the
Grad-CAM heatmap fall inside the radiologist-annotated bounding box?

This is the key differentiator vs. "look at this pretty heatmap" explainability
— we actually validate against ground-truth annotations.
"""

import numpy as np
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt


def pointing_game_score(
    heatmap: np.ndarray,
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int] = (224, 224),
) -> bool:
    """Check if the max activation point falls inside the bounding box.

    Args:
        heatmap: (H, W) Grad-CAM heatmap.
        bbox: (x, y, w, h) normalized bounding box coordinates [0, 1].
        image_size: (H, W) of the image.

    Returns:
        True if max activation is inside the bbox (a "hit").
    """
    h, w = image_size
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    max_y, max_x = max_idx[0], max_idx[1]

    # Convert normalized bbox to pixel coordinates
    bx = int(bbox[0] * w)
    by = int(bbox[1] * h)
    bw = int(bbox[2] * w)
    bh = int(bbox[3] * h)

    hit = (bx <= max_x <= bx + bw) and (by <= max_y <= by + bh)
    return hit


def evaluate_pointing_game(
    heatmaps: List[np.ndarray],
    bboxes: List[Tuple[float, float, float, float]],
    labels: List[str],
    image_size: Tuple[int, int] = (224, 224),
) -> Dict[str, Dict[str, float]]:
    """Evaluate pointing game accuracy per pathology.

    Args:
        heatmaps: List of (H, W) heatmaps.
        bboxes: List of (x, y, w, h) normalized bounding boxes.
        labels: List of pathology label strings.
        image_size: (H, W) of images.

    Returns:
        Dict of {pathology: {'accuracy': float, 'hits': int, 'total': int}}.
    """
    results_by_label = {}

    for hm, bbox, label in zip(heatmaps, bboxes, labels):
        if label not in results_by_label:
            results_by_label[label] = {"hits": 0, "total": 0}

        hit = pointing_game_score(hm, bbox, image_size)
        results_by_label[label]["total"] += 1
        if hit:
            results_by_label[label]["hits"] += 1

    # Compute accuracies
    for label in results_by_label:
        total = results_by_label[label]["total"]
        hits = results_by_label[label]["hits"]
        results_by_label[label]["accuracy"] = hits / total if total > 0 else 0.0

    return results_by_label


def plot_pointing_game_results(
    results: Dict[str, Dict[str, float]],
    output_dir: str = "./figures",
    filename: str = "pointing_game_results.png",
) -> str:
    """Plot pointing game accuracy per pathology as a bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    labels = sorted(results.keys())
    accs = [results[l]["accuracy"] for l in labels]
    totals = [results[l]["total"] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, accs, alpha=0.85, edgecolor="black")

    for bar, total in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={total}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Pointing Game Accuracy")
    ax.set_title("Explainability Validation — Grad-CAM vs. Radiologist Bounding Boxes")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
