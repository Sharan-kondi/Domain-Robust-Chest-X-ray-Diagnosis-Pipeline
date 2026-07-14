"""
subgroup_audit.py — Fairness and subgroup performance analysis.

Breaks down model performance by demographic subgroups (age, sex)
on the external test set (Open-I). Results go into MODEL_CARD.md.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from typing import Dict, Optional, List
import os
import json

from src.data.label_harmonization import LABEL_NAMES


def audit_by_subgroup(
    predictions: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    group_column: str,
) -> Dict[str, Dict]:
    """Compute per-class AUC broken down by a demographic group.

    Args:
        predictions: (N, C) predicted probabilities.
        labels: (N, C) ground truth multi-hot.
        metadata: DataFrame with at least `group_column`.
        group_column: Column name to stratify by (e.g., 'sex', 'age_group').

    Returns:
        Dict of {group_value: {label_name: auc, ..., 'macro_auc': float}}.
    """
    results = {}

    for group_val in metadata[group_column].unique():
        mask = metadata[group_column] == group_val
        group_preds = predictions[mask]
        group_labels = labels[mask]

        per_class = {}
        for i, name in enumerate(LABEL_NAMES):
            try:
                auc = roc_auc_score(group_labels[:, i], group_preds[:, i])
                per_class[name] = float(auc)
            except ValueError:
                per_class[name] = None

        valid = [v for v in per_class.values() if v is not None]
        macro = float(np.mean(valid)) if valid else 0.0

        results[str(group_val)] = {
            "per_class_auc": per_class,
            "macro_auc": macro,
            "n_samples": int(mask.sum()),
        }

    return results


def plot_subgroup_audit(
    audit_results: Dict[str, Dict],
    group_name: str,
    output_dir: str = "./figures",
    filename: Optional[str] = None,
) -> str:
    """Plot macro AUC by subgroup as a bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    groups = sorted(audit_results.keys())
    aucs = [audit_results[g]["macro_auc"] for g in groups]
    counts = [audit_results[g]["n_samples"] for g in groups]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(groups, aucs, alpha=0.85, edgecolor="black")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"n={count}", ha="center", fontsize=9)

    ax.set_xlabel(group_name)
    ax.set_ylabel("Macro AUC")
    ax.set_title(f"Subgroup Performance Audit — by {group_name}")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if filename is None:
        filename = f"subgroup_audit_{group_name.lower()}.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[audit] Saved subgroup plot -> {path}")
    return path


def save_audit_report(
    audit_results: Dict,
    output_path: str = "results/subgroup_audit.json",
) -> None:
    """Save audit results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"[audit] Saved audit report -> {output_path}")
