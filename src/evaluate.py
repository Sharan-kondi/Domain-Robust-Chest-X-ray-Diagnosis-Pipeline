"""
evaluate.py — Evaluation pipeline for in-distribution and external test sets.

Computes per-class and macro AUC, generates the domain-gap comparison table,
and saves results for the README.
"""

import argparse
import os
import json
import yaml
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, classification_report
from typing import Dict

from src.device import get_device
from src.models.multitask_head import MultiTaskChestXray
from src.data.cache_tensors import load_cached
from src.data.datasets import build_test_loader
from src.data.label_harmonization import LABEL_NAMES


def evaluate_model(
    model: MultiTaskChestXray,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    site_name: str = "unknown",
) -> Dict:
    """Evaluate model on a test set and return metrics.

    Returns:
        Dict with per-class AUC, macro AUC, and raw predictions.
    """
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            output = model(images)
            probs = torch.sigmoid(output["logits"]).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Per-class AUC
    per_class_auc = {}
    for i, name in enumerate(LABEL_NAMES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            per_class_auc[name] = float(auc)
        except ValueError:
            per_class_auc[name] = None  # Only one class present

    # Macro AUC (over computable classes)
    valid_aucs = [v for v in per_class_auc.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    results = {
        "site": site_name,
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "n_samples": len(all_probs),
    }

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results — {site_name.upper()}")
    print(f"{'=' * 50}")
    print(f"  Macro AUC: {macro_auc:.4f}")
    for name, auc in per_class_auc.items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {name:20s}: {auc_str}")
    print(f"  N samples: {len(all_probs)}")

    return results


def run_evaluation(config_path: str, checkpoint_path: str) -> None:
    """Run full evaluation pipeline."""
    cfg = yaml.safe_load(open(config_path))
    data_cfg = yaml.safe_load(open("configs/data.yaml"))
    device = get_device()

    cache_dir = data_cfg.get("cache_dir", "./cached_tensors")
    image_size = data_cfg.get("image_size", 224)
    batch_size = cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 6)

    # Load model
    model = MultiTaskChestXray.load_from_checkpoint(checkpoint_path, map_location=device)

    all_results = {}

    # Evaluate on each site
    for site in ["nih", "vindr", "openi"]:
        try:
            images, labels = load_cached(cache_dir, site)
            test_loader = build_test_loader(
                images, labels, batch_size, num_workers, image_size,
            )
            results = evaluate_model(model, test_loader, device, site_name=site)
            all_results[site] = results
        except FileNotFoundError:
            print(f"[eval] Skipping {site} — cached tensors not found")

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = "results/evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    run_evaluation(args.config, args.checkpoint)
