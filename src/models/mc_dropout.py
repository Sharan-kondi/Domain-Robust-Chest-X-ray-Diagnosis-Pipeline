"""
mc_dropout.py — Monte Carlo Dropout for uncertainty estimation.

At inference, keeps dropout layers active and runs N stochastic forward
passes. Returns mean predictions and uncertainty (predictive entropy or
standard deviation).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm


def enable_mc_dropout(model: nn.Module) -> None:
    """Set all dropout layers to training mode (active at inference)."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    images: torch.Tensor,
    n_passes: int = 20,
    device: torch.device = None,
) -> Dict[str, np.ndarray]:
    """Run MC Dropout inference.

    Args:
        model: Trained model with dropout layers.
        images: (B, 3, H, W) input batch.
        n_passes: Number of stochastic forward passes.
        device: Device to run on.

    Returns:
        Dict with:
            'mean_probs': (B, C) mean predicted probabilities
            'std_probs': (B, C) standard deviation of probabilities
            'entropy': (B,) predictive entropy per sample
            'all_probs': (n_passes, B, C) all forward pass probabilities
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    enable_mc_dropout(model)  # Keep dropout active

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            output = model(images.to(device))
            logits = output["logits"] if isinstance(output, dict) else output
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.stack(all_probs)   # (n_passes, B, C)
    mean_probs = all_probs.mean(axis=0)  # (B, C)
    std_probs = all_probs.std(axis=0)    # (B, C)

    # Predictive entropy: H = -sum(p * log(p) + (1-p) * log(1-p))
    eps = 1e-8
    entropy = -(
        mean_probs * np.log(mean_probs + eps)
        + (1 - mean_probs) * np.log(1 - mean_probs + eps)
    ).sum(axis=1)  # (B,)

    return {
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "entropy": entropy,
        "all_probs": all_probs,
    }


def compute_routing_stats(
    entropy: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute human-review routing statistics.

    Args:
        entropy: (N,) predictive entropy per sample.
        threshold: Entropy threshold above which samples are routed.

    Returns:
        Dict with routing rate, mean entropy of routed/confident samples.
    """
    routed_mask = entropy > threshold
    routing_rate = routed_mask.mean()

    return {
        "threshold": threshold,
        "routing_rate": float(routing_rate),
        "n_routed": int(routed_mask.sum()),
        "n_confident": int((~routed_mask).sum()),
        "mean_entropy_routed": float(entropy[routed_mask].mean()) if routed_mask.any() else 0.0,
        "mean_entropy_confident": float(entropy[~routed_mask].mean()) if (~routed_mask).any() else 0.0,
    }
