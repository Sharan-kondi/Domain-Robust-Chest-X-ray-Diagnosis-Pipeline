"""
coral.py — CORAL (CORrelation ALignment) domain generalization loss.

Hand-written (~30 lines of core logic) — no external DG library needed.
CORAL aligns the second-order statistics (covariance) of feature
representations between source domains during training.

Reference: Sun & Saenko, "Deep CORAL", ECCV 2016 Workshops.
"""

import torch
import torch.nn as nn


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """Compute CORAL loss between two sets of feature vectors.

    Minimizes the distance between the covariance matrices of the source
    and target feature distributions. This encourages the model to learn
    domain-invariant representations.

    Args:
        source_features: (N_s, D) features from source domain.
        target_features: (N_t, D) features from target domain.

    Returns:
        Scalar CORAL loss.
    """
    d = source_features.shape[1]

    # Center the features
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)

    # Compute covariance matrices
    n_s = source_features.shape[0]
    n_t = target_features.shape[0]

    cov_source = (source_centered.T @ source_centered) / (n_s - 1 + 1e-8)
    cov_target = (target_centered.T @ target_centered) / (n_t - 1 + 1e-8)

    # Frobenius norm of the difference
    loss = (cov_source - cov_target).pow(2).sum() / (4 * d * d)

    return loss


class CORALLoss(nn.Module):
    """Module wrapper for CORAL loss with configurable weight."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted CORAL loss.

        Args:
            source_features: (N_s, D) from domain A.
            target_features: (N_t, D) from domain B.

        Returns:
            Weighted scalar loss.
        """
        return self.weight * coral_loss(source_features, target_features)
