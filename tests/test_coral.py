"""
test_coral.py — Unit tests for CORAL loss.
"""

import pytest
import torch
from src.dg.coral import coral_loss, CORALLoss


def test_coral_same_distribution():
    """CORAL loss should be ~0 for identical distributions."""
    torch.manual_seed(42)
    features = torch.randn(64, 128)
    loss = coral_loss(features, features)
    assert loss.item() < 1e-6


def test_coral_different_distributions():
    """CORAL loss should be > 0 for different distributions."""
    torch.manual_seed(42)
    source = torch.randn(64, 128)
    target = torch.randn(64, 128) * 3 + 2  # Shifted and scaled
    loss = coral_loss(source, target)
    assert loss.item() > 0.0


def test_coral_loss_module():
    """CORALLoss module should apply weight correctly."""
    torch.manual_seed(42)
    source = torch.randn(32, 64)
    target = torch.randn(32, 64) * 2

    base_loss = coral_loss(source, target)
    weighted_loss = CORALLoss(weight=0.5)(source, target)

    assert abs(weighted_loss.item() - 0.5 * base_loss.item()) < 1e-5


def test_coral_gradient_flows():
    """CORAL loss should allow gradient flow."""
    source = torch.randn(16, 32, requires_grad=True)
    target = torch.randn(16, 32)
    loss = coral_loss(source, target)
    loss.backward()
    assert source.grad is not None
    assert source.grad.shape == source.shape
