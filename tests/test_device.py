"""
test_device.py — Unit tests for device selection.
"""

import torch
from src.device import get_device


def test_get_device_returns_device():
    """get_device should return a torch.device or compatible object."""
    device = get_device()
    # Should be usable with torch tensors
    t = torch.zeros(1).to(device)
    assert t is not None
