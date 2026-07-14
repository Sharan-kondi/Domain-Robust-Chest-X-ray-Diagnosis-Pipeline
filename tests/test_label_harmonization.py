"""
test_label_harmonization.py — Unit tests for label harmonization logic.
"""

import pytest
import numpy as np
from src.data.label_harmonization import LABEL_NAMES, NUM_CLASSES


def test_label_count():
    """Verify we have exactly 8 shared labels."""
    assert NUM_CLASSES == 8
    assert len(LABEL_NAMES) == 8


def test_label_names_complete():
    """Check all expected labels are present."""
    expected = {
        "Atelectasis", "Cardiomegaly", "Consolidation", "Effusion",
        "Pneumonia", "Pneumothorax", "Nodule/Mass", "No Finding",
    }
    assert set(LABEL_NAMES) == expected


def test_no_finding_is_last():
    """No Finding should be the last label (index 7)."""
    assert LABEL_NAMES[-1] == "No Finding"
    assert LABEL_NAMES.index("No Finding") == 7
