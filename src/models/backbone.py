"""
backbone.py — Feature extractor backbone using timm.

Supports ResNet18 and EfficientNet-B0 with ImageNet pretrained weights.
Returns feature vectors from the global average pooling layer.
"""

import timm
import torch
import torch.nn as nn
from typing import Optional


class BackboneEncoder(nn.Module):
    """Feature extractor wrapping a timm model.

    Removes the final classification head and returns pooled feature vectors.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head → returns features
        )
        self.feature_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features.

        Args:
            x: (B, 3, H, W) input images.

        Returns:
            (B, feature_dim) feature vectors.
        """
        return self.model(x)

    def get_feature_dim(self) -> int:
        return self.feature_dim
