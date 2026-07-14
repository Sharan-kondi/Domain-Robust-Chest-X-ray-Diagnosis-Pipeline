"""
gradcam_eval.py — Grad-CAM generation and visualization.

Uses pytorch-grad-cam to produce class activation maps that highlight
which regions of the X-ray the model focuses on for each pathology.
"""

import torch
import numpy as np
from typing import Optional, List, Dict
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import os


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out["logits"]
        return out


def generate_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: int = 0,
    device: torch.device = None,
    method: str = "gradcam",
) -> np.ndarray:
    """Generate Grad-CAM heatmaps for a batch of images.

    Args:
        model: Trained model.
        images: (B, 3, H, W) input images.
        target_layer: The convolutional layer to compute CAM from.
        target_class: Which class to generate CAM for.
        device: Compute device.
        method: 'gradcam' or 'gradcam++'.

    Returns:
        (B, H, W) numpy array of heatmaps in [0, 1].
    """
    if device is None:
        device = next(model.parameters()).device

    wrapper = LogitsWrapper(model)

    # Custom target for multi-label output
    class ClassTarget:
        def __init__(self, category):
            self.category = category

        def __call__(self, model_output):
            return model_output[self.category]

    cam_class = GradCAMPlusPlus if method == "gradcam++" else GradCAM

    cam = cam_class(model=wrapper, target_layers=[target_layer])
    targets = [ClassTarget(target_class)] * len(images)

    grayscale_cam = cam(input_tensor=images.to(device), targets=targets)
    return grayscale_cam  # (B, H, W)


def visualize_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Grad-CAM",
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Overlay Grad-CAM heatmap on an image and optionally save.

    Args:
        image: (H, W, 3) float32 image in [0, 1].
        heatmap: (H, W) float32 heatmap in [0, 1].
        title: Plot title.
        output_path: If provided, save figure here.

    Returns:
        Path to saved figure, or None.
    """
    overlay = show_cam_on_image(image, heatmap, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path
    else:
        plt.show()
        plt.close(fig)
        return None
