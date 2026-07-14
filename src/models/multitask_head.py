"""
multitask_head.py — Multi-task model: classification + bounding box regression.

Architecture:
    BackboneEncoder → shared features
        ├── Classification head → (B, num_classes) logits
        └── BBox regression head → (B, 4) normalized box coordinates

The classification head uses BCE with logits for multi-label classification.
The bbox head uses Smooth L1 loss, applied only to images with bbox annotations.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score
import numpy as np

from src.models.backbone import BackboneEncoder
from src.data.label_harmonization import LABEL_NAMES, NUM_CLASSES


class MultiTaskChestXray(pl.LightningModule):
    """Multi-task chest X-ray model with classification + bbox heads.

    This is the core model used for both baseline and CORAL training.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        bbox_enabled: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        cls_loss_weight: float = 1.0,
        bbox_loss_weight: float = 0.5,
        dropout_rate: float = 0.3,
        coral_lambda: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []

        # Backbone
        self.backbone = BackboneEncoder(backbone_name, pretrained)
        feat_dim = self.backbone.get_feature_dim()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        # Bounding box regression head (optional)
        self.bbox_enabled = bbox_enabled
        if bbox_enabled:
            self.bbox_regressor = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4),  # (x, y, w, h) normalized
                nn.Sigmoid(),       # Bound to [0, 1]
            )

        # Losses
        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.bbox_criterion = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 3, H, W) images.

        Returns:
            Dict with 'logits' and optionally 'bbox'.
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        output = {"logits": logits, "features": features}

        if self.bbox_enabled:
            bbox = self.bbox_regressor(features)
            output["bbox"] = bbox

        return output

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss."""
        output = self(batch["image"])

        # Classification loss (always)
        cls_loss = self.cls_criterion(output["logits"], batch["label"])

        total_loss = self.hparams.cls_loss_weight * cls_loss
        losses = {"cls_loss": cls_loss}

        # Bbox loss (only if annotations present)
        if self.bbox_enabled and "bbox" in batch:
            bbox_mask = batch.get("bbox_mask", torch.ones(len(batch["label"]), dtype=torch.bool))
            if bbox_mask.any():
                bbox_loss = self.bbox_criterion(
                    output["bbox"][bbox_mask],
                    batch["bbox"][bbox_mask],
                )
                total_loss = total_loss + self.hparams.bbox_loss_weight * bbox_loss
                losses["bbox_loss"] = bbox_loss

        # CORAL loss (optional domain alignment)
        if hasattr(self.hparams, "coral_lambda") and self.hparams.coral_lambda > 0 and "site_id" in batch:
            site_ids = batch["site_id"]
            unique_sites = torch.unique(site_ids)
            if len(unique_sites) > 1:
                # Filter features by site ID
                feat_0 = output["features"][site_ids == unique_sites[0]]
                feat_1 = output["features"][site_ids == unique_sites[1]]
                if len(feat_0) > 1 and len(feat_1) > 1:
                    from src.dg.coral import coral_loss
                    c_loss = coral_loss(feat_0, feat_1)
                    total_loss = total_loss + self.hparams.coral_lambda * c_loss
                    losses["coral_loss"] = c_loss

        losses["total_loss"] = total_loss
        losses["logits"] = output["logits"]
        return losses

    def training_step(self, batch, batch_idx):
        losses = self._compute_loss(batch)
        self.log("train_loss", losses["total_loss"], prog_bar=True)
        self.log("train_cls_loss", losses["cls_loss"])
        if "coral_loss" in losses:
            self.log("train_coral_loss", losses["coral_loss"])
        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        losses = self._compute_loss(batch)
        self.log("val_loss", losses["total_loss"], prog_bar=True)

        probs = torch.sigmoid(losses["logits"]).detach().cpu()
        labels = batch["label"].detach().cpu()
        self.validation_step_outputs.append({"probs": probs, "labels": labels})
        return losses["total_loss"]

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        
        all_probs = torch.cat([x["probs"] for x in self.validation_step_outputs]).numpy()
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).numpy()
        
        try:
            auc = roc_auc_score(all_labels, all_probs, average="macro")
            self.log("val_auc", float(auc), prog_bar=True, sync_dist=True)
        except ValueError:
            pass
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 5,
        )
        return [optimizer], [scheduler]
