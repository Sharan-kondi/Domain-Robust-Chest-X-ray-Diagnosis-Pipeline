"""
train.py — Main training entrypoint.

Supports two modes:
  1. Baseline fine-tuning (classification + bbox, no domain alignment)
  2. CORAL-aligned training (adds CORAL loss between NIH and VinDr features)

Usage:
    python -m src.train --config configs/finetune.yaml
    python -m src.train --config configs/dg_coral.yaml
"""

import argparse
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.device import get_device
from src.models.multitask_head import MultiTaskChestXray
from src.data.cache_tensors import load_cached
from src.data.datasets import build_train_val_loaders


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path: str) -> None:
    """Run training based on config."""
    cfg = load_config(config_path)

    # Load base config if specified
    if "base_config" in cfg:
        base_path = os.path.join(os.path.dirname(config_path), cfg["base_config"])
        base_cfg = load_config(base_path)
        base_cfg.update(cfg)
        cfg = base_cfg

    print(f"[train] Config: {config_path}")
    print(f"[train] Backbone: {cfg.get('backbone', 'resnet18')}")
    print(f"[train] Epochs: {cfg.get('epochs', 5)}")

    device = get_device()

    # Load cached training data
    data_cfg = load_config("configs/data.yaml")
    cache_dir = data_cfg.get("cache_dir", "./cached_tensors")

    # Load and combine training sites dynamically
    train_images = []
    train_labels = []
    train_site_ids = []

    site_idx = 0
    for name, info in data_cfg.get("datasets", {}).items():
        if info.get("role") == "train":
            try:
                imgs, lbls = load_cached(cache_dir, name)
                train_images.append(imgs)
                train_labels.append(lbls)
                train_site_ids.append(torch.full((len(imgs),), site_idx, dtype=torch.long))
                print(f"[train] Loaded {name} dataset for training (site_id={site_idx}).")
                site_idx += 1
            except FileNotFoundError:
                print(f"[train] {name} dataset not found in cache. Skipping.")

    if not train_images:
        raise FileNotFoundError("No training datasets found in cache! Please cache training data first.")

    images = torch.cat(train_images)
    labels = torch.cat(train_labels)
    site_ids = torch.cat(train_site_ids)

    # Build data loaders
    train_loader, val_loader = build_train_val_loaders(
        images, labels, site_ids,
        val_fraction=data_cfg.get("val_fraction", 0.15),
        batch_size=cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 6),
        image_size=data_cfg.get("image_size", 224),
        seed=cfg.get("seed", 42),
    )

    # Initialize model
    model = MultiTaskChestXray(
        backbone_name=cfg.get("backbone", "resnet18"),
        pretrained=cfg.get("pretrained", True),
        num_classes=len(data_cfg.get("labels", [])) or 8,
        bbox_enabled=cfg.get("bbox_enabled", True),
        learning_rate=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-5),
        cls_loss_weight=cfg.get("loss_weights", {}).get("classification", 1.0),
        bbox_loss_weight=cfg.get("loss_weights", {}).get("bbox_regression", 0.5),
        coral_lambda=cfg.get("coral_lambda", 0.0),
    )

    # Callbacks
    checkpoint_dir = cfg.get("checkpoint_dir", "./checkpoints")
    experiment_name = cfg.get("experiment_name", "baseline")
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{experiment_name}-{{epoch:02d}}-{{val_auc:.3f}}",
            monitor="val_auc",
            mode="max",
            save_top_k=2,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            verbose=True,
        ),
    ]

    # Trainer
    accelerator = "cpu"
    if torch.cuda.is_available():
        accelerator = "gpu"

    trainer = pl.Trainer(
        max_epochs=cfg.get("epochs", 5),
        accelerator=accelerator,
        callbacks=callbacks,
        deterministic=cfg.get("deterministic", True),
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"[train] Training complete. Best model saved to {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chest X-ray model")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    args = parser.parse_args()
    train(args.config)
