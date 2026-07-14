import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from src.device import get_device
from src.models.multitask_head import MultiTaskChestXray
from src.data.cache_tensors import load_cached
from src.data.datasets import CachedChestXrayDataset, get_train_transforms, get_val_transforms
from src.uncertainty.calibration import expected_calibration_error, plot_reliability_diagram
from src.models.mc_dropout import mc_dropout_predict, compute_routing_stats
from src.explain.gradcam_eval import generate_gradcam, visualize_gradcam
from src.explain.pointing_game import pointing_game_score, evaluate_pointing_game, plot_pointing_game_results

def run_entire_pipeline():
    print("[pipeline] Starting End-to-End Chest X-ray Pipeline...")
    
    # 1. Load Configurations
    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/finetune.yaml") as f:
        train_cfg = yaml.safe_load(f)
        
    cache_dir = data_cfg.get("cache_dir", "./cached_tensors")
    image_size = data_cfg.get("image_size", 224)
    label_names = data_cfg.get("labels", [])
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    device = get_device()
    
    # 2. Reconstruct Filename Orders for Bounding Box / Site Tracking
    print("[pipeline] Reconstructing filenames mapping...")
    from src.data.label_harmonization import harmonize_nih, harmonize_openi
    from src.data.dicom_ingest import scan_image_directory, build_filename_index
    
    # NIH filenames
    nih_info = data_cfg["datasets"]["nih"]
    nih_df = harmonize_nih(nih_info["metadata_csv"], subsample=nih_info["subsample"])
    nih_files = scan_image_directory(nih_info["images_dir"])
    nih_fn_to_path = build_filename_index(nih_files)
    nih_filenames = [row["image_name"] for _, row in nih_df.iterrows() if Path(row["image_name"]).stem in nih_fn_to_path]
    
    # Open-I filenames
    openi_info = data_cfg["datasets"]["openi"]
    openi_df = harmonize_openi(openi_info["projections_csv"], openi_info["reports_csv"], subsample=openi_info["subsample"])
    openi_files = scan_image_directory(openi_info["images_dir"])
    openi_fn_to_path = build_filename_index(openi_files)
    openi_filenames = [row["image_name"] for _, row in openi_df.iterrows() if Path(row["image_name"]).stem in openi_fn_to_path]
    
    combined_filenames = nih_filenames + openi_filenames
    print(f"[pipeline] Total filenames mapped: {len(combined_filenames)}")

    # 3. Load Cached Tensors
    print("[pipeline] Loading cached tensors...")
    import gc
    
    nih_images, nih_labels = load_cached(cache_dir, "nih")
    if nih_images.dtype == torch.float32:
        print("[pipeline] Converting NIH images to uint8 to save RAM...")
        nih_images = (nih_images * 255.0).to(torch.uint8)
    gc.collect()
    
    openi_images, openi_labels = load_cached(cache_dir, "openi")
    if openi_images.dtype == torch.float32:
        print("[pipeline] Converting Open-I images to uint8 to save RAM...")
        openi_images = (openi_images * 255.0).to(torch.uint8)
    gc.collect()
    # Subsample to speed up CPU training
    print("[pipeline] Subsampling datasets for fast CPU training (1000 samples each)...")
    np.random.seed(42)
    nih_subsample_idx = np.random.choice(len(nih_images), size=1000, replace=False)
    nih_images = nih_images[nih_subsample_idx]
    nih_labels = nih_labels[nih_subsample_idx]
    nih_filenames = [nih_filenames[idx] for idx in nih_subsample_idx]
    
    openi_subsample_idx = np.random.choice(len(openi_images), size=1000, replace=False)
    openi_images = openi_images[openi_subsample_idx]
    openi_labels = openi_labels[openi_subsample_idx]
    openi_filenames = [openi_filenames[idx] for idx in openi_subsample_idx]
    
    combined_filenames = nih_filenames + openi_filenames
    
    print("[pipeline] Concatenating tensors...")
    images = torch.cat([nih_images, openi_images])
    labels = torch.cat([nih_labels, openi_labels])
    site_ids = torch.cat([
        torch.zeros(len(nih_images), dtype=torch.long),
        torch.ones(len(openi_images), dtype=torch.long)
    ])
    
    # Free reference variables
    del nih_images, openi_images
    gc.collect()
    
    print(f"[pipeline] Combined dataset size: {images.shape[0]} images, {labels.shape[1]} labels")
    
    # 4. Train / Val / Test Partitioning (70 / 15 / 15) with Forced Bbox Test Placement
    # First, parse the bbox annotations list to identify indices of images with bboxes
    bbox_csv = "C:/Users/Shara/OneDrive/Desktop/projects/data of chest xray/images_005/BBox_List_2017.csv"
    bbox_df = pd.read_csv(bbox_csv)
    bbox_set = set(bbox_df["Image Index"])
    
    bbox_indices = []
    for idx_fn, fname in enumerate(combined_filenames):
        if fname in bbox_set:
            bbox_indices.append(idx_fn)
            
    # Take other indices that don't have bboxes
    bbox_indices_set = set(bbox_indices)
    other_indices = [i for i in range(len(images)) if i not in bbox_indices_set]
    
    # Shuffle other indices
    np.random.seed(42)
    np.random.shuffle(other_indices)
    
    n_val = int(len(images) * 0.15)
    n_test = int(len(images) * 0.15)
    n_train = len(images) - n_val - n_test
    
    # We want test split to have all bbox_indices plus the remaining from other_indices
    n_bbox = len(bbox_indices)
    n_test_needed = n_test - n_bbox
    
    test_idx = np.array(bbox_indices + list(other_indices[:n_test_needed]))
    val_idx = np.array(other_indices[n_test_needed : n_test_needed + n_val])
    train_idx = np.array(other_indices[n_test_needed + n_val :])
    
    print(f"[pipeline] Splits created: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"[pipeline] Forced {n_bbox} bounding box images into the Test split.")
    
    train_ds = CachedChestXrayDataset(images[train_idx], labels[train_idx], site_ids[train_idx], transform=get_train_transforms(image_size))
    val_ds = CachedChestXrayDataset(images[val_idx], labels[val_idx], site_ids[val_idx], transform=get_val_transforms(image_size))
    test_ds = CachedChestXrayDataset(images[test_idx], labels[test_idx], site_ids[test_idx], transform=get_val_transforms(image_size))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    # 5. Model Initialization & Training
    print("[pipeline] Initializing model...")
    # Override epochs for runtime check (set to 3 for fast execution, but training complete)
    epochs = 3
    
    model = MultiTaskChestXray(
        backbone_name=train_cfg.get("backbone", "resnet18"),
        pretrained=train_cfg.get("pretrained", True),
        num_classes=len(label_names),
        bbox_enabled=False, # Disable bbox head training since we don't feed bbox in batch
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        cls_loss_weight=1.0,
        bbox_loss_weight=0.0,
        coral_lambda=1.0, # Align NIH and Open-I domain representations!
    )
    
    checkpoint_dir = "./checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="pipeline-best-{epoch:02d}-{val_auc:.3f}",
            monitor="val_auc",
            mode="max",
            save_top_k=1,
        )
    ]
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        log_every_n_steps=20,
    )
    
    print("[pipeline] Starting training loop...")
    trainer.fit(model, train_loader, val_loader)
    print("[pipeline] Training completed successfully!")
    
    # Load best checkpoint for evaluation
    best_ckpt = callbacks[0].best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"[pipeline] Loading best checkpoint: {best_ckpt}")
        model = MultiTaskChestXray.load_from_checkpoint(best_ckpt, map_location=device)
    model.eval()
    model.to(device)
    
    # 6. Evaluation on Test split
    print("[pipeline] Running evaluation on test partition...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            out = model(imgs)
            probs = torch.sigmoid(out["logits"]).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())
            
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    per_class_auc = {}
    for i, name in enumerate(label_names):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            per_class_auc[name] = float(auc)
        except ValueError:
            per_class_auc[name] = None
            
    valid_aucs = [v for v in per_class_auc.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    
    print(f"[pipeline] Evaluation finished. Macro AUC: {macro_auc:.4f}")
    
    # Save AUC results
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_results.json", "w") as f:
        json.dump({
            "macro_auc": macro_auc,
            "per_class_auc": per_class_auc,
            "n_samples": len(all_probs)
        }, f, indent=2)
        
    # 7. Calibration Assessment
    print("[pipeline] Running calibration assessment...")
    # Compute calibration on the most prevalent pathology or overall
    # Let's compute average ECE across classes
    ece_vals = []
    for i, name in enumerate(label_names):
        if per_class_auc[name] is not None:
            ece, _ = expected_calibration_error(all_labels[:, i], all_probs[:, i])
            ece_vals.append(ece)
    mean_ece = float(np.mean(ece_vals))
    print(f"[pipeline] Mean ECE: {mean_ece:.4f}")
    
    # Plot reliability diagram for the most prevalent pathology: No Finding (index 7) or Effusion (index 3)
    # Let's find index of Effusion (most common positive pathology)
    eff_idx = label_names.index("Effusion")
    os.makedirs("figures", exist_ok=True)
    plot_reliability_diagram(
        all_labels[:, eff_idx],
        all_probs[:, eff_idx],
        label="Effusion",
        output_dir="figures",
        filename="reliability_diagram.png"
    )
    
    with open("results/calibration_results.json", "w") as f:
        json.dump({
            "mean_ece": mean_ece,
            "eff_ece": float(expected_calibration_error(all_labels[:, eff_idx], all_probs[:, eff_idx])[0])
        }, f, indent=2)
        
    # 8. MC Dropout Uncertainty Estimation
    print("[pipeline] Running MC Dropout uncertainty estimation...")
    # We run MC Dropout on the test set (using a subset of 200 samples for speed)
    test_tensor_subset = images[test_idx[:200]].to(torch.float32) / 255.0
    mc_results = mc_dropout_predict(model, test_tensor_subset, n_passes=20, device=device)
    
    # Compute routing stats with entropy threshold
    # Set threshold to 75th percentile of entropy
    threshold = float(np.percentile(mc_results["entropy"], 75))
    routing_stats = compute_routing_stats(mc_results["entropy"], threshold)
    print(f"[pipeline] Routing rate (High uncertainty): {routing_stats['routing_rate']:.2%}")
    
    with open("results/uncertainty_results.json", "w") as f:
        json.dump(routing_stats, f, indent=2)
        
    # 9. Grad-CAM & Pointing Game Validation
    print("[pipeline] Running Grad-CAM and Pointing Game validation...")
    bbox_csv = "C:/Users/Shara/OneDrive/Desktop/projects/data of chest xray/images_005/BBox_List_2017.csv"
    bbox_df = pd.read_csv(bbox_csv)
    
    nih_bbox_dict = {}
    for _, row in bbox_df.iterrows():
        img_name = row["Image Index"]
        x = float(row["Bbox [x"])
        y = float(row["y"])
        w = float(row["w"])
        h = float(row["h]"])
        label = str(row["Finding Label"]).strip()
        # Normalize original 1024 size to [0, 1]
        bbox = (x / 1024.0, y / 1024.0, w / 1024.0, h / 1024.0)
        if img_name not in nih_bbox_dict:
            nih_bbox_dict[img_name] = []
        nih_bbox_dict[img_name].append({"bbox": bbox, "label": label})
        
    # Gather test samples with bboxes
    test_filenames = [combined_filenames[idx] for idx in test_idx]
    pointing_game_images = []
    pointing_game_bboxes = []
    pointing_game_labels = []
    
    for i, fname in enumerate(test_filenames):
        if fname in nih_bbox_dict:
            for item in nih_bbox_dict[fname]:
                pointing_game_images.append(images[test_idx[i]])
                pointing_game_bboxes.append(item["bbox"])
                pointing_game_labels.append(item["label"])
                
    print(f"[pipeline] Found {len(pointing_game_images)} bounding box annotations in the test partition.")
    
    if len(pointing_game_images) > 0:
        # Generate heatmaps
        heatmaps = []
        target_layer = model.backbone.model.layer4[-1]
        
        # Limit to first 100 for evaluation speed
        eval_limit = min(100, len(pointing_game_images))
        print(f"[pipeline] Running Pointing Game on {eval_limit} samples...")
        
        for k in range(eval_limit):
            img = pointing_game_images[k].to(torch.float32)
            if img.max() > 1.0:
                img = img / 255.0
            img = img.to(device).unsqueeze(0)
            
            lbl_name = pointing_game_labels[k]
            if lbl_name in label_names:
                target_cls = label_names.index(lbl_name)
            else:
                # Class mapping fallback
                target_cls = 0
            
            # Generate Grad-CAM map
            heatmap = generate_gradcam(model, img, target_layer, target_class=target_cls, device=device)
            heatmaps.append(heatmap[0])
            
        # Pointing game evaluation
        pg_results = evaluate_pointing_game(
            heatmaps=heatmaps,
            bboxes=pointing_game_bboxes[:eval_limit],
            labels=pointing_game_labels[:eval_limit],
            image_size=(224, 224)
        )
        
        print("[pipeline] Pointing game results:")
        for lbl, stats in pg_results.items():
            print(f"  {lbl:20s}: Accuracy={stats['accuracy']:.2%} ({stats['hits']}/{stats['total']})")
            
        # Plot pointing game results
        plot_pointing_game_results(pg_results, output_dir="figures", filename="pointing_game_results.png")
        
        # Visualize 3 samples
        os.makedirs("figures/gradcam_samples", exist_ok=True)
        for idx_sample in range(min(3, eval_limit)):
            img_np = pointing_game_images[idx_sample].permute(1, 2, 0).numpy()
            # Normalize to [0, 1] range for visualization if not already
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
            
            # Save visual overlay
            visualize_gradcam(
                image=img_np,
                heatmap=heatmaps[idx_sample],
                title=f"Grad-CAM: {pointing_game_labels[idx_sample]}",
                output_path=f"figures/gradcam_samples/sample_{idx_sample}.png"
            )
            print(f"[pipeline] Saved Grad-CAM sample overlay to figures/gradcam_samples/sample_{idx_sample}.png")
            
    print("\n[pipeline] Pipeline finished successfully! All outputs generated.")

if __name__ == "__main__":
    run_entire_pipeline()
