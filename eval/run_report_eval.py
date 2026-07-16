"""
run_report_eval.py — Standalone report evaluation harness.
Runs model predictions + MC Dropout, feeds them through RadAgent, and evaluates performance.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path

# Add root folder to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from src.device import get_device
from src.models.multitask_head import MultiTaskChestXray
from src.data.cache_tensors import load_cached
from src.models.mc_dropout import mc_dropout_predict
from radagent import run_radagent_pipeline, LLMClient
from eval.metrics import compute_report_metrics

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Effusion",
    "Pneumonia", "Pneumothorax", "Nodule/Mass", "No Finding",
]

def preprocess_image(img_tensor: torch.Tensor) -> torch.Tensor:
    """Preprocesses a single cached image tensor for model inference."""
    # Convert to float32 in [0, 1] if uint8
    if img_tensor.dtype == torch.uint8:
        img_tensor = img_tensor.to(torch.float32) / 255.0
        
    # Standard ImageNet normalization: (x - mean) / std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    return img_normalized

def run_evaluation(num_samples_per_site: int = 5):
    print("======================================================================")
    print("RadAgent Report Evaluation Harness")
    print("======================================================================")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Load Model Checkpoint
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory '{checkpoint_dir}' does not exist.")
        return
        
    ckpts = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not ckpts:
        print(f"Error: No checkpoints found in '{checkpoint_dir}'. Please run training first.")
        return
        
    def get_auc(p):
        import re
        match = re.search(r"val_auc=([0-9]+(?:\.[0-9]+)?)", p)
        return float(match.group(1)) if match else 0.0
        
    ckpts.sort(key=get_auc)
    best_ckpt = ckpts[-1]
    print(f"Loading best checkpoint: {best_ckpt} (Val AUC: {get_auc(best_ckpt):.3f})")
    
    model = MultiTaskChestXray.load_from_checkpoint(best_ckpt, map_location=device)
    model.eval()
    
    # Initialize LLM Client
    llm_client = LLMClient()
    print(f"Agent LLM provider configured: {llm_client.provider.upper()}")
    if llm_client.provider == "anthropic" and not llm_client.api_key:
        print("WARNING: Anthropic key missing, using Mock LLM fallback.")
    
    # Load thresholds
    CONFIDENCE_THRESHOLD = 0.5
    ROUTING_THRESHOLD = 4.94
    if os.path.exists("results/uncertainty_results.json"):
        try:
            with open("results/uncertainty_results.json", "r") as f:
                ur_data = json.load(f)
                ROUTING_THRESHOLD = ur_data.get("threshold", 4.94)
        except Exception:
            pass
    print(f"Classification settings: CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD}, ROUTING_THRESHOLD={ROUTING_THRESHOLD}")
    
    # 2. Load Tensors
    cache_dir = "./cached_tensors"
    print("Loading cached dataset tensors...")
    try:
        nih_images, nih_labels = load_cached(cache_dir, "nih")
        openi_images, openi_labels = load_cached(cache_dir, "openi")
    except Exception as e:
        print(f"Error loading cached tensors: {e}")
        return
        
    # Sample subset for evaluation
    np.random.seed(42)
    nih_indices = np.random.choice(len(nih_images), size=num_samples_per_site, replace=False)
    openi_indices = np.random.choice(len(openi_images), size=num_samples_per_site, replace=False)
    
    # Combine results
    all_runs = []
    
    sites = [("NIH", nih_images, nih_labels, nih_indices), 
             ("Open-I", openi_images, openi_labels, openi_indices)]
             
    print("\nRunning model inference + report generation...")
    for site_name, images_pt, labels_pt, indices in sites:
        print(f"\n--- Site: {site_name} ({len(indices)} samples) ---")
        for idx in indices:
            img = images_pt[idx]
            label = labels_pt[idx]
            
            # Preprocess and normalize
            img_processed = preprocess_image(img).unsqueeze(0).to(device) # Add batch dim
            
            # MC Dropout
            mc_res = mc_dropout_predict(model, img_processed, n_passes=20, device=device)
            mean_probs = mc_res["mean_probs"][0]
            entropy = float(mc_res["entropy"][0])
            
            # Construct predictions structure
            predictions_list = []
            for i, name in enumerate(LABEL_NAMES):
                prob = float(mean_probs[i])
                predictions_list.append({
                    "label": name,
                    "probability": prob,
                    "positive": prob > CONFIDENCE_THRESHOLD,
                })
                
            prediction_response = {
                "predictions": predictions_list,
                "uncertainty": entropy,
                "needs_human_review": entropy > ROUTING_THRESHOLD,
                "gradcam_available": False,
            }
            
            # Execute RadAgent state machine pipeline
            pipeline_result = run_radagent_pipeline(prediction_response, llm_client)
            
            # Append metadata
            pipeline_result["site"] = site_name
            pipeline_result["index"] = int(idx)
            pipeline_result["ground_truth"] = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES)) if label[i] > 0.5]
            
            all_runs.append(pipeline_result)
            
            # Quick summary printed to console
            print(f"Sample {idx} (GT: {pipeline_result['ground_truth']}): "
                  f"Escalated={pipeline_result['escalated']}, "
                  f"Discrepancies={pipeline_result.get('discrepancy_count', 0)}")
                  
    # 3. Calculate Metrics
    print("\nCalculating metrics...")
    nih_runs = [r for r in all_runs if r["site"] == "NIH"]
    openi_runs = [r for r in all_runs if r["site"] == "Open-I"]
    
    nih_metrics = compute_report_metrics(nih_runs)
    openi_metrics = compute_report_metrics(openi_runs)
    combined_metrics = compute_report_metrics(all_runs)
    
    # 4. Save results
    os.makedirs("results", exist_ok=True)
    
    # Detailed results
    report_eval_path = "results/report_eval_results.json"
    with open(report_eval_path, "w") as f:
        # Simplify runs for logging (remove full numpy values if any)
        log_runs = []
        for r in all_runs:
            log_runs.append({
                "site": r["site"],
                "index": r["index"],
                "ground_truth": r["ground_truth"],
                "final_report": r["final_report"],
                "verification": r["verification"],
                "escalated": r["escalated"],
                "trace_id": r["trace_id"],
                "steps": r["steps"]
            })
        json.dump({
            "metrics": combined_metrics,
            "runs": log_runs
        }, f, indent=2)
        
    # Cross-domain gap results
    domain_gap_path = "results/cross_domain_report_gap.json"
    with open(domain_gap_path, "w") as f:
        json.dump({
            "nih_metrics": nih_metrics,
            "openi_metrics": openi_metrics,
            "discrepancy_difference": {
                "hallucination_gap": nih_metrics["hallucination_rate"] - openi_metrics["hallucination_rate"],
                "omission_gap": nih_metrics["omission_rate"] - openi_metrics["omission_rate"],
                "escalation_gap": nih_metrics["escalation_rate"] - openi_metrics["escalation_rate"]
            }
        }, f, indent=2)
        
    print("\n======================================================================")
    print("Report Evaluation Summary")
    print("======================================================================")
    print(f"Total Cases Analyzed: {combined_metrics['total_cases']}")
    print(f"Combined Hallucination Rate: {combined_metrics['hallucination_rate']:.3f}")
    print(f"Combined Omission Rate:      {combined_metrics['omission_rate']:.3f}")
    print(f"Revision Loop Success Rate:  {combined_metrics['revision_correction_rate']:.2%}")
    print(f"Human Escalation Rate:       {combined_metrics['escalation_rate']:.2%}")
    print("----------------------------------------------------------------------")
    print("Site Breakdown:")
    print(f"  NIH (In-Distribution):")
    print(f"    Hallucination Rate: {nih_metrics['hallucination_rate']:.3f}")
    print(f"    Omission Rate:      {nih_metrics['omission_rate']:.3f}")
    print(f"    Escalation Rate:    {nih_metrics['escalation_rate']:.2%}")
    print(f"  Open-I (Out-of-Distribution):")
    print(f"    Hallucination Rate: {openi_metrics['hallucination_rate']:.3f}")
    print(f"    Omission Rate:      {openi_metrics['omission_rate']:.3f}")
    print(f"    Escalation Rate:    {openi_metrics['escalation_rate']:.2%}")
    print("======================================================================")
    print(f"Saved evaluation results to {report_eval_path}")
    print(f"Saved cross-domain report metrics to {domain_gap_path}")

if __name__ == "__main__":
    # You can customize sample counts as script arguments
    num_samples = 5
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            pass
            
    run_evaluation(num_samples)
