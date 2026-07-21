"""
bootstrap_ci.py — Non-parametric Bootstrap Confidence Interval Engine.

Computes 95% CIs (B=1000 resamples) for all RadAgent and classification metrics,
turning point estimates into statistically rigorous intervals.

Usage:
    python -m eval.bootstrap_ci --results results/report_eval_results.json
    python -m eval.bootstrap_ci --eval-results results/evaluation_results.json
"""

import json
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent.absolute()))


# ── Bootstrap Core ───────────────────────────────────────────────────────────

def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a given statistic.
    
    Args:
        data: 1D numpy array of observations.
        statistic_fn: Function mapping an array to a scalar statistic.
        n_bootstrap: Number of bootstrap resamples (default=1000).
        confidence_level: CI level (default=0.95 for 95% CI).
        seed: Random seed for reproducibility.
        
    Returns:
        Dict with keys: point_estimate, ci_lower, ci_upper, std_error, n_samples.
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    
    if n == 0:
        return {
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "std_error": 0.0,
            "n_samples": 0
        }
    
    point_estimate = float(statistic_fn(data))
    
    # Generate bootstrap distribution
    bootstrap_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        resample_indices = rng.randint(0, n, size=n)
        resample = data[resample_indices]
        bootstrap_stats[b] = statistic_fn(resample)
    
    # Percentile method CI
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    std_error = float(np.std(bootstrap_stats, ddof=1))
    
    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "n_samples": n
    }


def format_ci(result: Dict[str, float], as_percentage: bool = False) -> str:
    """Format a CI result as a readable string."""
    mult = 100.0 if as_percentage else 1.0
    suffix = "%" if as_percentage else ""
    return (
        f"{result['point_estimate'] * mult:.2f}{suffix} "
        f"[{result['ci_lower'] * mult:.2f}, {result['ci_upper'] * mult:.2f}]{suffix} "
        f"(SE={result['std_error'] * mult:.3f}{suffix}, n={result['n_samples']})"
    )


# ── Agent Pipeline Metrics with CIs ─────────────────────────────────────────

def compute_agent_metrics_with_ci(
    runs: List[Dict[str, Any]],
    n_bootstrap: int = 1000
) -> Dict[str, Dict[str, float]]:
    """Compute all agent pipeline metrics with bootstrap 95% CIs.
    
    Args:
        runs: List of pipeline execution result dicts.
        n_bootstrap: Number of bootstrap resamples.
        
    Returns:
        Dict mapping metric_name -> {point_estimate, ci_lower, ci_upper, std_error, n_samples}.
    """
    n = len(runs)
    if n == 0:
        return {}
    
    # Extract per-case binary/count arrays
    hallucination_counts = np.array([
        len(r.get("verification", {}).get("hallucinations", []))
        for r in runs
    ], dtype=float)
    
    omission_counts = np.array([
        len(r.get("verification", {}).get("omissions", []))
        for r in runs
    ], dtype=float)
    
    escalated_flags = np.array([
        1.0 if r.get("escalated", False) else 0.0
        for r in runs
    ], dtype=float)
    
    # Revision success: did verification discrepancies decrease after revision?
    revision_success = []
    for r in runs:
        steps = r.get("steps", [])
        verify_steps = [s for s in steps if s.get("node") == "verification"]
        if len(verify_steps) >= 2:
            initial = verify_steps[0].get("discrepancy_count", 0)
            final = verify_steps[1].get("discrepancy_count", 0)
            if initial > 0:
                revision_success.append(1.0 if final < initial else 0.0)
    
    revision_success_arr = np.array(revision_success, dtype=float) if revision_success else np.array([0.0])
    
    # Total discrepancies per case
    total_discrepancies = hallucination_counts + omission_counts
    
    results = {}
    
    results["hallucination_rate"] = bootstrap_ci(
        hallucination_counts, np.mean, n_bootstrap
    )
    results["omission_rate"] = bootstrap_ci(
        omission_counts, np.mean, n_bootstrap
    )
    results["escalation_rate"] = bootstrap_ci(
        escalated_flags, np.mean, n_bootstrap
    )
    results["mean_discrepancy_count"] = bootstrap_ci(
        total_discrepancies, np.mean, n_bootstrap
    )
    results["revision_correction_rate"] = bootstrap_ci(
        revision_success_arr, np.mean, n_bootstrap
    )
    
    return results


# ── Classification Metrics with CIs ─────────────────────────────────────────

def compute_classification_ci(
    per_class_aucs: List[float],
    n_bootstrap: int = 1000
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap CI for macro AUC from per-class AUC values.
    
    Args:
        per_class_aucs: List of AUC values, one per class.
        n_bootstrap: Number of bootstrap resamples.
        
    Returns:
        Dict with 'macro_auc' CI.
    """
    aucs = np.array(per_class_aucs, dtype=float)
    return {
        "macro_auc": bootstrap_ci(aucs, np.mean, n_bootstrap)
    }


def compute_calibration_ci(
    confidences: List[float],
    accuracies: List[float],
    n_bootstrap: int = 1000
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap CI for Expected Calibration Error (ECE).
    
    Args:
        confidences: Bin-center confidence values.
        accuracies: Corresponding bin accuracies.
        n_bootstrap: Number of bootstrap resamples.
        
    Returns:
        Dict with 'ece' CI.
    """
    confs = np.array(confidences, dtype=float)
    accs = np.array(accuracies, dtype=float)
    
    # ECE = mean absolute difference between confidence and accuracy per bin
    ece_per_bin = np.abs(confs - accs)
    
    return {
        "ece": bootstrap_ci(ece_per_bin, np.mean, n_bootstrap)
    }


# ── CLI Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI calculator for pipeline metrics")
    parser.add_argument("--results", type=str, default="results/report_eval_results.json",
                        help="Path to agent pipeline evaluation results JSON")
    parser.add_argument("--eval-results", type=str, default="results/evaluation_results.json",
                        help="Path to classification evaluation results JSON")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap resamples")
    parser.add_argument("--output", type=str, default="results/bootstrap_ci_results.json",
                        help="Output path for CI results")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Bootstrap 95% Confidence Interval Analysis")
    print("=" * 70)
    
    output = {}
    
    # ── Agent Pipeline CIs ──
    if os.path.exists(args.results):
        with open(args.results, "r") as f:
            report_data = json.load(f)
        
        runs = report_data.get("runs", [])
        print(f"\nAgent Pipeline: {len(runs)} cases loaded")
        
        agent_cis = compute_agent_metrics_with_ci(runs, args.n_bootstrap)
        output["agent_pipeline"] = agent_cis
        
        print("\n  Agent Pipeline Metrics (95% CI):")
        for metric_name, ci_result in agent_cis.items():
            pct = metric_name.endswith("_rate")
            print(f"    {metric_name}: {format_ci(ci_result, as_percentage=pct)}")
    else:
        print(f"\n[SKIP] Agent results not found at {args.results}")
        # Generate synthetic demonstration data
        print("[DEMO] Generating synthetic pipeline results for CI demonstration...")
        synthetic_runs = _generate_synthetic_runs(50)
        agent_cis = compute_agent_metrics_with_ci(synthetic_runs, args.n_bootstrap)
        output["agent_pipeline"] = agent_cis
        output["agent_pipeline_note"] = "Generated from synthetic demonstration data (N=50)"
        
        print("\n  Agent Pipeline Metrics — Synthetic Demo (95% CI):")
        for metric_name, ci_result in agent_cis.items():
            pct = metric_name.endswith("_rate")
            print(f"    {metric_name}: {format_ci(ci_result, as_percentage=pct)}")
    
    # ── Classification CIs ──
    if os.path.exists(args.eval_results):
        with open(args.eval_results, "r") as f:
            eval_data = json.load(f)
        
        per_class = eval_data.get("per_class_auc", {})
        if per_class:
            aucs = list(per_class.values())
            cls_cis = compute_classification_ci(aucs, args.n_bootstrap)
            output["classification"] = cls_cis
            
            print(f"\n  Classification Metrics (95% CI):")
            for metric_name, ci_result in cls_cis.items():
                print(f"    {metric_name}: {format_ci(ci_result)}")
    else:
        print(f"\n[SKIP] Classification results not found at {args.eval_results}")
        # Synthetic classification AUCs for demonstration
        synthetic_aucs = [0.62, 0.71, 0.53, 0.68, 0.49, 0.55, 0.58, 0.67]
        cls_cis = compute_classification_ci(synthetic_aucs, args.n_bootstrap)
        output["classification"] = cls_cis
        output["classification_note"] = "Generated from synthetic per-class AUC values"
        
        print(f"\n  Classification Metrics — Synthetic Demo (95% CI):")
        for metric_name, ci_result in cls_cis.items():
            print(f"    {metric_name}: {format_ci(ci_result)}")
    
    # ── Save ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to {args.output}")
    print(f"{'=' * 70}")


def _generate_synthetic_runs(n: int = 50) -> List[Dict[str, Any]]:
    """Generate synthetic pipeline runs for CI demonstration when real data is unavailable."""
    rng = np.random.RandomState(42)
    runs = []
    for i in range(n):
        n_hallucinations = int(rng.choice([0, 0, 0, 1, 1, 2]))
        n_omissions = int(rng.choice([0, 0, 1, 1, 2]))
        escalated = bool(rng.random() < 0.15)
        
        hallucinations = [{"claim": f"synthetic_claim_{j}", "explanation": "test"} 
                         for j in range(n_hallucinations)]
        omissions = [{"finding": f"synthetic_finding_{j}", "explanation": "test"} 
                    for j in range(n_omissions)]
        
        initial_disc = n_hallucinations + n_omissions
        final_disc = max(0, initial_disc - int(rng.choice([0, 1, 1, 2])))
        
        steps = [
            {"node": "drafting", "revision": 0},
            {"node": "verification", "discrepancy_count": initial_disc},
        ]
        if initial_disc > 0:
            steps.extend([
                {"node": "drafting", "revision": 1},
                {"node": "verification", "discrepancy_count": final_disc},
            ])
        
        runs.append({
            "verification": {"hallucinations": hallucinations, "omissions": omissions},
            "escalated": escalated,
            "steps": steps,
            "final_report": f"Synthetic report {i}",
            "trace_id": f"syn-{i:04d}"
        })
    return runs


if __name__ == "__main__":
    main()
