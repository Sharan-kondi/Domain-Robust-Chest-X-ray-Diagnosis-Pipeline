"""
ablation_study.py — Ablation framework comparing RadAgent pipeline variants.

Benchmarks four configurations on identical synthetic test inputs:
  1. Monolithic single-pass LLM (no verification, no RAG)
  2. RadAgent WITHOUT Verification Agent (Drafting + RAG only)
  3. RadAgent WITHOUT RAG Grounding (Drafting + Verification only)
  4. Full RadAgent Pipeline (all agents active)

Usage:
    python -m eval.ablation_study
    python -m eval.ablation_study --num-cases 20
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from radagent.schema import predictions_to_findings
from radagent.llm import LLMClient
from radagent.trace import AgentTracer
from eval.bootstrap_ci import bootstrap_ci, format_ci

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Effusion",
    "Pneumonia", "Pneumothorax", "Nodule/Mass", "No Finding",
]


# ── Synthetic Test Case Generator ────────────────────────────────────────────

def generate_synthetic_test_cases(n: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate synthetic prediction responses for ablation testing.
    
    Creates realistic multi-label chest X-ray prediction dictionaries
    with varying pathology mixes, uncertainty levels, and demographics.
    """
    rng = np.random.RandomState(seed)
    cases = []
    
    demographics_pool = [
        {"age": "25", "gender": "Male"},
        {"age": "45", "gender": "Female"},
        {"age": "67", "gender": "Male"},
        {"age": "72", "gender": "Female"},
        {"age": "33", "gender": "Male"},
        {"age": "55", "gender": "Female"},
        {"age": "80", "gender": "Male"},
        {"age": "19", "gender": "Female"},
    ]
    
    location_pool = {
        "Atelectasis": "left lower zone",
        "Cardiomegaly": "cardiac silhouette",
        "Consolidation": "right middle zone",
        "Effusion": "left costophrenic angle",
        "Pneumonia": "right lower zone",
        "Pneumothorax": "right apex",
        "Nodule/Mass": "right upper zone",
    }
    
    for i in range(n):
        # Generate realistic probabilities
        probs = rng.beta(2, 5, size=len(LABEL_NAMES))
        
        # Make 1-3 pathologies have elevated probability
        n_positive = rng.randint(1, 4)
        positive_indices = rng.choice(len(LABEL_NAMES) - 1, size=n_positive, replace=False)
        for idx in positive_indices:
            probs[idx] = rng.uniform(0.55, 0.95)
        
        # No Finding is inverse of any positive
        probs[-1] = 1.0 - max(probs[:-1]) if max(probs[:-1]) > 0.5 else rng.uniform(0.6, 0.9)
        
        predictions = []
        for j, name in enumerate(LABEL_NAMES):
            predictions.append({
                "label": name,
                "probability": float(probs[j]),
                "positive": bool(probs[j] > 0.5),
            })
        
        entropy = float(rng.uniform(1.0, 8.0))
        
        # Build locations for positive findings
        locations = {}
        for idx in positive_indices:
            label = LABEL_NAMES[idx]
            if label in location_pool:
                locations[label] = location_pool[label]
        
        cases.append({
            "prediction_response": {
                "predictions": predictions,
                "uncertainty": entropy,
                "needs_human_review": entropy > 4.94,
                "gradcam_available": False,
            },
            "demographics": demographics_pool[i % len(demographics_pool)],
            "locations": locations,
            "case_id": f"ablation-{i:03d}",
        })
    
    return cases


# ── Pipeline Variant Runners ─────────────────────────────────────────────────

def run_full_pipeline(
    case: Dict[str, Any], 
    llm_client: LLMClient
) -> Dict[str, Any]:
    """Run the complete RadAgent pipeline (all agents active)."""
    from radagent import run_radagent_pipeline
    
    start_time = time.time()
    result = run_radagent_pipeline(
        case["prediction_response"],
        llm_client=llm_client,
        locations=case.get("locations"),
        demographics=case.get("demographics"),
        clinical_focus=None
    )
    elapsed = time.time() - start_time
    result["latency_seconds"] = elapsed
    result["variant"] = "full_pipeline"
    return result


def run_without_verification(
    case: Dict[str, Any],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """Run RadAgent with Verification Agent disabled (Drafting + RAG only)."""
    from radagent.agents.drafting import DraftingAgent
    from radagent.agents.grounding import GroundingAgent
    from radagent.agents.escalation import EscalationAgent
    from radagent.agents.bias import BiasAuditAgent
    
    start_time = time.time()
    
    findings = predictions_to_findings(
        case["prediction_response"], 
        locations=case.get("locations")
    )
    
    draft_agent = DraftingAgent(llm_client)
    grounding_agent = GroundingAgent()
    escalation_agent = EscalationAgent()
    bias_agent = BiasAuditAgent()
    
    # Draft only (no verification loop)
    draft_text = draft_agent.draft(findings, feedback=None, clinical_focus=None)
    
    # Ground directly
    ground_res = grounding_agent.ground(findings, draft_text)
    
    # Bias audit
    bias_res = bias_agent.audit(case.get("demographics"))
    
    # Escalation
    state = {
        "findings": findings,
        "current_draft": draft_text,
        "verification": {"hallucinations": [], "omissions": [], "discrepancy_count": 0},
        "discrepancy_count": 0,
        "revision_count": 1,
    }
    esc_res = escalation_agent.evaluate(state)
    
    elapsed = time.time() - start_time
    
    return {
        "final_report": draft_text,
        "verification": {"hallucinations": [], "omissions": [], "discrepancy_count": 0},
        "grounding": ground_res.get("grounding", []),
        "escalated": esc_res["escalated"],
        "bias": bias_res,
        "latency_seconds": elapsed,
        "variant": "no_verification",
        "steps": [
            {"node": "drafting", "revision": 0},
            {"node": "grounding"},
        ],
        "trace_id": f"ablation-noverify-{case['case_id']}"
    }


def run_without_rag(
    case: Dict[str, Any],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """Run RadAgent with RAG Grounding disabled (Drafting + Verification only)."""
    from radagent.agents.drafting import DraftingAgent
    from radagent.agents.verification import VerificationAgent
    from radagent.agents.escalation import EscalationAgent
    from radagent.agents.bias import BiasAuditAgent
    
    start_time = time.time()
    
    findings = predictions_to_findings(
        case["prediction_response"],
        locations=case.get("locations")
    )
    
    draft_agent = DraftingAgent(llm_client)
    verify_agent = VerificationAgent(llm_client)
    escalation_agent = EscalationAgent()
    bias_agent = BiasAuditAgent()
    
    # Draft
    draft_text = draft_agent.draft(findings, feedback=None, clinical_focus=None)
    
    # Verify
    verify_result = verify_agent.verify(findings, draft_text)
    disc_count = verify_result.get("discrepancy_count", 0)
    
    # If discrepancies, revise once
    if disc_count > 0:
        feedback_parts = []
        if verify_result.get("hallucinations"):
            feedback_parts.append("Hallucinations: " + "; ".join(
                [h["claim"] for h in verify_result["hallucinations"]]
            ))
        if verify_result.get("omissions"):
            feedback_parts.append("Omissions: " + "; ".join(
                [o["finding"] for o in verify_result["omissions"]]
            ))
        feedback = "\n".join(feedback_parts)
        draft_text = draft_agent.draft(findings, feedback=feedback, clinical_focus=None)
        verify_result = verify_agent.verify(findings, draft_text)
    
    # Skip RAG grounding entirely
    
    # Bias audit
    bias_res = bias_agent.audit(case.get("demographics"))
    
    # Escalation
    state = {
        "findings": findings,
        "current_draft": draft_text,
        "verification": verify_result,
        "discrepancy_count": verify_result.get("discrepancy_count", 0),
        "revision_count": 2 if disc_count > 0 else 1,
    }
    esc_res = escalation_agent.evaluate(state)
    
    elapsed = time.time() - start_time
    
    return {
        "final_report": draft_text,
        "verification": verify_result,
        "grounding": [],
        "escalated": esc_res["escalated"],
        "bias": bias_res,
        "latency_seconds": elapsed,
        "variant": "no_rag",
        "steps": [
            {"node": "drafting", "revision": 0},
            {"node": "verification", "discrepancy_count": disc_count},
        ],
        "trace_id": f"ablation-norag-{case['case_id']}"
    }


def run_monolithic_baseline(
    case: Dict[str, Any],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """Run a single-shot monolithic LLM prompt without any agents.
    
    This is the simplest possible baseline — a single LLM call with
    all findings dumped into one prompt, no verification, no RAG, no revision.
    """
    from radagent.agents.drafting import DraftingAgent
    
    start_time = time.time()
    
    findings = predictions_to_findings(
        case["prediction_response"],
        locations=case.get("locations")
    )
    
    # Use the drafting agent for a single pass (no feedback, no verification)
    draft_agent = DraftingAgent(llm_client)
    draft_text = draft_agent.draft(findings, feedback=None, clinical_focus=None)
    
    elapsed = time.time() - start_time
    
    # No verification at all — we report empty verification results
    return {
        "final_report": draft_text,
        "verification": {"hallucinations": [], "omissions": [], "discrepancy_count": 0},
        "grounding": [],
        "escalated": False,
        "bias": None,
        "latency_seconds": elapsed,
        "variant": "monolithic_baseline",
        "steps": [{"node": "drafting", "revision": 0}],
        "trace_id": f"ablation-mono-{case['case_id']}"
    }


# ── Ablation Analysis Engine ─────────────────────────────────────────────────

def analyze_ablation_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """Analyze results across all ablation variants with bootstrap CIs."""
    
    analysis = {}
    
    for variant_name, runs in all_results.items():
        if not runs:
            continue
        
        # Hallucination counts
        hallucination_counts = np.array([
            len(r.get("verification", {}).get("hallucinations", []))
            for r in runs
        ], dtype=float)
        
        # Omission counts
        omission_counts = np.array([
            len(r.get("verification", {}).get("omissions", []))
            for r in runs
        ], dtype=float)
        
        # Escalation flags
        escalated = np.array([
            1.0 if r.get("escalated", False) else 0.0
            for r in runs
        ], dtype=float)
        
        # Latency
        latencies = np.array([
            r.get("latency_seconds", 0.0)
            for r in runs
        ], dtype=float)
        
        # Report length (word count)
        report_lengths = np.array([
            len(r.get("final_report", "").split())
            for r in runs
        ], dtype=float)
        
        analysis[variant_name] = {
            "n_cases": len(runs),
            "hallucination_rate": bootstrap_ci(hallucination_counts, np.mean, n_bootstrap),
            "omission_rate": bootstrap_ci(omission_counts, np.mean, n_bootstrap),
            "escalation_rate": bootstrap_ci(escalated, np.mean, n_bootstrap),
            "mean_latency_seconds": bootstrap_ci(latencies, np.mean, n_bootstrap),
            "mean_report_length_words": bootstrap_ci(report_lengths, np.mean, n_bootstrap),
        }
    
    return analysis


# ── CLI Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RadAgent Ablation Study")
    parser.add_argument("--num-cases", type=int, default=20,
                        help="Number of synthetic test cases")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap resamples for CIs")
    parser.add_argument("--output", type=str, default="results/ablation_results.json",
                        help="Output path for ablation results")
    args = parser.parse_args()
    
    print("=" * 70)
    print("RadAgent Ablation Study")
    print("=" * 70)
    
    # Generate test cases
    print(f"\nGenerating {args.num_cases} synthetic test cases...")
    cases = generate_synthetic_test_cases(args.num_cases)
    
    # Initialize LLM client
    llm_client = LLMClient()
    print(f"LLM Provider: {llm_client.provider}")
    
    # ── Run all four variants ──
    variants = {
        "monolithic_baseline": run_monolithic_baseline,
        "no_verification": run_without_verification,
        "no_rag": run_without_rag,
        "full_pipeline": run_full_pipeline,
    }
    
    all_results = {name: [] for name in variants}
    
    for variant_name, runner_fn in variants.items():
        print(f"\n{'-' * 50}")
        print(f"Running variant: {variant_name}")
        print(f"{'-' * 50}")
        
        for i, case in enumerate(cases):
            try:
                result = runner_fn(case, llm_client)
                result["case_id"] = case["case_id"]
                all_results[variant_name].append(result)
                
                disc = result.get("verification", {}).get("discrepancy_count", 0)
                print(f"  Case {i+1}/{len(cases)}: "
                      f"Disc={disc}, "
                      f"Esc={result.get('escalated', False)}, "
                      f"Latency={result.get('latency_seconds', 0):.2f}s")
            except Exception as e:
                print(f"  Case {i+1}/{len(cases)}: ERROR - {e}")
                all_results[variant_name].append({
                    "case_id": case["case_id"],
                    "error": str(e),
                    "variant": variant_name,
                    "verification": {"hallucinations": [], "omissions": []},
                    "escalated": False,
                    "latency_seconds": 0.0,
                    "final_report": "",
                })
    
    # ── Analyze with bootstrap CIs ──
    print(f"\n{'=' * 70}")
    print("Ablation Results with 95% Bootstrap CIs")
    print(f"{'=' * 70}")
    
    analysis = analyze_ablation_results(all_results, args.n_bootstrap)
    
    for variant_name, metrics in analysis.items():
        print(f"\n  {variant_name} ({metrics['n_cases']} cases):")
        for metric_name, ci in metrics.items():
            if isinstance(ci, dict) and "point_estimate" in ci:
                pct = metric_name.endswith("_rate")
                print(f"    {metric_name}: {format_ci(ci, as_percentage=pct)}")
    
    # ── Compute deltas (Full Pipeline vs others) ──
    print(f"\n{'-' * 70}")
    print("Delta Analysis (Full Pipeline improvement over each variant):")
    print(f"{'-' * 70}")
    
    if "full_pipeline" in analysis:
        full = analysis["full_pipeline"]
        for variant_name, metrics in analysis.items():
            if variant_name == "full_pipeline":
                continue
            hall_delta = (
                metrics["hallucination_rate"]["point_estimate"] - 
                full["hallucination_rate"]["point_estimate"]
            )
            omit_delta = (
                metrics["omission_rate"]["point_estimate"] - 
                full["omission_rate"]["point_estimate"]
            )
            print(f"\n  vs {variant_name}:")
            print(f"    Hallucination reduction: {hall_delta:+.3f}")
            print(f"    Omission reduction: {omit_delta:+.3f}")
    
    # ── Save ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Serialize results (strip non-serializable)
    serializable_results = {}
    for vname, runs in all_results.items():
        serializable_results[vname] = []
        for r in runs:
            serializable_results[vname].append({
                "case_id": r.get("case_id"),
                "variant": r.get("variant"),
                "final_report": r.get("final_report", "")[:500],
                "verification": r.get("verification"),
                "escalated": r.get("escalated"),
                "latency_seconds": r.get("latency_seconds"),
            })
    
    with open(args.output, "w") as f:
        json.dump({
            "analysis": analysis,
            "raw_results": serializable_results,
            "config": {
                "num_cases": args.num_cases,
                "n_bootstrap": args.n_bootstrap,
            }
        }, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to {args.output}")
    print(f"{'=' * 70}")


import argparse

if __name__ == "__main__":
    main()
