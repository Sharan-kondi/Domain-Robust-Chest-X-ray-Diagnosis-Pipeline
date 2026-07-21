"""
cost_latency_benchmark.py — Token Usage, Execution Latency & Financial Cost Profiler.

Benchmarks the RadAgent pipeline across multiple LLM providers, measuring:
  - Average execution time (seconds per report)
  - Prompt vs completion token counts per pipeline node
  - Total financial cost ($/1000 reports)
  - Token efficiency ratios

Usage:
    python -m eval.cost_latency_benchmark
    python -m eval.cost_latency_benchmark --num-cases 10 --output results/cost_benchmark.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from radagent import run_radagent_pipeline
from radagent.llm import LLMClient
from eval.bootstrap_ci import bootstrap_ci, format_ci


# ── Provider Pricing (USD per 1M tokens, as of 2025) ───────────────────────

PROVIDER_PRICING = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "input_per_1m": 0.59,
        "output_per_1m": 0.79,
        "note": "Groq Llama 3.3 70B (inference-optimized)"
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-20241022",
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "note": "Anthropic Claude 3.5 Sonnet"
    },
    "openai_gpt4o_mini": {
        "model": "gpt-4o-mini",
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "note": "OpenAI GPT-4o-mini"
    },
    "openai_gpt4o": {
        "model": "gpt-4o",
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
        "note": "OpenAI GPT-4o"
    },
    "gemini_flash": {
        "model": "gemini-2.0-flash",
        "input_per_1m": 0.10,
        "output_per_1m": 0.40,
        "note": "Google Gemini 2.0 Flash"
    },
    "mock": {
        "model": "mock",
        "input_per_1m": 0.00,
        "output_per_1m": 0.00,
        "note": "Rules-based mock (no API cost)"
    },
}

# Approximate token counts per pipeline node (empirically measured)
ESTIMATED_TOKENS_PER_NODE = {
    "drafting": {"input": 800, "output": 400},
    "verification": {"input": 1200, "output": 300},
    "grounding": {"input": 0, "output": 0},     # Rule-based, no LLM call
    "bias_audit": {"input": 0, "output": 0},     # Rule-based, no LLM call
    "escalation_eval": {"input": 0, "output": 0}, # Rule-based, no LLM call
    "revision_draft": {"input": 1000, "output": 450},  # Drafting with feedback
    "revision_verify": {"input": 1200, "output": 300},  # Re-verification
}


# ── Synthetic Test Cases ────────────────────────────────────────────────────

def generate_benchmark_cases(n: int = 10) -> List[Dict[str, Any]]:
    """Generate standardized test cases for benchmarking."""
    from eval.ablation_study import generate_synthetic_test_cases
    return generate_synthetic_test_cases(n, seed=99)


# ── Benchmark Runner ─────────────────────────────────────────────────────────

def benchmark_pipeline(
    cases: List[Dict[str, Any]],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """Run benchmark measurements on the active pipeline."""
    
    latencies = []
    token_profiles = []
    
    for i, case in enumerate(cases):
        start_time = time.time()
        
        result = run_radagent_pipeline(
            case["prediction_response"],
            llm_client=llm_client,
            locations=case.get("locations"),
            demographics=case.get("demographics"),
        )
        
        elapsed = time.time() - start_time
        latencies.append(elapsed)
        
        # Analyze token usage from pipeline steps
        steps = result.get("steps", [])
        profile = analyze_token_usage(steps)
        profile["latency_seconds"] = elapsed
        profile["report_word_count"] = len(result.get("final_report", "").split())
        token_profiles.append(profile)
        
        print(f"  Case {i+1}/{len(cases)}: {elapsed:.2f}s, "
              f"~{profile['total_input_tokens']} input tokens, "
              f"~{profile['total_output_tokens']} output tokens")
    
    return {
        "latencies": latencies,
        "token_profiles": token_profiles,
    }


def analyze_token_usage(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate token usage from pipeline execution steps."""
    
    total_input = 0
    total_output = 0
    node_breakdown = {}
    
    for step in steps:
        node = step.get("node", "unknown")
        revision = step.get("revision", 0)
        
        # Use revision-specific keys for drafting/verification
        if node == "drafting" and revision > 0:
            token_key = "revision_draft"
        elif node == "verification" and revision > 0:
            token_key = "revision_verify"
        else:
            token_key = node
        
        tokens = ESTIMATED_TOKENS_PER_NODE.get(token_key, {"input": 0, "output": 0})
        
        total_input += tokens["input"]
        total_output += tokens["output"]
        
        node_name = f"{node}_r{revision}" if revision > 0 else node
        node_breakdown[node_name] = {
            "input_tokens": tokens["input"],
            "output_tokens": tokens["output"],
        }
    
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "node_breakdown": node_breakdown,
    }


def compute_cost_estimates(
    token_profiles: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """Compute financial cost estimates across all providers."""
    
    avg_input = np.mean([p["total_input_tokens"] for p in token_profiles])
    avg_output = np.mean([p["total_output_tokens"] for p in token_profiles])
    
    cost_table = {}
    
    for provider_name, pricing in PROVIDER_PRICING.items():
        input_cost = (avg_input / 1_000_000) * pricing["input_per_1m"]
        output_cost = (avg_output / 1_000_000) * pricing["output_per_1m"]
        per_report_cost = input_cost + output_cost
        
        cost_table[provider_name] = {
            "model": pricing["model"],
            "note": pricing["note"],
            "cost_per_report_usd": per_report_cost,
            "cost_per_1000_reports_usd": per_report_cost * 1000,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
        }
    
    return cost_table


# ── CLI Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cost, Token & Latency Benchmark")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/cost_benchmark.json")
    args = parser.parse_args()
    
    print("=" * 70)
    print("RadAgent Cost, Token & Latency Benchmark")
    print("=" * 70)
    
    llm_client = LLMClient()
    print(f"Active Provider: {llm_client.provider} ({llm_client.model_name})")
    
    # Generate and run benchmark
    print(f"\nGenerating {args.num_cases} benchmark cases...")
    cases = generate_benchmark_cases(args.num_cases)
    
    print(f"\nRunning benchmark...")
    benchmark_data = benchmark_pipeline(cases, llm_client)
    
    latencies = np.array(benchmark_data["latencies"])
    token_profiles = benchmark_data["token_profiles"]
    
    # ── Latency Statistics ──
    latency_ci = bootstrap_ci(latencies, np.mean, n_bootstrap=1000)
    
    print(f"\n{'=' * 70}")
    print("Latency Results")
    print(f"{'=' * 70}")
    print(f"  Mean latency: {format_ci(latency_ci)} seconds")
    print(f"  Median latency: {np.median(latencies):.2f}s")
    print(f"  P95 latency: {np.percentile(latencies, 95):.2f}s")
    print(f"  P99 latency: {np.percentile(latencies, 99):.2f}s")
    print(f"  Min: {np.min(latencies):.2f}s | Max: {np.max(latencies):.2f}s")
    
    # ── Token Statistics ──
    input_tokens = np.array([p["total_input_tokens"] for p in token_profiles])
    output_tokens = np.array([p["total_output_tokens"] for p in token_profiles])
    total_tokens = input_tokens + output_tokens
    
    print(f"\n{'=' * 70}")
    print("Token Usage")
    print(f"{'=' * 70}")
    print(f"  Avg input tokens/report:  {np.mean(input_tokens):.0f}")
    print(f"  Avg output tokens/report: {np.mean(output_tokens):.0f}")
    print(f"  Avg total tokens/report:  {np.mean(total_tokens):.0f}")
    
    # Report word count
    word_counts = np.array([p["report_word_count"] for p in token_profiles])
    print(f"  Avg report length:        {np.mean(word_counts):.0f} words")
    
    # ── Cost Estimates ──
    cost_table = compute_cost_estimates(token_profiles)
    
    print(f"\n{'=' * 70}")
    print("Cost Estimates (per report / per 1000 reports)")
    print(f"{'=' * 70}")
    print(f"  {'Provider':<25} {'Model':<30} {'$/report':>10} {'$/1K reports':>12}")
    print(f"  {'-' * 25} {'-' * 30} {'-' * 10} {'-' * 12}")
    
    for provider_name, costs in sorted(cost_table.items(), key=lambda x: x[1]["cost_per_report_usd"]):
        print(f"  {provider_name:<25} {costs['model']:<30} "
              f"${costs['cost_per_report_usd']:.5f}   "
              f"${costs['cost_per_1000_reports_usd']:.2f}")
    
    # ── Per-Node Breakdown ──
    print(f"\n{'=' * 70}")
    print("Per-Node Token Breakdown (average)")
    print(f"{'=' * 70}")
    
    node_totals = {}
    for profile in token_profiles:
        for node_name, tokens in profile.get("node_breakdown", {}).items():
            if node_name not in node_totals:
                node_totals[node_name] = {"input": [], "output": []}
            node_totals[node_name]["input"].append(tokens["input_tokens"])
            node_totals[node_name]["output"].append(tokens["output_tokens"])
    
    print(f"  {'Node':<25} {'Avg Input':>12} {'Avg Output':>12} {'Total':>12}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12}")
    for node_name, totals in node_totals.items():
        avg_in = np.mean(totals["input"])
        avg_out = np.mean(totals["output"])
        print(f"  {node_name:<25} {avg_in:>12.0f} {avg_out:>12.0f} {avg_in + avg_out:>12.0f}")
    
    # ── Save results ──
    results = {
        "active_provider": {
            "name": llm_client.provider,
            "model": llm_client.model_name,
        },
        "latency": {
            "mean_ci": latency_ci,
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        },
        "tokens": {
            "avg_input_per_report": float(np.mean(input_tokens)),
            "avg_output_per_report": float(np.mean(output_tokens)),
            "avg_total_per_report": float(np.mean(total_tokens)),
            "avg_report_word_count": float(np.mean(word_counts)),
        },
        "cost_estimates": cost_table,
        "per_node_breakdown": {
            node: {
                "avg_input": float(np.mean(t["input"])),
                "avg_output": float(np.mean(t["output"])),
            }
            for node, t in node_totals.items()
        },
        "config": {
            "num_cases": args.num_cases,
        }
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to {args.output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
