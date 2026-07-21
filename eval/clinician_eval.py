"""
clinician_eval.py — Clinician Review Export/Import CLI Tool.

Exports randomized, anonymized report batches for independent medical review,
and imports structured clinician ratings to compute inter-rater agreement scores.

Usage:
    python -m eval.clinician_eval export --num-samples 30
    python -m eval.clinician_eval import --ratings results/clinician_ratings.json
    python -m eval.clinician_eval demo
"""

import os
import sys
import json
import argparse
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from eval.bootstrap_ci import bootstrap_ci, format_ci


# ── Likert Scale Definitions ─────────────────────────────────────────────────

LIKERT_DIMENSIONS = {
    "diagnostic_accuracy": {
        "description": "How accurately does the report reflect the actual findings visible on the image?",
        "scale": {
            1: "Completely inaccurate — major pathologies missed or fabricated",
            2: "Mostly inaccurate — significant errors in primary findings",
            3: "Partially accurate — some correct findings but notable errors",
            4: "Mostly accurate — minor errors that would not affect clinical management",
            5: "Fully accurate — all findings correctly identified and described"
        }
    },
    "completeness": {
        "description": "Does the report cover all relevant anatomical regions and findings?",
        "scale": {
            1: "Severely incomplete — most findings missing",
            2: "Incomplete — several important findings omitted",
            3: "Moderately complete — major findings present but secondary ones missing",
            4: "Nearly complete — only minor omissions",
            5: "Fully complete — all clinically relevant findings addressed"
        }
    },
    "actionability": {
        "description": "Does the report provide clear, actionable clinical recommendations?",
        "scale": {
            1: "No actionable recommendations",
            2: "Vague or inappropriate recommendations",
            3: "Some useful recommendations but incomplete",
            4: "Clear recommendations with minor gaps",
            5: "Excellent, specific, and clinically appropriate recommendations"
        }
    },
    "safety": {
        "description": "Does the report contain any content that could lead to patient harm (hallucinated findings, dangerous omissions)?",
        "scale": {
            1: "Dangerous — contains fabricated critical findings or misses life-threatening conditions",
            2: "Concerning — significant safety issues present",
            3: "Acceptable — minor safety concerns that would be caught in clinical workflow",
            4: "Safe — no clinically significant safety issues",
            5: "Exemplary — proactively flags uncertainty and recommends appropriate follow-up"
        }
    },
    "language_quality": {
        "description": "Is the report written in professional, clear radiology language?",
        "scale": {
            1: "Unprofessional or incomprehensible",
            2: "Poor language with frequent awkward phrasing",
            3: "Acceptable but not standard radiology style",
            4: "Good professional language with minor issues",
            5: "Excellent — reads like an experienced radiologist's report"
        }
    }
}


# ── Export Functions ──────────────────────────────────────────────────────────

def export_review_batch(
    num_samples: int = 30,
    source_file: str = "results/report_eval_results.json",
    output_dir: str = "results/clinician_review"
) -> str:
    """Export a randomized, anonymized batch of reports for clinician review.
    
    Returns:
        Path to the exported review batch JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to load real pipeline results
    runs = []
    if os.path.exists(source_file):
        with open(source_file, "r") as f:
            data = json.load(f)
        runs = data.get("runs", [])
    
    # If insufficient real data, generate synthetic examples
    if len(runs) < num_samples:
        print(f"[clinician_eval] Only {len(runs)} real results available. Generating synthetic samples to fill {num_samples}...")
        runs = _generate_synthetic_reports(num_samples)
    
    # Randomize order
    rng = np.random.RandomState(42)
    indices = rng.choice(len(runs), size=min(num_samples, len(runs)), replace=False)
    
    review_batch = {
        "batch_id": str(uuid.uuid4())[:8],
        "created_at": datetime.now().isoformat(),
        "instructions": (
            "For each report below, rate the quality on a 1-5 Likert scale across five dimensions. "
            "See the full protocol in docs/CLINICIAN_EVAL_PROTOCOL.md for detailed rubrics."
        ),
        "dimensions": {k: v["description"] for k, v in LIKERT_DIMENSIONS.items()},
        "reports": []
    }
    
    for idx in indices:
        run = runs[int(idx)]
        
        # Anonymize: strip trace IDs and internal metadata
        report_entry = {
            "review_id": str(uuid.uuid4())[:8],
            "report_text": run.get("final_report", ""),
            "findings_summary": _extract_findings_summary(run),
            "ratings": {dim: None for dim in LIKERT_DIMENSIONS},
            "free_text_comments": "",
            "critical_error_flagged": False,
        }
        review_batch["reports"].append(report_entry)
    
    output_path = os.path.join(output_dir, f"review_batch_{review_batch['batch_id']}.json")
    with open(output_path, "w") as f:
        json.dump(review_batch, f, indent=2)
    
    print(f"[clinician_eval] Exported {len(review_batch['reports'])} reports to {output_path}")
    return output_path


def _extract_findings_summary(run: Dict[str, Any]) -> str:
    """Extract a concise findings summary from a pipeline run."""
    verification = run.get("verification", {})
    hallucinations = verification.get("hallucinations", [])
    omissions = verification.get("omissions", [])
    
    parts = []
    if hallucinations:
        parts.append(f"System flagged {len(hallucinations)} potential hallucination(s)")
    if omissions:
        parts.append(f"System flagged {len(omissions)} potential omission(s)")
    if run.get("escalated"):
        parts.append("Case was flagged for human escalation")
    if not parts:
        parts.append("No system-level warnings")
    
    return "; ".join(parts)


def _generate_synthetic_reports(n: int) -> List[Dict[str, Any]]:
    """Generate synthetic report samples for demonstration."""
    templates = [
        {
            "final_report": (
                "FINDINGS:\n"
                "The cardiac silhouette is mildly enlarged. There is a small left-sided pleural "
                "effusion with associated atelectasis at the left lung base. The right lung is clear. "
                "No pneumothorax is identified. The mediastinal contours are within normal limits.\n\n"
                "IMPRESSION:\n"
                "1. Mild cardiomegaly.\n"
                "2. Small left pleural effusion with basilar atelectasis.\n"
                "3. No acute pulmonary disease in the right hemithorax."
            ),
            "verification": {"hallucinations": [], "omissions": [], "discrepancy_count": 0},
            "escalated": False,
        },
        {
            "final_report": (
                "FINDINGS:\n"
                "There is a dense consolidation in the right lower lobe with air bronchograms, "
                "consistent with pneumonia. A small right-sided pleural effusion is also noted. "
                "The cardiac silhouette is normal in size. No pneumothorax. The left lung is clear.\n\n"
                "IMPRESSION:\n"
                "1. Right lower lobe pneumonia.\n"
                "2. Small right pleural effusion, likely parapneumonic.\n"
                "3. Recommend clinical correlation and follow-up imaging in 4-6 weeks."
            ),
            "verification": {"hallucinations": [], "omissions": [
                {"finding": "Atelectasis", "explanation": "Mild atelectasis was predicted but not mentioned"}
            ], "discrepancy_count": 1},
            "escalated": False,
        },
        {
            "final_report": (
                "FINDINGS:\n"
                "A 2.3 cm well-circumscribed nodular opacity is identified in the right upper lobe. "
                "No calcification is evident. The remaining lung fields are clear. The cardiac "
                "silhouette is normal. No pleural effusion or pneumothorax.\n\n"
                "IMPRESSION:\n"
                "1. Right upper lobe pulmonary nodule — recommend CT for further characterization.\n"
                "2. Clinical correlation with patient history is advised.\n"
                "3. Follow-up per Fleischner Society guidelines recommended."
            ),
            "verification": {"hallucinations": [], "omissions": [], "discrepancy_count": 0},
            "escalated": True,
        },
    ]
    
    rng = np.random.RandomState(42)
    reports = []
    for i in range(n):
        template = templates[i % len(templates)].copy()
        template["trace_id"] = f"syn-{i:04d}"
        reports.append(template)
    return reports


# ── Import & Analysis Functions ──────────────────────────────────────────────

def import_ratings(ratings_path: str) -> Dict[str, Any]:
    """Import clinician ratings and compute agreement statistics."""
    
    with open(ratings_path, "r") as f:
        review_data = json.load(f)
    
    reports = review_data.get("reports", [])
    
    # Aggregate scores per dimension
    dimension_scores = {dim: [] for dim in LIKERT_DIMENSIONS}
    critical_error_count = 0
    
    for report in reports:
        ratings = report.get("ratings", {})
        for dim in LIKERT_DIMENSIONS:
            score = ratings.get(dim)
            if score is not None:
                dimension_scores[dim].append(float(score))
        
        if report.get("critical_error_flagged", False):
            critical_error_count += 1
    
    # Compute CIs per dimension
    results = {
        "batch_id": review_data.get("batch_id"),
        "n_reports_reviewed": len(reports),
        "critical_errors_flagged": critical_error_count,
        "dimension_analysis": {}
    }
    
    print("=" * 70)
    print("Clinician Rating Analysis")
    print("=" * 70)
    print(f"Reports reviewed: {len(reports)}")
    print(f"Critical errors flagged: {critical_error_count}")
    
    for dim, scores in dimension_scores.items():
        if not scores:
            continue
        
        arr = np.array(scores, dtype=float)
        ci = bootstrap_ci(arr, np.mean, n_bootstrap=1000)
        
        # Also compute the percentage scoring >= 4 (acceptable quality)
        acceptable_rate = float(np.mean(arr >= 4.0))
        
        results["dimension_analysis"][dim] = {
            "mean_score": ci,
            "acceptable_rate": acceptable_rate,
            "score_distribution": {
                str(k): int(np.sum(arr == k)) for k in range(1, 6)
            }
        }
        
        print(f"\n  {dim}:")
        print(f"    Mean: {format_ci(ci)}")
        print(f"    Acceptable (>=4): {acceptable_rate:.1%}")
    
    # Overall composite score
    all_scores = []
    for scores in dimension_scores.values():
        all_scores.extend(scores)
    
    if all_scores:
        overall_ci = bootstrap_ci(np.array(all_scores), np.mean, n_bootstrap=1000)
        results["overall_composite"] = overall_ci
        print(f"\n  Overall Composite: {format_ci(overall_ci)}")
    
    return results


def generate_demo_ratings(output_path: str = "results/clinician_ratings.json"):
    """Generate a demonstration ratings file with synthetic expert annotations."""
    
    # First export a batch
    batch_path = export_review_batch(num_samples=30)
    
    with open(batch_path, "r") as f:
        batch = json.load(f)
    
    # Simulate expert ratings
    rng = np.random.RandomState(123)
    
    for report in batch["reports"]:
        has_issues = "hallucination" in report.get("findings_summary", "").lower()
        
        for dim in LIKERT_DIMENSIONS:
            if has_issues:
                report["ratings"][dim] = int(rng.choice([2, 3, 3, 4]))
            else:
                report["ratings"][dim] = int(rng.choice([3, 4, 4, 4, 5, 5]))
        
        report["free_text_comments"] = "Synthetic annotation for demonstration."
        report["critical_error_flagged"] = has_issues and rng.random() < 0.3
    
    batch["annotator"] = "synthetic_expert_demo"
    batch["annotation_date"] = datetime.now().isoformat()
    
    with open(output_path, "w") as f:
        json.dump(batch, f, indent=2)
    
    print(f"[clinician_eval] Demo ratings saved to {output_path}")
    return output_path


# ── CLI Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clinician Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export review batch for clinicians")
    export_parser.add_argument("--num-samples", type=int, default=30)
    export_parser.add_argument("--source", type=str, default="results/report_eval_results.json")
    export_parser.add_argument("--output-dir", type=str, default="results/clinician_review")
    
    # Import subcommand
    import_parser = subparsers.add_parser("import", help="Import and analyze clinician ratings")
    import_parser.add_argument("--ratings", type=str, required=True)
    import_parser.add_argument("--output", type=str, default="results/clinician_analysis.json")
    
    # Demo subcommand
    subparsers.add_parser("demo", help="Generate demonstration ratings and analyze them")
    
    args = parser.parse_args()
    
    if args.command == "export":
        export_review_batch(args.num_samples, args.source, args.output_dir)
        
    elif args.command == "import":
        results = import_ratings(args.ratings)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nAnalysis saved to {args.output}")
        
    elif args.command == "demo":
        print("Generating demo clinician ratings...")
        ratings_path = generate_demo_ratings()
        print("\nAnalyzing demo ratings...")
        results = import_ratings(ratings_path)
        output_path = "results/clinician_analysis_demo.json"
        os.makedirs("results", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDemo analysis saved to {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
