"""
metrics.py — Metrics calculation functions for evaluating drafted and verified radiology reports.
"""

from typing import List, Dict, Any

def compute_report_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes aggregate performance metrics for the reporting agent system.
    
    Args:
        results: List of execution results dictionary from run_radagent_pipeline.
        
    Returns:
        Dict containing average hallucination rate, omission rate, revision success rate, etc.
    """
    total_cases = len(results)
    if total_cases == 0:
        return {
            "total_cases": 0,
            "hallucination_rate": 0.0,
            "omission_rate": 0.0,
            "average_discrepancy_count": 0.0,
            "revision_trigger_rate": 0.0,
            "revision_correction_rate": 0.0,
            "escalation_rate": 0.0
        }
        
    total_hallucinations = 0
    total_omissions = 0
    total_escalated = 0
    
    cases_with_initial_discrepancy = 0
    cases_corrected_by_revision = 0
    cases_triggered_revision = 0
    
    for r in results:
        # Final audit
        verification = r.get("verification", {})
        total_hallucinations += len(verification.get("hallucinations", []))
        total_omissions += len(verification.get("omissions", []))
        
        if r.get("escalated", False):
            total_escalated += 1
            
        # Analyze revision loop from trace steps
        steps = r.get("steps", [])
        
        # Check first verification step
        first_verify = None
        second_verify = None
        for step in steps:
            if step.get("node") == "verification":
                if first_verify is None:
                    first_verify = step
                else:
                    second_verify = step
                    break
                    
        if first_verify:
            initial_count = first_verify.get("discrepancy_count", 0)
            if initial_count > 0:
                cases_with_initial_discrepancy += 1
                cases_triggered_revision += 1
                
                # Check if second verify exists and has fewer discrepancies
                if second_verify:
                    final_count = second_verify.get("discrepancy_count", 0)
                    if final_count < initial_count:
                        cases_corrected_by_revision += 1
                elif initial_count > 0 and r.get("escalated", False) is False:
                    # Special case where a revision succeeded but didn't run second verify or count was 0
                    cases_corrected_by_revision += 1

    revision_correction_rate = 0.0
    if cases_with_initial_discrepancy > 0:
        revision_correction_rate = cases_corrected_by_revision / cases_with_initial_discrepancy
        
    return {
        "total_cases": total_cases,
        "hallucination_rate": total_hallucinations / total_cases,
        "omission_rate": total_omissions / total_cases,
        "average_discrepancy_count": (total_hallucinations + total_omissions) / total_cases,
        "revision_trigger_rate": cases_triggered_revision / total_cases,
        "revision_correction_rate": revision_correction_rate,
        "escalation_rate": total_escalated / total_cases
    }
