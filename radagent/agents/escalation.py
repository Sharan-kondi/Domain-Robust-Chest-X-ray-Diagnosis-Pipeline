"""
escalation.py — Escalation Agent that determines if a report requires human radiologist review.
"""

from typing import Dict, Any, List

class EscalationAgent:
    """Agent responsible for checking safety parameters and deciding if human review is needed."""
    
    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates clinical triggers to determine if case should be escalated.
        
        Triggers:
        1. Classifier uncertainty (predictive entropy > routing threshold).
        2. Discrepancy count > 0 after verification.
        3. Critical positive pathology lacks a grounding citation.
        
        Returns:
            Dict containing:
                "escalated": bool
                "escalation_reasons": List[str]
        """
        findings = state.get("findings", {})
        verification = state.get("verification", {}) or {}
        grounding = state.get("grounding", []) or []
        
        # Trigger 1: High Classifier Uncertainty
        needs_review = findings.get("needs_human_review", False)
        
        # Trigger 2: Verification Discrepancies
        discrepancy_count = state.get("discrepancy_count")
        if discrepancy_count is None:
            discrepancy_count = verification.get("discrepancy_count", 0)
            
        # Trigger 3: Grounding Citation Failure for Positive Pathology
        lacks_grounding = False
        pos_pathologies = [
            f.get("pathology") 
            for f in findings.get("findings", []) 
            if f.get("status") == "positive"
        ]
        grounded_pathologies = [g.get("pathology") for g in grounding]
        for p in pos_pathologies:
            if p not in grounded_pathologies:
                lacks_grounding = True
                break
                
        # Trigger 4: Demographic Performance Disparity
        bias = state.get("bias", {}) or {}
        disparity_detected = bias.get("bias_disparity_detected", False)
        
        # Consolidate routing decision
        escalated = needs_review or (discrepancy_count > 0) or lacks_grounding or disparity_detected
        
        reasons = []
        if needs_review:
            reasons.append("High classifier predictive uncertainty (exceeds safety routing threshold)")
        if discrepancy_count > 0:
            reasons.append(f"Unresolved report discrepancies ({discrepancy_count} hallucinations/omissions remaining)")
        if lacks_grounding:
            reasons.append("ACR clinical grounding failure: positive finding lacks standard imaging recommendation citation")
        if disparity_detected:
            reasons.append(f"Demographic performance disparity safety override: patient falls under vulnerable cohort ({bias.get('bias_subgroup')})")
            
        return {
            "escalated": escalated,
            "escalation_reasons": reasons
        }
