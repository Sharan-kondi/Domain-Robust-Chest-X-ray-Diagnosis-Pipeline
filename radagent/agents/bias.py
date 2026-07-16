"""
bias.py — Bias Audit Agent checking demographic subgroups for performance disparities.
"""

from typing import Dict, Any, Optional

class BiasAuditAgent:
    """Agent responsible for checking demographic performance disparities and warning routing."""

    def audit(self, demographics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Audits demographic characteristics of the patient scan for known performance disparities.
        
        Baseline subgroup audit performance:
        - Age group 70+: macro AUC is 0.449 (baseline average is 0.585).
          This drop occurs due to multiple co-morbidities and structural changes.
          
        Args:
            demographics: Dictionary containing 'age' (int/str) and 'gender' (str).
            
        Returns:
            Dict containing bias results:
            {
                "bias_disparity_detected": bool,
                "bias_subgroup": Optional[str],
                "bias_warning": Optional[str]
            }
        """
        if not demographics:
            return {
                "bias_disparity_detected": False,
                "bias_subgroup": None,
                "bias_warning": None
            }
            
        age = demographics.get("age")
        gender = demographics.get("gender", "").strip().upper()
        
        disparity_detected = False
        subgroup = None
        warning = None
        
        # Audit Age Group
        if age is not None:
            try:
                age_val = int(age)
                if age_val >= 70:
                    disparity_detected = True
                    subgroup = "Age Group: 70+"
                    warning = "Demographic Performance Warning: Diagnostics for patients aged 70+ show a documented subgroup performance drop (empirical macro AUC 0.449 vs 0.585 baseline) due to complex geriatric co-morbidities."
            except ValueError:
                # Catch string brackets like "70+" or "70-79"
                age_str = str(age)
                if "70" in age_str or "80" in age_str or "90" in age_str:
                    disparity_detected = True
                    subgroup = "Age Group: 70+"
                    warning = "Demographic Performance Warning: Diagnostics for patients aged 70+ show a documented subgroup performance drop (empirical macro AUC 0.449 vs 0.585 baseline) due to complex geriatric co-morbidities."

        return {
            "bias_disparity_detected": disparity_detected,
            "bias_subgroup": subgroup,
            "bias_warning": warning
        }
