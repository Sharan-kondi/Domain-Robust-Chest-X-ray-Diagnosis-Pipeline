"""
schema.py — Bridge mapping classifier PredictionResponse to structured agent input.
"""

from typing import Dict, Any, Union, Optional

def predictions_to_findings(pred_response: Any, locations: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Converts a classifier PredictionResponse (or dict representation)
    into structured findings for consumption by the Drafting Agent.
    
    Args:
        pred_response: The PredictionResponse Pydantic model or equivalent dict.
        locations: Optional dict mapping pathology labels to spatial quadrants (e.g., {'Effusion': 'left lower zone'}).
        
    Returns:
        Dict representing structured findings:
        {
            "findings": [
                {
                    "pathology": str,
                    "location": Optional[str],
                    "severity": Optional[str],
                    "confidence": float,
                    "status": str ("positive" | "uncertain" | "negative")
                },
                ...
            ],
            "predictive_entropy": float,
            "needs_human_review": bool,
            "gradcam_available": bool
        }
    """
    # Normalize input to a dictionary
    if hasattr(pred_response, "model_dump"):
        data = pred_response.model_dump()
    elif hasattr(pred_response, "dict"):
        data = pred_response.dict()
    elif isinstance(pred_response, dict):
        data = pred_response
    else:
        data = dict(pred_response)

    predictions = data.get("predictions", [])
    findings = []
    
    for p in predictions:
        # Normalize single prediction to dict
        if hasattr(p, "model_dump"):
            p_dict = p.model_dump()
        elif hasattr(p, "dict"):
            p_dict = p.dict()
        elif isinstance(p, dict):
            p_dict = p
        else:
            p_dict = dict(p)

        label = p_dict.get("label")
        probability = p_dict.get("probability", 0.0)
        positive = p_dict.get("positive", False)
        
        # Omit 'No Finding' from positive pathology findings list
        if label == "No Finding":
            continue
            
        # Determine status: positive, uncertain, or negative
        if positive:
            status = "positive"
        elif 0.35 <= probability <= 0.65:
            status = "uncertain"
        else:
            status = "negative"
            
        # Get dynamic localization quadrant if available
        location = None
        if locations and label in locations:
            location = locations[label]
            
        findings.append({
            "pathology": label,
            "location": location,
            "severity": None,  # Severity not modeled yet
            "confidence": probability,
            "status": status,
        })

    return {
        "findings": findings,
        "predictive_entropy": data.get("uncertainty", 0.0),
        "needs_human_review": data.get("needs_human_review", False),
        "gradcam_available": data.get("gradcam_available", False),
    }
