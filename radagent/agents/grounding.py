"""
grounding.py — Grounding Agent that retrieves and maps ACR Appropriateness Criteria to report recommendations.
"""

from typing import Dict, Any, List, Optional
from radagent.rag.retriever import ACRRetriever

class GroundingAgent:
    """Agent responsible for grounding radiology report recommendations using clinical guidelines."""
    
    def __init__(self, retriever: Optional[ACRRetriever] = None):
        self.retriever = retriever or ACRRetriever()

    def ground(self, findings: Dict[str, Any], current_report: str) -> Dict[str, Any]:
        """Scans the findings, retrieves standard guidelines, and compiles structured grounding citations.
        
        Args:
            findings: Structured findings dict containing pathology items.
            current_report: The draft or finalized report text.
            
        Returns:
            Dict containing the 'grounding' key with a list of citations:
            [
                {
                    "acr_code": str,
                    "section": str,
                    "recommendation": str,
                    "citations": List[str],
                    "pathology": str
                },
                ...
            ]
        """
        citations = []
        findings_list = findings.get("findings", [])
        
        for f in findings_list:
            # We ground any positive or clinically uncertain findings
            if f.get("status") in ["positive", "uncertain"]:
                pathology = f.get("pathology")
                acr_data = self.retriever.retrieve(pathology)
                
                if acr_data:
                    citations.append({
                        "acr_code": acr_data.get("acr_code"),
                        "section": acr_data.get("section"),
                        "recommendation": acr_data.get("recommendation"),
                        "citations": acr_data.get("citations", []),
                        "pathology": pathology
                    })
                    
        return {
            "grounding": citations
        }
