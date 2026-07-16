"""
retriever.py — Keyword-based RAG retriever for fetching ACR Appropriateness Criteria guidelines.
"""

import os
import yaml
from typing import Dict, Any, Optional

class ACRRetriever:
    """Retriever for matching findings with standard ACR Appropriateness Criteria guidelines."""
    
    def __init__(self, yaml_path: str = "radagent/rag/acr_guidelines.yaml"):
        # Default fallback to find the file from project root
        self.yaml_path = yaml_path
        self.guidelines = {}
        
        if not os.path.isabs(self.yaml_path):
            # Check project relative directories
            possible_paths = [
                self.yaml_path,
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), self.yaml_path),
                os.path.join("radagent", "rag", "acr_guidelines.yaml")
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    self.yaml_path = p
                    break
                    
        if os.path.exists(self.yaml_path):
            try:
                with open(self.yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    self.guidelines = data.get("guidelines", {})
                print(f"[ACRRetriever] Loaded guidelines from {self.yaml_path}")
            except Exception as e:
                print(f"[ACRRetriever] Failed to load guidelines: {e}")
        else:
            print(f"[ACRRetriever] WARNING: Guideline database not found at {self.yaml_path}")

    def _normalize_key(self, pathology: str) -> str:
        """Normalizes pathology label names to database keys (e.g. Nodule/Mass -> nodule_mass)."""
        p_lower = pathology.lower().strip()
        if "nodule" in p_lower or "mass" in p_lower:
            return "nodule_mass"
        # Map simple replacements
        return p_lower.replace(" ", "_")

    def retrieve(self, pathology: str) -> Optional[Dict[str, Any]]:
        """Retrieves clinical guideline recommendations for a pathology finding."""
        norm_key = self._normalize_key(pathology)
        return self.guidelines.get(norm_key)
