"""
verification.py — Verification Agent that cross-checks the draft report against structured findings.
"""

import json
import re
from typing import Dict, Any, Optional
from radagent.llm import LLMClient

VERIFICATION_SYSTEM_PROMPT = """You are a rigorous radiology report auditor. 
Your task is to compare a generated radiology draft report against the original structured classifier findings and identify any discrepancies.

You must follow this critical rule:
- The draft report may contain a section labeled "PATIENT EXPLANATION:". You MUST ignore this section entirely during claims auditing and verification. Do not extract claims or flags from it. Only audit the "FINDINGS:" and "IMPRESSION:" sections.

You must identify two categories of discrepancies within the technical sections:
1. Hallucinations: 
   - A hallucination is when the draft report asserts the presence of a pathology that is NOT marked as 'positive' or 'uncertain' in the structured findings (e.g., status is 'negative' or not present in findings).
   - If a pathology is mentioned as absent/negative in the draft, this is NOT a hallucination. Only flag claims where the pathology is asserted as active/present without structured findings support.
   
2. Omissions:
   - An omission is when a pathology marked as 'positive' or 'uncertain' in the structured findings is completely missing or not mentioned in the draft report.
   - Negative findings do NOT need to be mentioned; omitting them is NOT an discrepancy.

You must output your audit results strictly in JSON format. Do not write any conversational text or explanation outside the JSON. The JSON structure must be:
{
  "hallucinations": [
    {
      "claim": "[The exact sentence/claim from the draft]",
      "explanation": "[Explanation of why it is unsupported]"
    }
  ],
  "omissions": [
    {
      "finding": "[The name of the omitted pathology]",
      "explanation": "[Explanation of why it should be included]"
    }
  ],
  "discrepancy_count": [Total number of hallucinations + omissions, as an integer]
}
"""

VERIFICATION_USER_TEMPLATE = """Structured findings:
{findings_json}

Draft report to verify:
{draft_text}

Please audit the draft report and return the structured JSON results.
"""

class VerificationAgent:
    """Agent that performs claims matching to identify hallucinations and omissions in a draft."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        
    def verify(self, findings: Dict[str, Any], draft_text: str) -> Dict[str, Any]:
        """Audits the draft report against structured findings, returning a dict of discrepancies."""
        findings_json = json.dumps(findings, indent=2)
        
        user_prompt = VERIFICATION_USER_TEMPLATE.format(
            findings_json=findings_json,
            draft_text=draft_text
        )
        
        raw_response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=VERIFICATION_SYSTEM_PROMPT
        )
        
        verify_result = self._parse_json_response(raw_response)
        
        # ── Pointing-Game Spatial Verification ──
        # Check if the text description of laterality conflicts with the Grad-CAM derived findings
        for f in findings.get("findings", []):
            if f.get("status") in ["positive", "uncertain"] and f.get("location"):
                pathology = f.get("pathology")
                true_loc = f.get("location").lower() # e.g. "left lower zone"
                
                # Split draft_text into sentences
                sentences = re.split(r"[.!?\n]", draft_text)
                for sent in sentences:
                    sent_lower = sent.lower()
                    p_terms = [pathology.lower(), pathology.lower().split("/")[0]]
                    matches_p = any(term in sent_lower for term in p_terms)
                    
                    if matches_p:
                        # Mismatch check (left in finding, right in text; or vice-versa)
                        if "left" in true_loc and "right" in sent_lower:
                            verify_result["hallucinations"].append({
                                "claim": sent.strip(),
                                "explanation": f"Spatial Mismatch: The report describes {pathology} as right-sided, but model attention (Grad-CAM peak) localizes it to the left side ({true_loc})."
                            })
                            verify_result["discrepancy_count"] = verify_result.get("discrepancy_count", 0) + 1
                        elif "right" in true_loc and "left" in sent_lower:
                            verify_result["hallucinations"].append({
                                "claim": sent.strip(),
                                "explanation": f"Spatial Mismatch: The report describes {pathology} as left-sided, but model attention (Grad-CAM peak) localizes it to the right side ({true_loc})."
                            })
                            verify_result["discrepancy_count"] = verify_result.get("discrepancy_count", 0) + 1
                            
        return verify_result

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Cleans and parses the JSON response from the LLM, with robust fallbacks."""
        # Find JSON boundaries
        cleaned = text.strip()
        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            # Strip first line
            cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)
            cleaned = cleaned.strip()
            
        try:
            parsed = json.loads(cleaned)
            # Ensure required fields exist
            if not isinstance(parsed, dict):
                parsed = {}
            if "hallucinations" not in parsed:
                parsed["hallucinations"] = []
            if "omissions" not in parsed:
                parsed["omissions"] = []
            if "discrepancy_count" not in parsed:
                parsed["discrepancy_count"] = len(parsed["hallucinations"]) + len(parsed["omissions"])
            return parsed
        except json.JSONDecodeError as e:
            print(f"[VerificationAgent] Failed to parse JSON: {e}. Raw response: {text}")
            # Try a regex-based extract of json if there is trailing noise
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if "hallucinations" not in parsed: parsed["hallucinations"] = []
                    if "omissions" not in parsed: parsed["omissions"] = []
                    parsed["discrepancy_count"] = len(parsed["hallucinations"]) + len(parsed["omissions"])
                    return parsed
                except Exception:
                    pass
            # Fallback to no discrepancies
            return {
                "hallucinations": [],
                "omissions": [],
                "discrepancy_count": 0,
                "error": "Failed to parse verification response"
            }
