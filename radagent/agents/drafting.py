"""
drafting.py — Drafting Agent that drafts radiology reports with confidence-aware hedging.
"""

import json
from typing import Dict, Any, Optional
from radagent.llm import LLMClient

DRAFTING_SYSTEM_PROMPT = """You are an expert radiology AI reporting assistant. 
Your task is to draft a structured radiology report based strictly on the provided structured findings.

You must follow these rules:
1. Divide your report into three clear sections:
   FINDINGS:
   [Technical bullet points describing specific findings using professional medical terminology]

   IMPRESSION:
   [A technical summary paragraph of the main clinical takeaways using professional medical terminology]

   PATIENT EXPLANATION:
   [A translation of the findings and impression into very simple, easy-to-understand, reassuring, and clear plain-English terms for the patient. Explain medical terms clearly (e.g., explain cardiomegaly as "mild enlargement of the heart muscle", effusion as "fluid collection around the lung", consolidation as "shadowing or minor fluid collection in the lung tissue", atelectasis as "minor collapse of tiny air sacs"). Be encouraging and avoid unnecessarily scary wording.]

2. Confidence-aware Phrasing:
   - For findings with status 'positive' (or confidence/probability > 0.65), assert the finding with high clinical confidence.
   - For findings with status 'uncertain' (or confidence/probability between 0.50 and 0.65), use hedged/conditional language (e.g., "findings suggestive of", "cannot exclude", "suspicious for").
   - For findings with status 'negative', do not assert them as present. Mention them as absent or not seen if clinically relevant, or omit them.
   - If all findings are negative (or if the only significant finding is No Finding), state "No acute cardiopulmonary abnormality" or similar in the findings, impression, and patient explanation.

3. Anatomy, Location, and Severity:
   - Do NOT invent/hallucinate specific locations (e.g., "right lower lobe", "retrocardiac") or severities (e.g., "moderate", "severe") that are not provided in the findings.
   - Since location and severity are currently null in the findings, describe the pathology generally without asserting specific zones or laterality unless location is provided.

4. Taxonomy limit:
   - Only discuss pathologies from the 8-class taxonomy: Atelectasis, Cardiomegaly, Consolidation, Effusion, Pneumonia, Pneumothorax, Nodule/Mass. Do not introduce extraneous diagnoses.
"""

DRAFTING_USER_TEMPLATE = """Here are the structured findings for the chest X-ray:
{findings_json}

Please draft the radiology report following the rules. Write ONLY the report starting with FINDINGS:, then IMPRESSION:, and ending with the PATIENT EXPLANATION: section. No greeting or conversational text.
"""

class DraftingAgent:
    """Agent that creates a radiology report draft from structured classifier findings."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        
    def draft(self, findings: Dict[str, Any], feedback: Optional[str] = None, clinical_focus: Optional[str] = None) -> str:
        """Generates a draft report. Optional feedback is provided during re-drafting."""
        findings_json = json.dumps(findings, indent=2)
        
        user_prompt = DRAFTING_USER_TEMPLATE.format(findings_json=findings_json)
        if feedback:
            user_prompt += f"\n\nCRITICAL FEEDBACK FROM VERIFICATION AGENT:\n{feedback}\nPlease correct the draft according to this feedback."
        if clinical_focus:
            user_prompt += f"\n\nCLINICIAN CUSTOM DIRECTIVE:\nPlease target/focus the report based on this directive: {clinical_focus}"
            
        draft_text = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=DRAFTING_SYSTEM_PROMPT
        )
        return draft_text.strip()
