"""
llm.py — Unified LLM client supporting Anthropic API and rules-based Mock fallback.
"""

import os
import yaml
import json
import re
import requests
from typing import Optional, Dict, Any

class LLMClient:
    """Unified client for executing LLM generations."""
    
    def __init__(self, config_path: str = "configs/radagent.yaml"):
        self.config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[LLMClient] Failed to load config: {e}")
                
        llm_cfg = self.config.get("llm", {})
        self.provider = llm_cfg.get("provider", "anthropic")
        self.model_name = llm_cfg.get("model_name", "claude-3-5-sonnet-20241022")
        self.temperature = llm_cfg.get("temperature", 0.1)
        self.max_tokens = llm_cfg.get("max_tokens", 1024)
        
        # API Keys
        if self.provider in ["gemini", "google"]:
            self.api_key = (
                os.environ.get("GEMINI_API_KEY", "") 
                or os.environ.get("GOOGLE_API_KEY", "") 
                or llm_cfg.get("api_key", "")
            )
            # Default model name for Gemini if still showing Claude
            if "claude" in self.model_name:
                self.model_name = "gemini-2.0-flash"
        elif self.provider == "groq":
            self.api_key = os.environ.get("GROQ_API_KEY", "") or llm_cfg.get("api_key", "")
            if "claude" in self.model_name or "gemini" in self.model_name:
                self.model_name = "llama-3.3-70b-versatile"
        else:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "") or llm_cfg.get("api_key", "")
        
        # Override to mock if key is missing
        if self.provider in ["anthropic", "gemini", "google", "groq"] and not self.api_key:
            print(f"[LLMClient] WARNING: API key for {self.provider} is not set in environment or config.")
            print("[LLMClient] Falling back to Mock LLM provider.")
            self.provider = "mock"
            
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate content from the LLM based on provider."""
        if self.provider == "mock":
            return self._generate_mock(prompt)
            
        if self.provider == "anthropic":
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            
            payload = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            if system_prompt:
                payload["system"] = system_prompt
                
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                res_data = response.json()
                content = res_data.get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", "")
                return ""
            except Exception as e:
                print(f"[LLMClient] Anthropic API request failed: {e}")
                print("[LLMClient] Falling back to Mock LLM for this generation.")
                return self._generate_mock(prompt)
                
        if self.provider in ["gemini", "google"]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            headers = {"content-type": "application/json"}
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens
                }
            }
            if system_prompt:
                payload["systemInstruction"] = {
                    "parts": [
                        {"text": system_prompt}
                    ]
                }
                
            import time
            for attempt in range(3):
                # Rate-limiting to respect the 15 RPM free-tier quota limit
                time.sleep(4.0)
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=30.0)
                    if response.status_code == 429:
                        sleep_time = (attempt + 1) * 8
                        print(f"[LLMClient] Rate limit (429) hit. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                    response.raise_for_status()
                    res_data = response.json()
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                    return ""
                except Exception as e:
                    if attempt == 2:
                        print(f"[LLMClient] Gemini API request failed: {e}")
                        print("[LLMClient] Falling back to Mock LLM for this generation.")
                        return self._generate_mock(prompt)
                    time.sleep(2.0)
            
            # If the loop finishes without returning, fallback to mock
            print("[LLMClient] All Gemini API attempts failed. Falling back to Mock LLM.")
            return self._generate_mock(prompt)

        if self.provider == "groq":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                res_data = response.json()
                choices = res_data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return ""
            except Exception as e:
                print(f"[LLMClient] Groq API request failed: {e}")
                print("[LLMClient] Falling back to Mock LLM for this generation.")
                return self._generate_mock(prompt)
                
        raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _extract_json_findings(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts valid structured findings JSON from LLM prompt or output."""
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            try:
                data = json.loads(text[first_brace:last_brace+1])
                if isinstance(data, dict) and "findings" in data:
                    return data
            except Exception:
                pass

        starts = [i for i, c in enumerate(text) if c == '{']
        for start in starts:
            for end in range(len(text) - 1, start, -1):
                if text[end] == '}':
                    try:
                        data = json.loads(text[start:end+1])
                        if isinstance(data, dict) and "findings" in data:
                            return data
                    except Exception:
                        continue
        return None

    def _generate_mock(self, prompt: str) -> str:
        """Rules-based template mock that parses prompt inputs and maps them logically."""
        # 1. Check if it's a verification request
        is_verify = "verify" in prompt.lower() or "discrepancy" in prompt.lower() or "hallucination" in prompt.lower()
        
        if is_verify:
            # Extract JSON findings
            findings_data = self._extract_json_findings(prompt)
            
            pos_uncertain = []
            if findings_data and "findings" in findings_data:
                for f in findings_data["findings"]:
                    if f.get("status") in ["positive", "uncertain"]:
                        pos_uncertain.append(f.get("pathology").lower())
            
            # Extract draft report from prompt
            draft_text = ""
            draft_match = re.search(r"(?:draft|report\s+text|draft\s+report):?\s*\n?(.*)", prompt, re.IGNORECASE | re.DOTALL)
            if draft_match:
                draft_text = draft_match.group(1).lower()
            else:
                draft_text = prompt.lower()
                
            hallucinations = []
            omissions = []
            
            # Standard checklist of our 8 classes
            all_classes = ["atelectasis", "cardiomegaly", "consolidation", "effusion", "pneumonia", "pneumothorax", "nodule/mass", "nodule", "mass"]
            
            # Find hallucinated positive claims in the draft text
            for path in all_classes:
                if path in draft_text:
                    # Check if negated (e.g., "no atelectasis", "no cardiomegaly", "heart size is normal")
                    is_negated = False
                    negation_patterns = [
                        rf"no\s+{path}",
                        rf"without\s+{path}",
                        rf"negative\s+for\s+{path}",
                        rf"clear\s+of\s+{path}",
                        rf"free\s+of\s+{path}",
                        rf"normal\s+{path}"
                    ]
                    for pat in negation_patterns:
                        if re.search(pat, draft_text):
                            is_negated = True
                            break
                            
                    if path == "cardiomegaly" and "normal heart" in draft_text:
                        is_negated = True
                        
                    if not is_negated:
                        # Draft claims presence of pathology. Check if present in findings.
                        matched_finding = False
                        for p_un in pos_uncertain:
                            if p_un in path or path in p_un:
                                matched_finding = True
                                break
                        if not matched_finding:
                            hallucinations.append({
                                "claim": f"Asserted presence of {path}",
                                "explanation": f"Draft asserts {path} but it is not in the positive/uncertain findings."
                            })
            
            # Check for omitted findings
            for p_un in pos_uncertain:
                found = False
                for term in [p_un, p_un.split("/")[0]]:
                    if term in draft_text:
                        found = True
                        break
                if not found:
                    omissions.append({
                        "finding": p_un.capitalize(),
                        "explanation": f"Positive finding {p_un} was not mentioned in the draft report."
                    })
                    
            res = {
                "hallucinations": hallucinations,
                "omissions": omissions,
                "discrepancy_count": len(hallucinations) + len(omissions)
            }
            return json.dumps(res, indent=2)
            
        else:
            # Drafting request
            # Extract JSON findings
            findings_data = self._extract_json_findings(prompt)
            
            findings_list = []
            if findings_data and "findings" in findings_data:
                findings_list = findings_data["findings"]
                
            positives = []
            uncertains = []
            for f in findings_list:
                status = f.get("status")
                pathology = f.get("pathology")
                conf = f.get("confidence", 0.0)
                if status == "positive":
                    positives.append((pathology, conf))
                elif status == "uncertain":
                    uncertains.append((pathology, conf))
                    
            findings_text = []
            impressions = []
            recommendations = []
            
            citation_idx = 1
            for f in findings_list:
                status = f.get("status")
                pathology = f.get("pathology")
                loc = f.get("location")
                loc_phrase = f" in the {loc}" if loc else ""
                
                if status in ["positive", "uncertain"]:
                    hedge = "is suggestive of" if status == "uncertain" else "is noted"
                    if pathology.lower() == "cardiomegaly":
                        findings_text.append("- Cardiomegaly: Enlargement of the cardiac silhouette is identified.")
                        impressions.append("Enlarged cardiac silhouette (cardiomegaly).")
                        recommendations.append(f"{citation_idx}. Echocardiography is recommended to evaluate ventricular function [{citation_idx}].")
                        citation_idx += 1
                    elif pathology.lower() == "effusion":
                        findings_text.append(f"- Pleural Effusion: Blunting of the costophrenic angle {hedge}{loc_phrase}.")
                        impressions.append(f"Pleural effusion{loc_phrase}.")
                        recommendations.append(f"{citation_idx}. Clinical correlation and contrast chest CT or ultrasound is recommended to evaluate the pleural effusion [{citation_idx}].")
                        citation_idx += 1
                    elif pathology.lower() == "pneumonia":
                        findings_text.append(f"- Pneumonia: Focal airspace opacity{loc_phrase} {hedge} infectious process.")
                        impressions.append(f"Infectious consolidation suggestive of pneumonia{loc_phrase}.")
                        recommendations.append(f"{citation_idx}. Clinical correlation and follow-up chest radiography or CT without contrast is recommended [{citation_idx}].")
                        citation_idx += 1
                    elif pathology.lower() == "pneumothorax":
                        findings_text.append(f"- Pneumothorax: A visceral pleural line{loc_phrase} without peripheral lung markings {hedge}.")
                        impressions.append(f"Pneumothorax{loc_phrase}.")
                        recommendations.append(f"{citation_idx}. Conservative management with follow-up radiography or thoracostomy placement is recommended [{citation_idx}].")
                        citation_idx += 1
                    elif pathology.lower() == "atelectasis":
                        findings_text.append(f"- Atelectasis: Linear opacity{loc_phrase} {hedge} subsegmental atelectasis.")
                        impressions.append(f"Subsegmental atelectasis{loc_phrase}.")
                        recommendations.append(f"{citation_idx}. Follow-up radiography or chest CT with contrast is recommended for persistent atelectasis [{citation_idx}].")
                        citation_idx += 1
                    elif pathology.lower() == "consolidation":
                        findings_text.append(f"- Consolidation: Dense airspace consolidation{loc_phrase} {hedge}.")
                        impressions.append(f"Airspace consolidation{loc_phrase}.")
                        recommendations.append(f"{citation_idx}. Follow-up chest radiography in 4-6 weeks or chest CT is recommended to confirm clearing [{citation_idx}].")
                        citation_idx += 1
                    elif "nodule" in pathology.lower() or "mass" in pathology.lower():
                        findings_text.append(f"- Nodule/Mass: A circumscribed opacity{loc_phrase} {hedge}.")
                        impressions.append(f"Pulmonary nodule or mass{loc_phrase}.")
                        recommendations.append(f"{citation_idx}. Comparison with prior films and high-resolution chest CT without contrast is recommended [{citation_idx}].")
                        citation_idx += 1
            
            if not findings_text:
                findings_text.append("- Lungs are clear. No focal consolidation, pleural effusion, or pneumothorax identified.")
                findings_text.append("- Cardiac silhouette and mediastinal contours are within normal limits.")
                findings_text.append("- Visualized bony structures are intact.")
                impressions.append("No acute cardiopulmonary abnormality.")
                
            draft = (
                "CHEST X-RAY REPORT\n\n"
                "FINDINGS:\n"
                + "\n".join(findings_text)
                + "\n\nIMPRESSION:\n"
                + "\n".join([f"{i+1}. {imp}" for i, imp in enumerate(impressions)])
            )
            
            if recommendations:
                draft += "\n\nRECOMMENDATIONS:\n" + "\n".join(recommendations)
                
            return draft
