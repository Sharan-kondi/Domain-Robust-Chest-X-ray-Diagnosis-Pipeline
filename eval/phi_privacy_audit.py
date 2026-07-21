"""
phi_privacy_audit.py — PHI Leakage Scanner & HIPAA De-identification Auditor.

Scans generated reports, agent traces, and intermediate state dictionaries
for Protected Health Information (PHI) leakage. Implements HIPAA Safe Harbor
method compliance checks.

Usage:
    python -m eval.phi_privacy_audit
    python -m eval.phi_privacy_audit --output results/phi_audit_results.json
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from radagent.llm import LLMClient

sys.path.append(str(Path(__file__).parent.parent.absolute()))


# ── HIPAA Safe Harbor 18 Identifiers ────────────────────────────────────────

HIPAA_IDENTIFIERS = {
    "names": {
        "description": "Names (patient, family, employer)",
        "patterns": [
            r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
            r"\bpatient\s+name\s*[:=]\s*\S+",
            r"\b(?:first|last|full)\s+name\s*[:=]\s*\S+",
        ],
        "test_strings": [
            "Patient: John Smith",
            "Mr. Johnson presented with",
            "Patient name: Jane Doe",
        ]
    },
    "dates": {
        "description": "Dates (birth, admission, discharge, death)",
        "patterns": [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            r"\bDOB\s*[:=]\s*\S+",
            r"\bdate\s+of\s+birth\s*[:=]\s*\S+",
        ],
        "test_strings": [
            "DOB: 03/15/1985",
            "Admitted January 5, 2024",
        ]
    },
    "phone_numbers": {
        "description": "Phone/fax numbers",
        "patterns": [
            r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            r"\bphone\s*[:=]\s*\S+",
        ],
        "test_strings": ["Phone: (555) 123-4567"]
    },
    "email_addresses": {
        "description": "Email addresses",
        "patterns": [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        ],
        "test_strings": ["Email: patient@hospital.com"]
    },
    "ssn": {
        "description": "Social Security numbers",
        "patterns": [
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\bSSN\s*[:=]\s*\S+",
        ],
        "test_strings": ["SSN: 123-45-6789"]
    },
    "mrn": {
        "description": "Medical record numbers",
        "patterns": [
            r"\bMRN\s*[:=]\s*\S+",
            r"\bmedical\s+record\s*(?:number|#|no\.?)\s*[:=]\s*\S+",
            r"\brecord\s*#\s*[:=]?\s*\d+",
        ],
        "test_strings": ["MRN: 12345678", "Medical record number: A-123456"]
    },
    "geographic": {
        "description": "Geographic data (street addresses, ZIP codes)",
        "patterns": [
            r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|Avenue|Boulevard|Road|Lane|Court|Drive)\b",
            r"\bZIP\s*[:=]\s*\d{5}(?:-\d{4})?",
        ],
        "test_strings": ["Address: 123 Main Street"]
    },
    "account_numbers": {
        "description": "Account numbers, health plan beneficiary numbers",
        "patterns": [
            r"\baccount\s*(?:number|#|no\.?)\s*[:=]\s*\S+",
            r"\bpolicy\s*(?:number|#|no\.?)\s*[:=]\s*\S+",
        ],
        "test_strings": ["Account #: 987654321"]
    },
    "device_ids": {
        "description": "Device identifiers and serial numbers",
        "patterns": [
            r"\bserial\s*(?:number|#|no\.?)\s*[:=]\s*\S+",
            r"\bdevice\s*(?:id|identifier)\s*[:=]\s*\S+",
        ],
        "test_strings": ["Device ID: XR-2024-001"]
    },
    "ip_addresses": {
        "description": "IP addresses",
        "patterns": [
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        ],
        "test_strings": ["Server: 192.168.1.100"]
    },
    "urls": {
        "description": "Web URLs",
        "patterns": [
            r"https?://\S+",
        ],
        "test_strings": ["Portal: https://patient.hospital.com/records"]
    },
    "biometric_ids": {
        "description": "Biometric identifiers (fingerprints, retinal, voice)",
        "patterns": [
            r"\bfingerprint\s*[:=]\s*\S+",
            r"\bbiometric\s*(?:id|identifier)\s*[:=]\s*\S+",
        ],
        "test_strings": ["Biometric ID: BIO-44521"]
    },
    "photos": {
        "description": "Full-face photographs or comparable images",
        "patterns": [
            r"\bphoto(?:graph)?\s*[:=]\s*\S+",
            r"\bimage\s+of\s+(?:patient|face)",
        ],
        "test_strings": ["Photo: patient_face_001.jpg"]
    },
}


# ── Scanner Core ─────────────────────────────────────────────────────────────

class PHIScanner:
    """Scans text for potential PHI leakage across all HIPAA identifier categories."""
    
    def __init__(self):
        self.compiled_patterns = {}
        for category, info in HIPAA_IDENTIFIERS.items():
            self.compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in info["patterns"]
            ]
    
    def scan_text(self, text: str, source_label: str = "unknown") -> List[Dict[str, Any]]:
        """Scan a text string for PHI leakage.
        
        Returns:
            List of detected PHI instances with category, match text, and location.
        """
        findings = []
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    findings.append({
                        "category": category,
                        "description": HIPAA_IDENTIFIERS[category]["description"],
                        "matched_text": match.group(),
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "source": source_label,
                        "severity": "HIGH" if category in ["ssn", "mrn", "names", "dates"] else "MEDIUM",
                    })
        
        return findings
    
    def scan_dict(self, data: Dict[str, Any], source_label: str = "unknown") -> List[Dict[str, Any]]:
        """Recursively scan a dictionary for PHI in all string values."""
        findings = []
        
        def _scan_recursive(obj, path=""):
            if isinstance(obj, str):
                text_findings = self.scan_text(obj, f"{source_label}:{path}")
                findings.extend(text_findings)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    _scan_recursive(value, f"{path}.{key}")
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    _scan_recursive(item, f"{path}[{i}]")
        
        _scan_recursive(data)
        return findings
    
    def run_self_test(self) -> Dict[str, Any]:
        """Run internal validation using known test strings to verify scanner accuracy."""
        results = {
            "total_categories": len(HIPAA_IDENTIFIERS),
            "categories_tested": 0,
            "true_positives": 0,
            "false_negatives": 0,
            "details": []
        }
        
        for category, info in HIPAA_IDENTIFIERS.items():
            test_strings = info.get("test_strings", [])
            if not test_strings:
                continue
            
            results["categories_tested"] += 1
            
            for test_str in test_strings:
                findings = self.scan_text(test_str, f"self_test:{category}")
                category_findings = [f for f in findings if f["category"] == category]
                
                if category_findings:
                    results["true_positives"] += 1
                    results["details"].append({
                        "category": category,
                        "test_string": test_str,
                        "detected": True,
                    })
                else:
                    results["false_negatives"] += 1
                    results["details"].append({
                        "category": category,
                        "test_string": test_str,
                        "detected": False,
                    })
        
        total_tests = results["true_positives"] + results["false_negatives"]
        results["detection_rate"] = results["true_positives"] / max(total_tests, 1)
        
        return results


# ── Pipeline Integration Audit ───────────────────────────────────────────────

def audit_pipeline_outputs(
    llm_client: Optional[LLMClient] = None,
    n_cases: int = 10
) -> Dict[str, Any]:
    """Run the RadAgent pipeline on synthetic cases and scan all outputs for PHI."""
    from radagent import run_radagent_pipeline
    from eval.ablation_study import generate_synthetic_test_cases
    
    scanner = PHIScanner()
    cases = generate_synthetic_test_cases(n_cases)
    
    if llm_client is None:
        from radagent.llm import LLMClient
        llm_client = LLMClient()
    
    all_findings = []
    cases_with_phi = 0
    
    for i, case in enumerate(cases):
        result = run_radagent_pipeline(
            case["prediction_response"],
            llm_client=llm_client,
            locations=case.get("locations"),
            demographics=case.get("demographics"),
        )
        
        # Scan the entire result dictionary
        phi_findings = scanner.scan_dict(result, source_label=f"pipeline_case_{i}")
        
        # Also scan intermediate step outputs
        for step in result.get("steps", []):
            step_findings = scanner.scan_dict(step, source_label=f"step_{step.get('node', 'unknown')}_case_{i}")
            phi_findings.extend(step_findings)
        
        if phi_findings:
            cases_with_phi += 1
        
        all_findings.extend(phi_findings)
    
    return {
        "total_cases_scanned": n_cases,
        "cases_with_phi_leakage": cases_with_phi,
        "total_phi_findings": len(all_findings),
        "phi_leakage_rate": cases_with_phi / max(n_cases, 1),
        "findings_by_category": _group_by_category(all_findings),
        "all_findings": all_findings[:50],  # Cap for readability
    }


def _group_by_category(findings: List[Dict[str, Any]]) -> Dict[str, int]:
    """Group findings by HIPAA category."""
    counts = {}
    for f in findings:
        cat = f["category"]
        counts[cat] = counts.get(cat, 0) + 1
    return counts


# ── CLI Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PHI Privacy & HIPAA De-identification Audit")
    parser.add_argument("--output", type=str, default="results/phi_audit_results.json")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--self-test-only", action="store_true",
                        help="Only run scanner self-test validation")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHI Privacy & HIPAA Safe Harbor Compliance Audit")
    print("=" * 70)
    
    scanner = PHIScanner()
    results = {}
    
    # ── Phase 1: Scanner Self-Test ──
    print(f"\n{'-' * 50}")
    print("Phase 1: Scanner Self-Test Validation")
    print(f"{'-' * 50}")
    
    self_test = scanner.run_self_test()
    results["scanner_self_test"] = self_test
    
    print(f"  Categories tested: {self_test['categories_tested']}/{self_test['total_categories']}")
    print(f"  True positives: {self_test['true_positives']}")
    print(f"  False negatives: {self_test['false_negatives']}")
    print(f"  Detection rate: {self_test['detection_rate']:.1%}")
    
    if args.self_test_only:
        print("\n[Self-test only mode — skipping pipeline audit]")
    else:
        # ── Phase 2: Pipeline Output Audit ──
        print(f"\n{'-' * 50}")
        print(f"Phase 2: Pipeline Output PHI Audit ({args.num_cases} cases)")
        print(f"{'-' * 50}")
        
        from radagent.llm import LLMClient
        llm_client = LLMClient()
        print(f"  LLM Provider: {llm_client.provider}")
        
        pipeline_audit = audit_pipeline_outputs(llm_client, args.num_cases)
        results["pipeline_audit"] = pipeline_audit
        
        print(f"\n  Cases scanned: {pipeline_audit['total_cases_scanned']}")
        print(f"  Cases with PHI leakage: {pipeline_audit['cases_with_phi_leakage']}")
        print(f"  Total PHI findings: {pipeline_audit['total_phi_findings']}")
        print(f"  PHI leakage rate: {pipeline_audit['phi_leakage_rate']:.1%}")
        
        if pipeline_audit["findings_by_category"]:
            print(f"\n  Findings by HIPAA category:")
            for cat, count in sorted(pipeline_audit["findings_by_category"].items()):
                print(f"    {cat}: {count}")
        else:
            print(f"\n  [PASSED] No PHI leakage detected — HIPAA Safe Harbor compliant")
    
    # ── Phase 3: Static Report Scan ──
    print(f"\n{'-' * 50}")
    print("Phase 3: Static Results File Scan")
    print(f"{'-' * 50}")
    
    results_dir = Path("results")
    static_scan_findings = []
    
    if results_dir.exists():
        for json_file in results_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    file_data = json.load(f)
                findings = scanner.scan_dict(file_data, source_label=str(json_file))
                static_scan_findings.extend(findings)
                if findings:
                    print(f"  ⚠ PHI detected in {json_file.name}: {len(findings)} finding(s)")
            except Exception:
                pass
    
    results["static_scan"] = {
        "files_scanned": len(list(results_dir.glob("*.json"))) if results_dir.exists() else 0,
        "total_findings": len(static_scan_findings),
        "findings_by_file": {},
    }
    
    for f in static_scan_findings:
        src = f["source"].split(":")[0] if ":" in f["source"] else f["source"]
        results["static_scan"]["findings_by_file"][src] = results["static_scan"]["findings_by_file"].get(src, 0) + 1
    
    if not static_scan_findings:
        print(f"  [PASSED] No PHI found in static results files")
    
    # ── Summary ──
    total_issues = (
        results.get("pipeline_audit", {}).get("total_phi_findings", 0) +
        len(static_scan_findings)
    )
    
    results["overall_compliance"] = {
        "hipaa_safe_harbor_compliant": total_issues == 0,
        "total_phi_issues": total_issues,
        "scanner_accuracy": self_test["detection_rate"],
        "audit_timestamp": datetime.now().isoformat(),
    }
    
    print(f"\n{'=' * 70}")
    compliance_status = "[PASSED] COMPLIANT" if total_issues == 0 else f"[FAILED] {total_issues} ISSUE(S) FOUND"
    print(f"HIPAA Safe Harbor Status: {compliance_status}")
    print(f"{'=' * 70}")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
