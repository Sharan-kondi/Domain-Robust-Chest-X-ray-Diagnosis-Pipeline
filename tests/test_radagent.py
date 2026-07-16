"""
test_radagent.py — Unit tests for the RadAgent report generation and verification pipeline.
"""

import os
import json
import pytest
from radagent.schema import predictions_to_findings
from radagent.llm import LLMClient
from radagent.agents.drafting import DraftingAgent
from radagent.agents.verification import VerificationAgent
from radagent.graph import create_graph, route_verification
from radagent import run_radagent_pipeline

# Sample mock prediction responses
@pytest.fixture
def sample_prediction():
    return {
        "predictions": [
            {"label": "Effusion", "probability": 0.82, "positive": True},
            {"label": "Atelectasis", "probability": 0.55, "positive": False},
            {"label": "Pneumothorax", "probability": 0.12, "positive": False},
            {"label": "No Finding", "probability": 0.18, "positive": False}
        ],
        "uncertainty": 2.1,
        "needs_human_review": False,
        "gradcam_available": False
    }

@pytest.fixture
def high_uncertainty_prediction():
    return {
        "predictions": [
            {"label": "Cardiomegaly", "probability": 0.58, "positive": False},
            {"label": "No Finding", "probability": 0.42, "positive": False}
        ],
        "uncertainty": 5.2,  # Over the 4.94 threshold
        "needs_human_review": True,
        "gradcam_available": False
    }

def test_predictions_to_findings(sample_prediction):
    """Verify that predictions are correctly categorized into positive/uncertain/negative."""
    findings = predictions_to_findings(sample_prediction)
    
    # 8-class model has 8 - 1 (No Finding) = 7 pathologies in findings
    assert len(findings["findings"]) == 3
    
    # Check status mapping
    effusion_finding = next(f for f in findings["findings"] if f["pathology"] == "Effusion")
    assert effusion_finding["status"] == "positive"
    assert effusion_finding["confidence"] == 0.82
    
    atelectasis_finding = next(f for f in findings["findings"] if f["pathology"] == "Atelectasis")
    assert atelectasis_finding["status"] == "uncertain"
    assert atelectasis_finding["confidence"] == 0.55
    
    pneumo_finding = next(f for f in findings["findings"] if f["pathology"] == "Pneumothorax")
    assert pneumo_finding["status"] == "negative"
    assert pneumo_finding["confidence"] == 0.12

def test_mock_llm_drafting():
    """Verify that the Mock LLM outputs appropriate phrasing for findings."""
    llm_client = LLMClient()
    llm_client.provider = "mock"
    
    drafting_agent = DraftingAgent(llm_client)
    findings = {
        "findings": [
            {"pathology": "Effusion", "status": "positive", "confidence": 0.85},
            {"pathology": "Atelectasis", "status": "uncertain", "confidence": 0.58}
        ]
    }
    
    draft = drafting_agent.draft(findings)
    assert "FINDINGS:" in draft
    assert "IMPRESSION:" in draft
    assert "pleural effusion" in draft.lower()
    assert "suggestive of subsegmental atelectasis" in draft.lower()

def test_mock_llm_verification_clean():
    """Verify that a perfect match draft produces 0 discrepancies."""
    llm_client = LLMClient()
    llm_client.provider = "mock"
    
    verify_agent = VerificationAgent(llm_client)
    findings = {
        "findings": [
            {"pathology": "Effusion", "status": "positive", "confidence": 0.85},
            {"pathology": "Atelectasis", "status": "uncertain", "confidence": 0.58}
        ]
    }
    
    # Perfect report matching findings
    draft = (
        "CHEST X-RAY REPORT\n\n"
        "FINDINGS:\n"
        "- Pleural Effusion: Blunting of the costophrenic angle is noted.\n"
        "- Atelectasis: Linear opacity is suggestive of subsegmental atelectasis.\n\n"
        "IMPRESSION:\n"
        "1. Pleural effusion.\n"
        "2. Subsegmental atelectasis."
    )
    
    report = verify_agent.verify(findings, draft)
    assert report["discrepancy_count"] == 0
    assert len(report["hallucinations"]) == 0
    assert len(report["omissions"]) == 0

def test_mock_llm_verification_with_discrepancies():
    """Verify that omissions and hallucinations are identified."""
    llm_client = LLMClient()
    llm_client.provider = "mock"
    
    verify_agent = VerificationAgent(llm_client)
    findings = {
        "findings": [
            {"pathology": "Effusion", "status": "positive", "confidence": 0.85},
            {"pathology": "Atelectasis", "status": "uncertain", "confidence": 0.58}
        ]
    }
    
    # Hallucination: Pneumothorax (asserted without being in findings)
    # Omission: Atelectasis (positive/uncertain in findings but omitted from draft)
    draft = "FINDINGS:\n- Pleural Effusion is noted.\n- Active pneumothorax is seen.\n\nIMPRESSION:\nEffusion and pneumothorax."
    
    report = verify_agent.verify(findings, draft)
    assert report["discrepancy_count"] > 0
    
    # Assert hallucination caught
    hallucination_labels = [h["claim"].lower() for h in report["hallucinations"]]
    assert any("pneumothorax" in label for label in hallucination_labels)
    
    # Assert omission caught
    omission_labels = [o["finding"].lower() for o in report["omissions"]]
    assert any("atelectasis" in label for label in omission_labels)

def test_route_verification_uncertainty():
    """Verify that routing goes to ground if needs_human_review is True (escalation evaluated there)."""
    state = {
        "findings": {"needs_human_review": True},
        "discrepancy_count": 0,
        "revision_count": 0
    }
    assert route_verification(state) == "ground"

def test_route_verification_flow():
    """Verify that routing loops back to redraft once, then exits to ground if errors persist."""
    # Scenario A: No discrepancies -> ground
    state_a = {
        "findings": {"needs_human_review": False},
        "discrepancy_count": 0,
        "revision_count": 1
    }
    assert route_verification(state_a) == "ground"
    
    # Scenario B: Discrepancies first time -> redraft
    state_b = {
        "findings": {"needs_human_review": False},
        "discrepancy_count": 2,
        "revision_count": 1  # 1 means draft node ran once (first draft)
    }
    assert route_verification(state_b) == "redraft"
    
    # Scenario C: Discrepancies second time (already revised once) -> ground (loop limit reached)
    state_c = {
        "findings": {"needs_human_review": False},
        "discrepancy_count": 1,
        "revision_count": 2  # 2 means draft node ran twice (draft + revision)
    }
    assert route_verification(state_c) == "ground"

def test_run_radagent_pipeline(sample_prediction):
    """Verify that the full state machine pipeline runs end-to-end with mock LLM."""
    llm_client = LLMClient()
    llm_client.provider = "mock"
    
    result = run_radagent_pipeline(sample_prediction, llm_client)
    
    assert "final_report" in result
    assert "verification" in result
    assert "escalated" in result
    assert "trace" in result
    assert result["escalated"] is False  # Perfect mock generation doesn't escalate
    assert len(result["steps"]) > 0

def test_spatial_verification():
    """Verify that a spatial mismatch is detected between findings location and report text."""
    llm_client = LLMClient()
    llm_client.provider = "mock"
    verify_agent = VerificationAgent(llm_client)
    
    # Finding says right lower zone, but draft says left-sided
    findings = {
        "findings": [
            {"pathology": "Effusion", "status": "positive", "confidence": 0.85, "location": "right lower zone"}
        ]
    }
    
    draft = "CHEST X-RAY REPORT\n\nFINDINGS:\n- Pleural effusion is noted on the left side.\n\nIMPRESSION:\n1. Left-sided pleural effusion."
    
    report = verify_agent.verify(findings, draft)
    assert report["discrepancy_count"] > 0
    hallucination_explanations = [h["explanation"].lower() for h in report["hallucinations"]]
    assert any("spatial mismatch" in exp for exp in hallucination_explanations)

def test_grounding_agent():
    """Verify that the GroundingAgent retrieves ACR guidelines for positive findings."""
    from radagent.agents.grounding import GroundingAgent
    grounding_agent = GroundingAgent()
    
    findings = {
        "findings": [
            {"pathology": "Cardiomegaly", "status": "positive", "confidence": 0.82}
        ]
    }
    
    res = grounding_agent.ground(findings, "CHEST X-RAY REPORT...")
    citations = res["grounding"]
    assert len(citations) == 1
    assert citations[0]["acr_code"] == "ACR-AC-2"
    assert "Echocardiography" in citations[0]["recommendation"]

def test_demographic_bias_audit():
    """Verify that the BiasAuditAgent flags performance disparities for the 70+ cohort."""
    from radagent.agents.bias import BiasAuditAgent
    bias_agent = BiasAuditAgent()
    
    # Under 70 -> no warning
    res_young = bias_agent.audit({"age": 45, "gender": "F"})
    assert res_young["bias_disparity_detected"] is False
    
    # 70+ -> warning triggered
    res_old = bias_agent.audit({"age": 72, "gender": "M"})
    assert res_old["bias_disparity_detected"] is True
    assert "AUC 0.449" in res_old["bias_warning"]

def test_run_radagent_pipeline_with_bias(sample_prediction):
    """Verify that demographic bias warnings enforce escalation in the pipeline."""
    llm_client = LLMClient()
    llm_client.provider = "mock"
    
    # Run with young patient
    res_young = run_radagent_pipeline(sample_prediction, llm_client, demographics={"age": 45})
    assert res_young["escalated"] is False
    
    # Run with elderly patient
    res_elderly = run_radagent_pipeline(sample_prediction, llm_client, demographics={"age": 75})
    assert res_elderly["escalated"] is True
    assert any("Demographic performance" in reason for reason in res_elderly["steps"][-1]["reasons"])
