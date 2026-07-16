"""
graph.py — LangGraph state machine definition connecting drafting, verification, revision, and escalation.
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from radagent.schema import predictions_to_findings
from radagent.llm import LLMClient
from radagent.trace import AgentTracer
from radagent.agents.drafting import DraftingAgent
from radagent.agents.verification import VerificationAgent
from radagent.agents.grounding import GroundingAgent
from radagent.agents.escalation import EscalationAgent
from radagent.agents.bias import BiasAuditAgent

# ── State Definition ─────────────────────────────────────────────────────────

class RadAgentState(TypedDict):
    """The graph state tracking keys throughout the pipeline."""
    findings: Dict[str, Any]
    final_report: Optional[str]
    verification: Optional[Dict[str, Any]]
    grounding: Optional[List[Dict[str, Any]]]
    escalated: Optional[bool]
    trace_id: Optional[str]
    demographics: Optional[Dict[str, Any]]
    bias: Optional[Dict[str, Any]]
    clinical_focus: Optional[str]
    
    # Internal routing & auditing state
    current_draft: Optional[str]
    discrepancy_count: Optional[int]
    revision_count: int
    feedback: Optional[str]
    steps: List[Dict[str, Any]]  # Records the trace logs

# ── Node Definitions ─────────────────────────────────────────────────────────

def build_nodes(llm_client: LLMClient):
    """Factory function to build node closures with shared LLM client."""
    draft_agent = DraftingAgent(llm_client)
    verify_agent = VerificationAgent(llm_client)
    grounding_agent = GroundingAgent()
    escalation_agent = EscalationAgent()
    bias_agent = BiasAuditAgent()
    
    def draft_node(state: RadAgentState) -> Dict[str, Any]:
        """Node for drafting or revising the radiology report."""
        findings = state["findings"]
        feedback = state.get("feedback")
        clinical_focus = state.get("clinical_focus")
        
        draft_text = draft_agent.draft(findings, feedback=feedback, clinical_focus=clinical_focus)
        
        step_log = {
            "node": "drafting",
            "revision": state["revision_count"],
            "feedback_applied": feedback is not None,
            "output_draft": draft_text
        }
        
        return {
            "current_draft": draft_text,
            "revision_count": state["revision_count"] + 1,
            "steps": state.get("steps", []) + [step_log]
        }
        
    def verify_node(state: RadAgentState) -> Dict[str, Any]:
        """Node for verifying the drafted report against findings."""
        findings = state["findings"]
        draft_text = state["current_draft"]
        
        verify_result = verify_agent.verify(findings, draft_text)
        discrepancy_count = verify_result.get("discrepancy_count", 0)
        
        # Compile revision feedback if discrepancies are present
        feedback = None
        if discrepancy_count > 0:
            feedback_parts = []
            if verify_result.get("hallucinations"):
                feedback_parts.append("Hallucinations detected: " + "; ".join(
                    [h["claim"] + " (" + h["explanation"] + ")" for h in verify_result["hallucinations"]]
                ))
            if verify_result.get("omissions"):
                feedback_parts.append("Omitted findings: " + "; ".join(
                    [o["finding"] + " (" + o["explanation"] + ")" for o in verify_result["omissions"]]
                ))
            feedback = "\n".join(feedback_parts)
            
        step_log = {
            "node": "verification",
            "discrepancies": verify_result,
            "discrepancy_count": discrepancy_count,
            "feedback_generated": feedback
        }
        
        return {
            "verification": verify_result,
            "discrepancy_count": discrepancy_count,
            "feedback": feedback,
            "steps": state.get("steps", []) + [step_log]
        }
        
    def ground_node(state: RadAgentState) -> Dict[str, Any]:
        """Node for grounding recommendations using RAG guidelines."""
        findings = state["findings"]
        current_report = state["current_draft"]
        
        res = grounding_agent.ground(findings, current_report)
        
        step_log = {
            "node": "grounding",
            "grounded_recommendations": len(res.get("grounding", []))
        }
        
        return {
            "grounding": res.get("grounding", []),
            "steps": state.get("steps", []) + [step_log]
        }
        
    def bias_node(state: RadAgentState) -> Dict[str, Any]:
        """Node for auditing demographic disparities."""
        demographics = state.get("demographics")
        bias_res = bias_agent.audit(demographics)
        
        step_log = {
            "node": "bias_audit",
            "bias_warning_triggered": bias_res["bias_disparity_detected"],
            "subgroup": bias_res["bias_subgroup"]
        }
        
        return {
            "bias": bias_res,
            "steps": state.get("steps", []) + [step_log]
        }
        
    def escalate_node(state: RadAgentState) -> Dict[str, Any]:
        """Node for evaluating escalation triggers and finalizing report state."""
        eval_res = escalation_agent.evaluate(state)
        
        step_log = {
            "node": "escalation_eval",
            "escalated": eval_res["escalated"],
            "reasons": eval_res["escalation_reasons"]
        }
        
        return {
            "escalated": eval_res["escalated"],
            "final_report": state["current_draft"],
            "steps": state.get("steps", []) + [step_log]
        }
        
    return draft_node, verify_node, ground_node, bias_node, escalate_node

# ── Router Definition ────────────────────────────────────────────────────────

def route_verification(state: RadAgentState) -> str:
    """Routes execution based on audit discrepancies and loop iteration limits."""
    # If we have discrepancies, check if we've already revised once
    # revision_count starts at 0, first draft node sets it to 1
    if state["discrepancy_count"] > 0 and state["revision_count"] < 2:
        print(f"[Router] Routing to redraft: {state['discrepancy_count']} discrepancies found.")
        return "redraft"
        
    print("[Router] Routing to ground: verification check complete.")
    return "ground"

# ── Graph Compilation ────────────────────────────────────────────────────────

def create_graph(llm_client: Optional[LLMClient] = None):
    """Compiles the StateGraph for the RadAgent reporting pipeline."""
    llm = llm_client or LLMClient()
    draft_node, verify_node, ground_node, bias_node, escalate_node = build_nodes(llm)
    
    workflow = StateGraph(RadAgentState)
    
    # Register Nodes
    workflow.add_node("draft_report", draft_node)
    workflow.add_node("verify_report", verify_node)
    workflow.add_node("ground_report", ground_node)
    workflow.add_node("bias_report", bias_node)
    workflow.add_node("escalate_report", escalate_node)
    
    # Establish Edges
    workflow.set_entry_point("draft_report")
    workflow.add_edge("draft_report", "verify_report")
    
    # Conditional route after verification
    workflow.add_conditional_edges(
        "verify_report",
        route_verification,
        {
            "redraft": "draft_report",
            "ground": "ground_report"
        }
    )
    
    workflow.add_edge("ground_report", "bias_report")
    workflow.add_edge("bias_report", "escalate_report")
    workflow.add_edge("escalate_report", END)
    
    return workflow.compile()
