"""
radagent — Unified Domain-Robust Radiology AI Agent System.
"""

from typing import Dict, Any, Optional

from radagent.schema import predictions_to_findings
from radagent.llm import LLMClient
from radagent.trace import AgentTracer
from radagent.graph import create_graph

__all__ = [
    "predictions_to_findings",
    "LLMClient",
    "AgentTracer",
    "create_graph",
    "run_radagent_pipeline",
]

def run_radagent_pipeline(
    predictions: Any, 
    llm_client: Optional[LLMClient] = None, 
    locations: Optional[Dict[str, str]] = None,
    demographics: Optional[Dict[str, Any]] = None,
    clinical_focus: Optional[str] = None
) -> Dict[str, Any]:
    """Runs the full RadAgent report generation, verification, and correction pipeline.
    
    Args:
        predictions: PredictionResponse object or prediction response dictionary from the model.
        llm_client: Optional LLMClient to override default configuration.
        locations: Optional dict mapping pathology labels to spatial quadrants.
        demographics: Optional dict containing patient demographics (age, gender).
        clinical_focus: Optional custom clinical target/focus directive from the clinician.
        
    Returns:
        Dict representing final state of the pipeline:
        {
            "findings": Dict[str, Any],
            "final_report": str,
            "verification": Dict[str, Any],
            "grounding": List[Dict[str, Any]],
            "escalated": bool,
            "trace_id": str,
            "steps": List[Dict[str, Any]],
            "trace": Dict[str, Any]
        }
    """
    findings = predictions_to_findings(predictions, locations=locations)
    graph = create_graph(llm_client)
    
    tracer = AgentTracer()
    
    initial_state = {
        "findings": findings,
        "final_report": None,
        "verification": None,
        "grounding": None,
        "escalated": None,
        "trace_id": tracer.trace_id,
        "current_draft": None,
        "discrepancy_count": None,
        "revision_count": 0,
        "feedback": None,
        "steps": [],
        "demographics": demographics,
        "bias": None,
        "clinical_focus": clinical_focus
    }
    
    final_state = graph.invoke(initial_state)
    
    # Save trace logging structure
    for step in final_state.get("steps", []):
        tracer.log_step(
            node_name=step["node"],
            inputs={"revision": step.get("revision")},
            outputs=step,
            metadata={"model": getattr(llm_client or LLMClient(), "model_name", "unknown")}
        )
        
    final_state["trace"] = tracer.get_trace()
    return final_state
