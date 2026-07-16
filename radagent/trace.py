"""
trace.py — Structured logging of every agent node execution step.
"""

import time
import uuid
from typing import Dict, Any, List, Optional

class AgentTracer:
    """Tracer that tracks inputs, outputs, and metadata for each node execution."""
    
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.steps: List[Dict[str, Any]] = []

    def log_step(self, node_name: str, inputs: Any, outputs: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record the input, output, and parameters of a single agent node execution."""
        step_log = {
            "timestamp": time.time(),
            "node": node_name,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {}
        }
        self.steps.append(step_log)
        print(f"[AgentTracer] Logged step: {node_name}")

    def get_trace(self) -> Dict[str, Any]:
        """Compile and return the complete trace history."""
        return {
            "trace_id": self.trace_id,
            "steps": self.steps
        }
