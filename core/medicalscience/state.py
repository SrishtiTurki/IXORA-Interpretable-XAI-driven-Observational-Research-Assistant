# core/medicalscience/state.py - UPDATED WITH module_usage
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

@dataclass
class ExecutionState:
    query: str
    session_id: str = ""
    domain: str = "biomed"
    
    intent: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    retrieved_evidence: List[Dict] = field(default_factory=list)
    scored_evidence: List[Dict] = field(default_factory=list)
    hypothesis: str = ""
    structured_analysis: Dict[str, Any] = field(default_factory=dict)
    final_response: str = ""
    
    confidence: float = 0.0
    step_times: Dict[str, float] = field(default_factory=dict)
    tokens_used: Dict[str, int] = field(default_factory=dict)
    
    trace: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    
    # NEW: RLHF-ready tracking
    module_usage: List[str] = field(default_factory=list)
    
    start_time: Optional[datetime] = None
    max_total_time: float = 180.0
    
    def add_trace(self, step: str, output: Any, metadata: Dict = None):
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "output": output,
            "elapsed_seconds": self._get_elapsed(),
            "metadata": metadata or {}
        }
        self.trace.append(entry)
        
        # Auto-track analytics modules
        if step.startswith("analytics_"):
            module = step.replace("analytics_", "")
            if module not in self.module_usage:
                self.module_usage.append(module)
    
    def add_error(self, step: str, error: str, fallback_used: bool = False):
        self.errors.append({
            "step": step,
            "error": str(error)[:200],
            "fallback_used": fallback_used,
            "timestamp": datetime.now().isoformat()
        })
    
    def _get_elapsed(self) -> float:
        if not self.start_time:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()
    
    def time_remaining(self) -> float:
        return max(0.0, self.max_total_time - self._get_elapsed())
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query[:100],                  
            "session_id": self.session_id,
            "domain": self.domain,
            "intent": self.intent,
            "confidence": self.confidence,
            "final_response": self.final_response[:500],
            "trace": self.trace,
            "module_usage": self.module_usage,
            "step_times": self.step_times,
            "errors": self.errors,
            "total_time": self._get_elapsed()
        }