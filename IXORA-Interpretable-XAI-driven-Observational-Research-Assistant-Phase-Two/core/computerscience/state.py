# core/computerscience/state.py - State management for computer science pipeline
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum, auto
import logging
from datetime import datetime
import json
from typing import Type, Dict, Any
from core.config import CS_DOMAINS

logger = logging.getLogger("cs.pipeline")

def create_domain_enum() -> Type[Enum]:
    """Dynamically create CSDomain enum from config"""
    return Enum(
        'CSDomain',
        {k: v for k, v in CS_DOMAINS.items()},
        type=str,
        module=__name__
    )

# Create the dynamic enum
CSDomain = create_domain_enum()

class AnalysisType(str, Enum):
    """Types of analysis that can be performed"""
    THEORETICAL = "theoretical"
    PRACTICAL = "practical"
    HYBRID = "hybrid"

@dataclass
class TheoreticalFramework:
    """Represents a theoretical framework or model"""
    name: str
    description: str
    complexity_class: Optional[str] = None
    key_papers: List[Dict[str, str]] = field(default_factory=list)
    open_problems: List[str] = field(default_factory=list)

@dataclass
class ExecutionState:
    """Tracks the state of a computer science query execution"""
    query: str
    session_id: str = ""
    domain: str = "computerscience"
    
    # Core processing state
    intent: str = ""
    analysis_type: AnalysisType = AnalysisType.HYBRID
    domains: Set[CSDomain] = field(default_factory=set)
    
    # Theoretical components
    theoretical_frameworks: List[TheoreticalFramework] = field(default_factory=list)
    formal_definitions: List[Dict] = field(default_factory=list)
    theorems: List[Dict] = field(default_factory=list)
    proofs: List[Dict] = field(default_factory=list)
    
    # Practical components
    code_artifacts: List[Dict] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Common state
    parameters: Dict[str, Any] = field(default_factory=dict)
    retrieved_evidence: List[Dict] = field(default_factory=list)
    scored_evidence: List[Dict] = field(default_factory=list)
    
    # Analysis results
    complexity_analysis: Dict[str, Any] = field(default_factory=dict)
    implications: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_domain(self, domain: Union[CSDomain, str]) -> None:
        """Add a CS domain to the analysis"""
        if isinstance(domain, str):
            domain = CSDomain(domain.lower())
        self.domains.add(domain)
        
        # Update analysis type based on domains
        theoretical_domains = {
            CSDomain.COMPLEXITY, 
            CSDomain.COMPUTABILITY,
            CSDomain.FORMAL_LANG
        }
        practical_domains = {
            CSDomain.SOFTWARE_ENG,
            CSDomain.PRAC_CODING
        }
        
        if self.domains & theoretical_domeworks and self.domains & practical_domains:
            self.analysis_type = AnalysisType.HYBRID
        elif self.domains & theoretical_domains:
            self.analysis_type = AnalysisType.THEORETICAL
        else:
            self.analysis_type = AnalysisType.PRACTICAL
    
    def add_theoretical_framework(self, name: str, description: str, 
                               complexity_class: str = None) -> None:
        """Add a theoretical framework to the state"""
        framework = TheoreticalFramework(
            name=name,
            description=description,
            complexity_class=complexity_class
        )
        self.theoretical_frameworks.append(framework)
        self.add_domain(CSDomain.COMPLEXITY)
    
    def add_code_artifact(self, name: str, content: str, 
                        language: str = "python") -> None:
        """Add a code artifact to the state"""
        self.code_artifacts.append({
            "name": name,
            "content": content,
            "language": language,
            "created_at": datetime.utcnow().isoformat()
        })
        self.add_domain(CSDomain.PRAC_CODING)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "query": self.query,
            "session_id": self.session_id,
            "domain": self.domain,
            "analysis_type": self.analysis_type.value,
            "domains": [d.value for d in self.domains],
            "theoretical_frameworks": [
                {"name": f.name, "description": f.description, 
                 "complexity_class": f.complexity_class}
                for f in self.theoretical_frameworks
            ],
            "code_artifacts": self.code_artifacts,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat()
        }
    hypothesis: str = ""
    structured_analysis: Dict[str, Any] = field(default_factory=dict)
    final_response: str = ""
    
    # Performance metrics
    confidence: float = 0.0
    step_times: Dict[str, float] = field(default_factory=dict)
    tokens_used: Dict[str, int] = field(default_factory=dict)
    
    # Debugging and trace
    trace: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    
    # RLHF and module tracking
    module_usage: List[str] = field(default_factory=list)
    
    # Timing control
    start_time: Optional[datetime] = None
    max_total_time: float = 180.0  # seconds
    
    def add_trace(self, step: str, output: Any, metadata: Dict = None):
        """Add a trace entry for debugging and analysis"""
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
        """Record an error that occurred during processing"""
        self.errors.append({
            "step": step,
            "error": str(error)[:200],  # Truncate long error messages
            "fallback_used": fallback_used,
            "timestamp": datetime.now().isoformat()
        })
    
    def _get_elapsed(self) -> float:
        """Get time elapsed since start in seconds"""
        if not self.start_time:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()
    
    def time_remaining(self) -> float:
        """Get remaining time before timeout"""
        return max(0.0, self.max_total_time - self._get_elapsed())
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for serialization"""
        return {
            "query": self.query[:100],  # Truncate long queries
            "session_id": self.session_id,
            "domain": self.domain,
            "intent": self.intent,
            "confidence": self.confidence,
            "final_response": self.final_response[:500],  # Truncate long responses
            "trace": self.trace,
            "module_usage": self.module_usage,
            "step_times": self.step_times,
            "errors": self.errors,
            "total_time": self._get_elapsed()
        }
