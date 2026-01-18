# core/rlhf/feedback_logger.py - DOMAIN-AWARE FEEDBACK LOGGER

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

logger = logging.getLogger("core.rlhf.feedback")

# Configuration
LOG_DIR = Path("logs/rlhf")
LOG_FILE = LOG_DIR / "feedback.jsonl"
MIN_SAMPLES_PER_DOMAIN = 50  # Minimum samples before considering training for a domain

@dataclass
class FeedbackEntry:
    """Structured feedback entry with domain support"""
    timestamp: str
    session_id: str
    preference: float  # 0-1 scale where 1 is best
    response_text: str
    query_hash: str
    query_text: str
    domain: str
    model_version: str = "unknown"
    alternatives: List[dict] = None
    reason: str = ""
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["metadata"] = data.get("metadata") or {}
        data["metadata"].update({
            "response_length": len(self.response_text),
            "has_alternatives": bool(self.alternatives),
            "has_reason": bool(self.reason),
            "domain": self.domain,
            "model_version": self.model_version,
            "timestamp": self.timestamp
        })
        return data

class FeedbackLogger:
    """Domain-aware feedback logger with training triggers"""
    
    def __init__(self, log_file: Optional[os.PathLike] = None):
        self.log_file = Path(log_file) if log_file else LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._domain_stats: Dict[str, int] = {}
        self._last_training_check: Dict[str, int] = {}
        
        # Initialize stats
        self._load_stats()
    
    def _load_stats(self):
        """Load domain statistics from existing log file"""
        if not self.log_file.exists():
            return
            
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        domain = entry.get("domain", "unknown").lower()
                        self._domain_stats[domain] = self._domain_stats.get(domain, 0) + 1
                    except (json.JSONDecodeError, AttributeError):
                        continue
        except Exception as e:
            logger.warning(f"Failed to load feedback stats: {e}")
    
    @staticmethod
    def _hash_query(query: str) -> str:
        """Generate a consistent hash for a query"""
        return hashlib.md5(query.encode("utf-8")).hexdigest()
    
    def log_feedback(
        self,
        query: str,
        response: str,
        preference: Union[str, float, int],
        domain: str,
        session_id: str = "unknown",
        alternatives: Optional[List[dict]] = None,
        reason: str = "",
        model_version: str = "unknown",
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Log feedback with domain awareness
        
        Args:
            query: The original user query
            response: The model's response
            preference: Feedback score ("good"/"bad" or 0-1)
            domain: Domain of the query (e.g., "biomed", "computerscience")
            session_id: User session ID
            alternatives: Other response options shown
            reason: Optional reason for the preference
            model_version: Version of the model that generated the response
            metadata: Additional metadata to store
            
        Returns:
            bool: True if logging was successful
        """
        try:
            # Normalize domain
            domain = domain.lower() if domain else "unknown"
            
            # Convert preference to numeric
            if isinstance(preference, str):
                pref_score = 1.0 if preference.lower() == "good" else 0.0
            elif isinstance(preference, (int, float)):
                pref_score = float(max(0.0, min(1.0, preference)))
            else:
                logger.warning(f"Invalid preference: {preference}")
                return False
            
            # Create and save entry
            entry = FeedbackEntry(
                timestamp=datetime.utcnow().isoformat(),
                session_id=session_id,
                preference=pref_score,
                response_text=response[:4000],  # Limit response length
                query_hash=self._hash_query(query),
                query_text=query[:1000],  # Limit query length
                domain=domain,
                model_version=model_version,
                alternatives=alternatives[:5] if alternatives else None,  # Limit alternatives
                reason=reason[:500],  # Limit reason length
                metadata=metadata or {}
            )
            
            # Append to log file
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            
            # Update stats
            self._domain_stats[domain] = self._domain_stats.get(domain, 0) + 1
            
            # Log the feedback
            logger.info(
                f"📝 Feedback logged | Domain: {domain} | "
                f"Preference: {pref_score:.2f} | "
                f"Query: {query[:30]}..."
            )
            
            # Check if we should trigger training for this domain
            self._check_and_train(domain)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}", exc_info=True)
            return False
    
    def _check_and_train(self, domain: str):
        """Check if we should trigger training for a domain"""
        try:
            domain_count = self._domain_stats.get(domain, 0)
            last_check = self._last_training_check.get(domain, 0)
            
            # Only check if we have enough new samples
            if domain_count - last_check >= MIN_SAMPLES_PER_DOMAIN:
                self._last_training_check[domain] = domain_count
                logger.info(
                    f"📊 RLHF: {domain_count} feedbacks for {domain}, "
                    f"considering training..."
                )
                # Schedule training in the background
                asyncio.create_task(self._schedule_training(domain))
                
        except Exception as e:
            logger.error(f"Training check failed for {domain}: {e}")
    
    async def _schedule_training(self, domain: str):
        """Schedule training for a domain"""
        try:
            from .trainer import train_reward_model
            logger.info(f"🚀 Scheduling training for domain: {domain}")
            await train_reward_model(domain=domain)
        except Exception as e:
            logger.error(f"Failed to schedule training for {domain}: {e}")
    
    def get_domain_stats(self) -> Dict[str, int]:
        """Get statistics about feedback per domain"""
        return dict(sorted(self._domain_stats.items()))

# Global instance
feedback_logger = FeedbackLogger()

def log_feedback_with_context(
    query: str,
    response: str,
    preference: Union[str, float, int],
    domain: str,
    **kwargs
) -> bool:
    """
    Log feedback with context (compatibility wrapper)
    
    Args:
        query: The original user query
        response: The model's response
        preference: Feedback score ("good"/"bad" or 0-1)
        domain: Domain of the query
        **kwargs: Additional arguments for FeedbackLogger.log_feedback
        
    Returns:
        bool: True if logging was successful
    """
    return feedback_logger.log_feedback(
        query=query,
        response=response,
        preference=preference,
        domain=domain,
        **kwargs
    )