# core/rlhf/feedback_logger.py - IMPROVED
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger("biomed.feedback")

LOG_FILE = "logs/rlhf_feedback.jsonl"

def log_feedback_with_context(
    session_id: str,
    preference: str,  # "good", "bad", or numeric score 1-5
    response_text: str = "",
    query_hash: str = "unknown",
    query_text: str = "",
    alternatives: list = None,  # Other response options shown
    reason: str = ""  # Optional reason for preference
) -> bool:
    """Enhanced feedback logging with context"""
    
    try:
        # Convert preference to numeric
        if preference == "good":
            pref_score = 1
        elif preference == "bad":
            pref_score = 0
        elif isinstance(preference, (int, float)) and 0 <= preference <= 1:
            pref_score = preference
        else:
            logger.warning(f"Invalid preference: {preference}")
            return False
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "preference": pref_score,
            "response_text": response_text[:2000],
            "query_hash": query_hash,
            "query_text": query_text[:500],  # Store query for context
            "alternatives": alternatives or [],
            "reason": reason[:200],
            "metadata": {
                "response_length": len(response_text),
                "has_alternatives": bool(alternatives),
                "has_reason": bool(reason)
            }
        }
        
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"📝 Feedback logged: {pref_score} for query {query_hash[:8]}")
        
        # Trigger training check if we have enough data
        _check_and_train()
        
        return True
        
    except Exception as e:
        logger.error(f"Feedback logging failed: {e}")
        return False

def _check_and_train():
    """Check if we should trigger training"""
    try:
        if not os.path.exists(LOG_FILE):
            return
        
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Train every 50 new feedbacks
        if len(lines) % 50 == 0:
            logger.info(f"📊 RLHF: {len(lines)} feedbacks collected, scheduling training")
            # Could schedule training here
            # asyncio.create_task(train_if_enough_feedback())
            
    except Exception as e:
        logger.warning(f"Training check failed: {e}")