# core/rlhf/feedback_logger.py - UPDATED

import os
import json
from datetime import datetime
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger("biomed.feedback")

LOG_FILE = Path("logs/rlhf_feedback.jsonl")

def log_feedback_with_context(
    session_id: str,
    preference: str,  # "good", "bad", or numeric score 1-5
    response_text: str = "",
    query_hash: str = "unknown",
    query_text: str = "",
    alternatives: list = None,
    reason: str = ""
) -> bool:
    """Enhanced feedback logging with context"""
    
    try:
        # Convert preference to numeric
        if preference == "good":
            pref_score = 1.0
        elif preference == "bad":
            pref_score = 0.0
        elif isinstance(preference, (int, float)) and 0 <= preference <= 1:
            pref_score = float(preference)
        else:
            logger.warning(f"Invalid preference: {preference}")
            return False
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "preference": pref_score,
            "response_text": response_text[:2000],
            "query_hash": query_hash,
            "query_text": query_text[:500],
            "alternatives": alternatives or [],
            "reason": reason[:200],
            "metadata": {
                "response_length": len(response_text),
                "has_alternatives": bool(alternatives),
                "has_reason": bool(reason)
            }
        }
        
        # Ensure log directory exists
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to log file
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"📝 Feedback logged: {pref_score} for query {query_hash[:8]}")
        
        # Trigger training check (async, non-blocking)
        asyncio.create_task(_check_and_train_async())
        
        return True
        
    except Exception as e:
        logger.error(f"Feedback logging failed: {e}")
        return False

async def _check_and_train_async():
    """Async version of training check"""
    try:
        if not LOG_FILE.exists():
            return
        
        # Count feedback entries
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        
        count = len(lines)
        
        # Train every 10 feedbacks
        if count >= 3 and count % 3 == 0:
            logger.info(f"📈 RLHF: {count} feedbacks → starting training...")
            
            try:
                # Import trainer
                from core.rlhf.trainer import train_reward_model
                
                # Run training
                success = await train_reward_model()
                if success:
                    from core.rlhf.reward_model import save_reward_model
                    save_reward_model()
                    logger.info("🎉 Reward model updated from user feedback!")
                else:
                    logger.warning("Training did not complete successfully")
                    
            except Exception as e:
                logger.error(f"Training failed: {e}")
                
    except Exception as e:
        logger.warning(f"Training check failed: {e}")

# Alias for compatibility
log_feedback = log_feedback_with_context