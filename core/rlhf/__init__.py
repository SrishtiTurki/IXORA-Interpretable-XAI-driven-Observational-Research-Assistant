# core/rlhf/__init__.py

"""
RLHF (Reinforcement Learning from Human Feedback) Module
Initializes and manages the reward model training system.
"""

import os
import logging
from .reward_model import get_reward_model, save_reward_model, initialize_reward_model
from .feedback_logger import log_feedback_with_context as log_feedback
from .trainer import train_reward_model
import torch

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__ = [
    "get_reward_model",
    "save_reward_model",
    "log_feedback",
    "train_reward_model",
    "initialize_rlhf_system",
    "check_rlhf_status",
    "get_feedback_count"
]

# Global RLHF system state
_rlhf_initialized = False
_feedback_count = 0

def initialize_rlhf_system(warmup: bool = True):
    """
    Initialize the complete RLHF system.
    
    Args:
        warmup: If True, pre-initialize the reward model and create directories
    """
    global _rlhf_initialized
    
    try:
        logger.info("🚀 Initializing RLHF system...")
        
        # 1. Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # 2. Initialize reward model
        if warmup:
            logger.info("🔄 Warming up reward model...")
            
            # Initialize and save fresh model if needed
            if initialize_reward_model():
                logger.info("✅ Created new reward model")
            else:
                logger.info("✅ Loaded existing reward model")
            
            # Get model to ensure it's loaded
            model = get_reward_model()
            
            # Test model with a simple query
            if warmup:
                logger.info("🧪 Testing reward model...")
                try:
                    with torch.no_grad():
                        test_text = "This is a test response for RLHF initialization."
                        score = model([test_text])
                        logger.info(f"✅ Reward model test passed: {score.item():.4f}")
                except Exception as e:
                    logger.warning(f"⚠️ Reward model test failed: {e}")
        
        # 3. Check for existing feedback
        feedback_file = "logs/rlhf_feedback.jsonl"
        if os.path.exists(feedback_file):
            global _feedback_count
            with open(feedback_file, "r", encoding="utf-8") as f:
                _feedback_count = sum(1 for line in f if line.strip())
            logger.info(f"📊 Found {_feedback_count} existing feedback entries")
        
        # 4. Create initial feedback if none exists
        if _feedback_count == 0 and warmup:
            logger.info("📝 Creating initial feedback examples...")
            _create_initial_feedback()
        
        _rlhf_initialized = True
        logger.info("🎉 RLHF system initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RLHF system: {e}")
        return False

def _create_initial_feedback():
    """Create initial feedback examples to bootstrap the system"""
    import json
    from datetime import datetime
    import hashlib
    
    feedback_file = "logs/rlhf_feedback.jsonl"
    
    # Example feedback pairs (good vs bad responses)
    examples = [
        {
            "query": "What's the optimal pH for yeast growth?",
            "good_response": "The optimal pH for Saccharomyces cerevisiae (baker's yeast) is typically between 4.0 and 6.0, with the ideal around pH 5.0-5.5. This pH range supports enzymatic activity and membrane function. Outside this range, growth rates decline significantly due to proton gradient disruption and enzyme denaturation.",
            "bad_response": "Yeast likes pH. Maybe around neutral.",
            "domain": "biomed"
        },
        {
            "query": "How to optimize PCR conditions?",
            "good_response": "PCR optimization involves several key parameters: 1) Annealing temperature (typically 50-65°C, optimize with gradient PCR), 2) Magnesium concentration (1.5-2.5 mM MgCl2), 3) Primer concentration (0.1-1.0 μM each), 4) Template amount (1-100 ng genomic DNA), 5) Cycle number (25-35 cycles). Use positive and negative controls.",
            "bad_response": "Just follow the kit instructions.",
            "domain": "biomed"
        },
        {
            "query": "What's the time complexity of binary search?",
            "good_response": "Binary search has O(log n) time complexity in the average and worst cases, where n is the number of elements. It requires the array to be sorted. The algorithm works by repeatedly dividing the search interval in half, comparing the target value to the middle element. Space complexity is O(1) for iterative implementation.",
            "bad_response": "It's fast. Faster than linear search.",
            "domain": "cs"
        }
    ]
    
    created_count = 0
    for example in examples:
        query_hash = hashlib.sha256(example["query"].encode()).hexdigest()[:16]
        
        # Log good feedback
        entry_good = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": "system_init",
            "preference": 1.0,
            "response_text": example["good_response"],
            "query_hash": query_hash,
            "query_text": example["query"],
            "alternatives": [example["bad_response"]],
            "reason": "Detailed, accurate, scientifically sound",
            "domain": example.get("domain", "general"),
            "metadata": {
                "response_length": len(example["good_response"]),
                "has_alternatives": True,
                "has_reason": True,
                "source": "initial_training"
            }
        }
        
        # Log bad feedback
        entry_bad = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": "system_init",
            "preference": 0.0,
            "response_text": example["bad_response"],
            "query_hash": query_hash,
            "query_text": example["query"],
            "alternatives": [example["good_response"]],
            "reason": "Vague, lacks scientific detail",
            "domain": example.get("domain", "general"),
            "metadata": {
                "response_length": len(example["bad_response"]),
                "has_alternatives": True,
                "has_reason": True,
                "source": "initial_training"
            }
        }
        
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_good, ensure_ascii=False) + "\n")
            f.write(json.dumps(entry_bad, ensure_ascii=False) + "\n")
            created_count += 2
    
    global _feedback_count
    _feedback_count = created_count
    logger.info(f"Created {created_count} initial feedback entries")

def check_rlhf_status():
    """Check the status of the RLHF system"""
    global _rlhf_initialized, _feedback_count
    
    try:
        model = get_reward_model()
        model_loaded = model._model_loaded if hasattr(model, '_model_loaded') else False
        model_trained = model.is_trained() if hasattr(model, 'is_trained') else False
        
        # Count feedbacks
        feedback_file = "logs/rlhf_feedback.jsonl"
        if os.path.exists(feedback_file):
            with open(feedback_file, "r", encoding="utf-8") as f:
                _feedback_count = sum(1 for line in f if line.strip())
        
        return {
            "initialized": _rlhf_initialized,
            "model_loaded": model_loaded,
            "model_trained": model_trained,
            "feedback_count": _feedback_count,
            "model_path_exists": os.path.exists("models/reward_model.pth"),
            "feedback_file_exists": os.path.exists(feedback_file),
            "ready_for_training": _feedback_count >= 3
        }
        
    except Exception as e:
        logger.error(f"Failed to check RLHF status: {e}")
        return {
            "initialized": False,
            "error": str(e)
        }

def get_feedback_count():
    """Get the current number of feedback entries"""
    global _feedback_count
    return _feedback_count

def train_on_existing_feedback():
    """Train the reward model on all existing feedback"""
    try:
        logger.info("🔄 Training reward model on existing feedback...")
        status = check_rlhf_status()
        
        if status["feedback_count"] < 3:
            logger.warning(f"Not enough feedback to train (need 5, have {status['feedback_count']})")
            return False
        
        # Import and run training
        from .trainer import train_reward_model
        import asyncio
        
        success = asyncio.run(train_reward_model())
        
        if success:
            logger.info("✅ Successfully trained reward model on existing feedback")
            return True
        else:
            logger.warning("⚠️ Training completed with issues")
            return False
            
    except Exception as e:
        logger.error(f"Failed to train on existing feedback: {e}")
        return False

# Optional: Auto-initialize on import (disabled by default)
_AUTO_INIT = os.getenv("RLHF_AUTO_INIT", "false").lower() == "true"
if _AUTO_INIT:
    import threading
    threading.Thread(target=initialize_rlhf_system, kwargs={"warmup": True}, daemon=True).start()