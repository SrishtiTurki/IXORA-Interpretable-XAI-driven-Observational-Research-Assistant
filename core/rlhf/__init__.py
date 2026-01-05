# core/rlhf/__init__.py - COMPLETELY FIXED VERSION

import logging

logger = logging.getLogger("rlhf")

# DO NOT IMPORT SciSpaCy HERE - it's causing startup failures
# We'll handle NLP imports lazily in functions that need them

# Export the main functions
from .feedback_logger import log_feedback, log_feedback_with_context
from .trainer import train_reward_model
from .reward_model import get_reward_model

logger.info("RLHF module loaded successfully (SciSpaCy disabled)")