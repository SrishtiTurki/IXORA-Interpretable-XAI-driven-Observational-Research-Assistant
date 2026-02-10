"""
core/utils.py - Minimal shared utilities (cleaned Feb 2026)
Only keeps actively used helpers: caching, session state, text formatting
"""

import json
import hashlib
import time
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("core.utils")


# ========== SIMPLE FILE-BASED CACHE ==========

def cache_set(key: str, value: Any, ttl: int = 3600) -> bool:
    """Save value to file cache with TTL (handles numpy/decimal types)"""
    try:
        os.makedirs("cache", exist_ok=True)
        cache_file = f"cache/{hashlib.md5(key.encode()).hexdigest()}.json"

        def convert_for_cache(obj):
            if isinstance(obj, dict):
                return {k: convert_for_cache(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_for_cache(item) for item in obj]
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            return str(obj)

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                "value": convert_for_cache(value),
                "expires": time.time() + ttl,
                "created": time.time()
            }, f, ensure_ascii=False)
        return True
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")
        return False


def cache_get(key: str) -> Optional[Any]:
    """Load from file cache if not expired"""
    try:
        cache_file = f"cache/{hashlib.md5(key.encode()).hexdigest()}.json"
        if not os.path.exists(cache_file):
            return None

        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if time.time() < data.get("expires", 0):
                return data.get("value")
            os.remove(cache_file)
            return None
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
        return None


# ========== SESSION STATE ==========

def load_session_state(session_id: str) -> Dict[str, Any]:
    """Load session from cache or return empty dict"""
    state = cache_get(f"session:{session_id}")
    if state:
        logger.debug(f"Loaded session {session_id}")
        return state
    logger.debug(f"No session found for {session_id}")
    return {}


def save_session_state(session_id: str, state: Dict[str, Any]):
    """Save session state (last 20 messages kept elsewhere)"""
    cache_set(f"session:{session_id}", state, ttl=86400)  # 24h default
    logger.debug(f"Saved session {session_id}")


# ========== TEXT HELPERS ==========

def clean_text(text: str) -> str:
    """Normalize whitespace"""
    if not text:
        return ""
    return ' '.join(text.split()).strip()


def format_parameters_for_display(parameters: Dict[str, Any]) -> str:
    """Pretty-print parameters for logs / UI"""
    if not parameters:
        return "No parameters extracted"

    lines = []
    for key, param in sorted(parameters.items()):
        value = param.get("value", "")
        unit = param.get("unit", "")
        conf = param.get("confidence", 0.0)
        method = param.get("method", "unknown")

        if isinstance(value, list) and len(value) == 2:
            val_str = f"{value[0]} – {value[1]}"
        else:
            val_str = str(value)

        lines.append(
            f"- **{key}**: {val_str} {unit} "
            f"(conf: {conf:.0%}, method: {method})"
        )
    return "\n".join(lines) if lines else "No valid parameters"

# core/utils.py (add this function near the bottom)

def select_explainability_method(
    user_input: str,
    parameters: Dict[str, Any]
) -> str:
    """
    Choose the best explainability method (SHAP, LIME, both, or none)
    based on query intent and number of parameters.

    Returns: "shap", "lime", "both", or "none"
    """
    if not parameters:
        return "none"

    input_lower = user_input.lower()

    # Explicit user preference
    if any(w in input_lower for w in ["both", "all methods", "shap and lime", "lime and shap"]):
        return "both"

    wants_shap = any(w in input_lower for w in ["global", "overall", "feature importance", "all parameters", "shap"])
    wants_lime = any(w in input_lower for w in ["local", "specific instance", "this case", "why this", "lime"])

    # Rule-based heuristics
    n_params = len(parameters)

    if n_params <= 3:
        # Few features → LIME is faster & more interpretable for local
        preferred = "lime"
    elif n_params >= 8:
        # Many features → SHAP for global importance & efficiency
        preferred = "shap"
    else:
        # Middle ground — default to SHAP (better for biomedical/global patterns)
        preferred = "shap"

    # Override based on explicit request
    if wants_shap and wants_lime:
        return "both"
    if wants_shap:
        return "shap"
    if wants_lime:
        return "lime"

    # Final fallback
    return preferred
# ========== DEPRECATED / REMOVED ==========
# The following were removed:
# - NLTK setup & parameter extraction (moved to domain-specific extractors)
# - detect_intent (replaced by Qwen LLM router)
# - select_explainability_method (can move to analytics.py if needed)
# - All regex/NLTK heavy lifting (redundant with LLM extraction)