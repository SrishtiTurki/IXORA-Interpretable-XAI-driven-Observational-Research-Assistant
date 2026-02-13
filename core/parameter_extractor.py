# core/parameter_extractor.py - UNIFIED DOMAIN-AWARE PARAMETER EXTRACTOR
# Primary: Domain-specific local LLM → Fallback: Mistral API

import json
import logging
import re
from typing import Dict, Any
from core.config import MISTRAL_USE_API, MISTRAL_API_KEY

logger = logging.getLogger("core.parameter_extractor")

# Import domain-specific draft generators (they reuse already-loaded models)
try:
    from core.computerscience.loaders import generate_cs_draft
    from core.medicalscience.loaders import generate_biomistral_draft  # Adjust if your biomed loader is elsewhere
except ImportError as e:
    logger.warning(f"Domain loaders import failed: {e}")
    generate_cs_draft = None
    generate_biomistral_draft = None

# Import Mistral API caller
from core.mistral import call_mistral_api
from core.model_loader import generate_with_qwen

# core/parameter_extractor.py - FIXED VERSION
import json
import logging
import re
from typing import Dict, Any
from core.config import MISTRAL_USE_API, MISTRAL_API_KEY

logger = logging.getLogger("core.parameter_extractor")

# Import Mistral API caller
from core.mistral import call_mistral_api

def _safe_json_parse(text: str) -> Dict[str, Any]:
    """Ultra-robust JSON parser for LLM parameter extraction"""
    if not text or not isinstance(text, str):
        logger.warning("Invalid input to _safe_json_parse")
        return {"parameters": {}}

    original_text = text
    text = text.strip()
    logger.debug(f"Parsing parameter JSON (length: {len(text)})")

    # Helper to clean common JSON issues
    def clean_json_string(s: str) -> str:
        s = s.strip()
        s = re.sub(r'^```json\s*|\s*```$', '', s, flags=re.IGNORECASE)
        s = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        s = re.sub(r'(\w+)\s*:', r'"\1":', s)
        s = re.sub(r',\s*}', '}', s)
        s = re.sub(r',\s*]', ']', s)
        s = s.replace("'", '"')
        json_match = re.search(r'\{.*\}', s, re.DOTALL)
        if json_match:
            s = json_match.group(0)
        return s

    # Step 1: Try direct parse
    try:
        cleaned = clean_json_string(text)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            if "parameters" in data:
                logger.info("Parsed JSON successfully (direct)")
                return data
            else:
                logger.info("Parsed JSON — wrapping in 'parameters'")
                return {"parameters": data}
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")

    # Step 2: Simple fallback - extract key-value pairs
    logger.info("Using fallback key-value extraction")
    params = {}
    
    # Look for patterns like "parameter: value" or "parameter = value"
    patterns = [
        r'(\w+)\s*[:=]\s*([^,\n]+)',
        r'"([^"]+)"\s*:\s*([^,\n}]+)',
        r'([A-Za-z_]+)\s+(?:of|at|is|was)\s+([0-9.]+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for key, value in matches:
            key = key.strip().lower().replace(' ', '_')
            value = value.strip().strip('"\',')
            
            # Try to parse numbers
            try:
                if '.' in value:
                    num_value = float(value)
                else:
                    num_value = int(value)
                params[key] = {
                    "value": num_value,
                    "unit": "",
                    "raw_text": value,
                    "confidence": 0.7
                }
            except ValueError:
                params[key] = {
                    "value": value,
                    "unit": "",
                    "raw_text": value,
                    "confidence": 0.6
                }
    
    if params:
        logger.info(f"Extracted {len(params)} parameters via fallback")
        return {"parameters": params}

    logger.warning(f"Failed to extract any parameters")
    return {"parameters": {}}

async def extract_parameters(query: str, domain: str = "biomed") -> Dict[str, Any]:
    """
    Improved parameter extraction:
    - Very strict anti-hallucination rules
    - No invented ranges unless explicitly stated
    - Post-processing validation against original query
    - Lower temperature + JSON mode preference
    """
    logger.debug(f"Starting parameter extraction | domain={domain} | query='{query[:120]}...'")

    query = query.strip()
    if not query:
        return {"parameters": {}, "_metadata": {"note": "empty query"}}

    # ── Very strict, anti-hallucination prompt ───────────────────────────────
    common_params = (
        "pH, temperature, concentration, time, rpm, agitation, volume, media, "
        "strain, substrate, drug dose, replicates, inoculum size"
        if domain == "biomed" else
        "learning_rate, batch_size, epochs, optimizer, dropout, momentum, "
        "weight_decay, dataset, model_type, hardware, seed"
    )

    prompt = f"""You are an extremely precise scientific parameter extractor.
Your ONLY job is to find EXACTLY the numbers, ranges, inequalities and units THAT LITERALLY APPEAR in the text.

STRICT RULES — YOU MUST FOLLOW ALL OF THEM:
1. ONLY extract values/ranges when they are EXPLICITLY WRITTEN in the query.
2. NEVER invent, guess, extrapolate, symmetrize, average or hallucinate any number or range.
3. If only ONE number is given (e.g. "37 °C", "learning rate 0.001") → do NOT turn it into a range.
4. ONLY create a range [low, high] when BOTH endpoints are explicitly stated (e.g. "from 20 to 40", "between 6.5 and 7.5", "64–256").
5. Approximate language ("around 37", "about 150 rpm", "~7.0", "approx 0.01") → extract the stated number, but set confidence 0.6–0.75 and include "approximate" in raw_text.
6. NEVER add parameters that do not appear in the query at all.
7. If nothing extractable → return EXACTLY {{"parameters": {{}}}} — no fake entries.
8. Return ONLY valid JSON — no explanation, no markdown, no comments, no extra text.

Common parameters in this domain: {common_params}

QUERY:
{query}

Return ONLY JSON in this exact structure:
{{
  "parameters": {{
    "parameter_name": {{
      "value": number | [min, max] (only if explicit range),
      "unit": "unit if present" | "",
      "raw_text": "the EXACT phrase from the query",
      "confidence": 0.0 to 1.0,
      "note": "optional — only if approximate / inequality / explicit range"
    }}
  }}
}}

If no parameters → {{"parameters": {{}}}}
"""

    try:
        logger.info("Calling Mistral API with strict parameter extraction prompt")

        # Prefer JSON mode when available (Mistral Large / recent models support it)
        result = await call_mistral_api(
            prompt=prompt,
            max_tokens=600,
            temperature=0.0,           # ← crucial: zero creativity
            # If your Mistral version supports it:
            # response_format={"type": "json_object"}
        )

        # Parse safely
        parsed = _safe_json_parse(result)

        # ── Post-processing: validate every extracted item against original query ──
        validated_params = {}
        query_lower = query.lower()

        for name, param in parsed.get("parameters", {}).items():
            raw = (param.get("raw_text") or "").lower().strip()
            value = param.get("value")
            is_range = isinstance(value, list) and len(value) == 2

            # 1. raw_text MUST appear in query (strongest filter)
            if not raw or raw not in query_lower:
                logger.debug(f"Discarded hallucinated param: {name} — raw_text '{raw}' not found")
                continue

            # 2. For ranges: both numbers should appear
            if is_range:
                v1, v2 = str(value[0]), str(value[1])
                if v1 not in query_lower or v2 not in query_lower:
                    logger.debug(f"Discarded fake range for {name}: {value}")
                    continue  # or downgrade to single value if desired

            # 3. Confidence adjustment for approximate language
            approx_markers = ["around", "about", "~", "approx", "roughly", "approximately"]
            if any(m in raw for m in approx_markers):
                param["confidence"] = min(param.get("confidence", 0.95), 0.75)
                if "note" not in param:
                    param["note"] = "approximate value"

            validated_params[name] = param

        final_result = {
            "parameters": validated_params,
            "_metadata": {
                "method": "mistral_api_strict",
                "success": bool(validated_params),
                "param_count": len(validated_params),
                "original_count": len(parsed.get("parameters", {})),
                "discarded": len(parsed.get("parameters", {})) - len(validated_params)
            }
        }

        if validated_params:
            logger.info(f"Extracted {len(validated_params)} validated parameters "
                       f"(discarded {final_result['_metadata']['discarded']})")
        else:
            logger.info("No valid parameters after strict validation")

        return final_result

    except Exception as e:
        logger.error(f"Parameter extraction failed: {e}", exc_info=True)
        return {
            "parameters": {},
            "_metadata": {
                "method": "failed",
                "success": False,
                "reason": str(e)[:120]
            }
        }