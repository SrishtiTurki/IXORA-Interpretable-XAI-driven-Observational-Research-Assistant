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


# core/parameter_extractor.py - Updated _safe_json_parse function
def _safe_json_parse(text: str) -> Dict[str, Any]:
    """Ultra-robust JSON parser for LLM parameter extraction — survives anything Mistral throws"""
    if not text or not isinstance(text, str):
        logger.warning("Invalid input to _safe_json_parse")
        return {"parameters": {}}

    original_text = text
    text = text.strip()
    logger.debug(f"Parsing parameter JSON (length: {len(text)})")

    # Helper to clean common JSON issues
    def clean_json_string(s: str) -> str:
        s = s.strip()
        # Remove code block markers if present
        s = re.sub(r'^```json\s*|\s*```$', '', s, flags=re.IGNORECASE)
        # Fix newlines and tabs
        s = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Fix unquoted keys
        s = re.sub(r'(\w+)\s*:', r'"\1":', s)
        # Fix trailing commas
        s = re.sub(r',\s*}', '}', s)
        s = re.sub(r',\s*]', ']', s)
        # Fix single quotes
        s = s.replace("'", '"')
        # Remove any text before/after JSON
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

    # Step 2: Find and extract any JSON block
    json_blocks = re.findall(r'\{.*?\}', text, re.DOTALL)
    for block in json_blocks:
        try:
            cleaned_block = clean_json_string(block)
            data = json.loads(cleaned_block)
            if isinstance(data, dict):
                if "parameters" in data:
                    logger.info("Parsed JSON from extracted block")
                    return data
                elif any(isinstance(v, dict) and "value" in v for v in data.values()):
                    logger.info("Found parameter-like dict — wrapping")
                    return {"parameters": data}
                else:
                    return {"parameters": data}
        except:
            continue

    # Step 3: Aggressive key-value fallback (for when JSON is totally broken)
    logger.info("Falling back to aggressive key-value extraction")
    params = {}

    # Pattern 1: "key": { ... }
    complex_matches = re.findall(r'"([^"]+)"\s*:\s*\{([^}]+)\}', text, re.DOTALL)
    for key, content in complex_matches:
        param = {"confidence": 0.8}
        # Extract value
        val_match = re.search(r'"?value"?\s*:\s*([^,}\]]+)', content)
        if val_match:
            val_str = val_match.group(1).strip().strip('"')
            try:
                param["value"] = json.loads(val_str)
            except:
                param["value"] = val_str
        # Extract unit
        unit_match = re.search(r'"?unit"?\s*:\s*([^,}\]]+)', content)
        if unit_match:
            unit = unit_match.group(1).strip().strip('"')
            if unit.lower() != "null":
                param["unit"] = unit
        # Extract raw_text
        raw_match = re.search(r'"?raw_text"?\s*:\s*([^,}\]]+)', content)
        if raw_match:
            param["raw_text"] = raw_match.group(1).strip().strip('"')
        else:
            param["raw_text"] = param.get("value", "")
        if param.get("value") is not None:
            params[key.strip()] = param

    # Pattern 2: Simple key = value or key: value
    if not params:
        simple_matches = re.findall(r'(\w[\w\s]*?)\s*[:=]\s*([^\n:,}]+)', text)
        for key_raw, value_raw in simple_matches:
            key = key_raw.strip().lower().replace(' ', '_').replace('-', '_')
            value = value_raw.strip().strip('"\',')

            # Extract number + unit
            num_match = re.search(r'(-?\d+\.?\d*)', value)
            unit_match = re.search(r'[a-zA-Z°%µ]+', value)

            param = {
                "raw_text": value,
                "confidence": 0.7
            }

            if num_match:
                param["value"] = float(num_match.group(1))
                param["unit"] = unit_match.group(0) if unit_match else ""
            else:
                param["value"] = value
                param["unit"] = ""

            params[key] = param

    # Pattern 3: Last resort — look for known parameter names in text
    if not params:
        known_keys = ["ph", "temperature", "rpm", "time", "concentration", "volume", "glucose", "inoculum"]
        for key in known_keys:
            pattern = rf"{key}\s*[:=]?\s*([^\s,;]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1)
                params[key] = {
                    "value": val,
                    "unit": "",
                    "raw_text": val,
                    "confidence": 0.6
                }

    if params:
        logger.info(f"Recovered {len(params)} parameters via aggressive fallback")
        return {"parameters": params}

    logger.warning(f"Failed to extract any parameters from: {original_text[:300]}...")
    return {"parameters": {}}

async def extract_parameters(query: str, domain: str = "biomed") -> Dict[str, Any]:
    """
    Unified parameter extraction — PRIMARY: Mistral-Large API
    Skips local BioMistral/CodeLlama for speed and reliability.
    Uses aggressive, domain-aware prompt for rich parameter extraction.
    """
    logger.debug(f"Starting parameter extraction | domain={domain} | query='{query[:100]}...'")
    
    query = query.strip()
    if not query:
        return {"parameters": {}, "_metadata": {"note": "empty query"}}

    # === DOMAIN-SPECIFIC PROMPTS ===
    if domain == "biomed":
        system_intro = "You are an expert biomedical research assistant specialized in experimental design and parameter extraction."
        examples = """
        - "around 37 degrees" → temperature: 37.0, unit: "°C"
        - "physiological pH", "pH about 7.4", "neutral pH" → ph: 7.4
        - "incubate overnight" → incubation_time: [16, 18], unit: "hours"
        - "shake at 150-200 rpm" → agitation_speed: [150, 200], unit: "rpm"
        - "50 µL" → volume: 50, unit: "µL"
        - "LB media with ampicillin" → media: "LB", antibiotic: "ampicillin"
        - "E. coli DH5α" → strain: "DH5α"
        """
        param_hint = "Common parameters: pH, temperature, concentration, volume, time, rpm, agitation, media, strain, substrate, drug, dose, replicates, wavelength, OD600, glucose, inoculum, induction, IPTG, etc."

    elif domain == "cs":
        system_intro = "You are an expert in machine learning, algorithms, and computational experiments."
        examples = """
        - "lr 0.001 or 1e-3" → learning_rate: 0.001
        - "batch 64 or 128" → batch_size: [64, 128]
        - "train for 50-100 epochs" → epochs: [50, 100]
        - "Adam optimizer" → optimizer: "Adam"
        - "dropout 0.3 to 0.5" → dropout: [0.3, 0.5]
        - "ResNet-50 on ImageNet" → model_type: "ResNet-50", dataset: "ImageNet"
        """
        param_hint = "Common parameters: learning_rate, batch_size, epochs, optimizer, dropout, hidden_units, layers, dataset, model_type, hardware, seed, precision, throughput, latency, etc."

    else:
        return {"parameters": {}, "_metadata": {"error": "unsupported domain"}}

    # === AGGRESSIVE, STRICT PROMPT ===
    prompt = f"""{system_intro}

Your ONLY task is to extract ALL possible experimental or computational parameters from the query below — be EXTREMELY thorough and aggressive.

RULES:
- Extract EVERY parameter, even if implied, casual, or uncertain
- Include standard/implied values (e.g., "cell culture" → temperature ~37°C, pH ~7.4)
- Handle informal phrasing: "around", "about", "roughly", "overnight", "room temp", "shake well"
- Convert to standard values: "room temp" → 25°C, "overnight" → [16, 18] hours
- Use ranges [min, max] when given or implied
- Always include units when possible
- Assign confidence: 0.9+ for explicit, 0.6–0.8 for strong implication, 0.4–0.6 for weak/uncertain

{param_hint}

QUERY: "{query}"

EXAMPLES:
{examples}

CRITICAL:
- Return ONLY valid JSON — no explanations, no markdown, no extra text
- If no parameters found, return: {{"parameters": {{}}}}
- Never refuse or say "no parameters mentioned"

OUTPUT FORMAT:
{{
  "parameters": {{
    "parameter_name_in_snake_case": {{
      "value": number or [min, max] or "string_value",
      "unit": "unit_string" or null,
      "raw_text": "original phrase from query",
      "confidence": 0.0 to 1.0
    }}
  }}
}}
"""

    # === PRIMARY: Use Mistral-Large API directly ===
    if MISTRAL_USE_API and MISTRAL_API_KEY:
        try:
            logger.info("Using Mistral-Large API as PRIMARY for parameter extraction")
            api_response = await call_mistral_api(
                prompt=prompt,
                max_tokens=500,        # Allow rich output
                temperature=0.1        # Very low → consistent, structured JSON
            )
            
            params = _safe_json_parse(api_response)
            if params and isinstance(params.get("parameters"), dict):
                params["_metadata"] = {
                    "method": "mistral_api_primary",
                    "model": "mistral-large-latest",
                    "success": True,
                    "param_count": len(params["parameters"])
                }
                logger.info(f"Mistral API successfully extracted {len(params['parameters'])} parameters")
                return params
                
        except Exception as e:
            logger.error(f"Mistral API failed during parameter extraction: {e}")

    # === NO LOCAL LLM FALLBACK FOR EXTRACTION ===
    # We intentionally skip BioMistral/CodeLlama here — they're too slow/unreliable for structured JSON

    # === FINAL: Clean empty fallback (no regex!) ===
    logger.warning("Parameter extraction failed — returning empty result")
    return {
        "parameters": {},
        "_metadata": {
            "method": "failed",
            "reason": "mistral_api_unavailable_or_error",
            "success": False
        }
    }