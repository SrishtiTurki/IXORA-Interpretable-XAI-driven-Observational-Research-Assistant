# core/parameter_extractor.py - OPTIMIZED VERSION
# Improvements:
# 1. Tighter timeout (30s → 15s for parameter extraction)
# 2. Faster JSON parsing with better fallbacks
# 3. Streamlined prompts for faster inference

import json
import logging
import re
from typing import Dict, Any
import asyncio
from core.config import MISTRAL_USE_API, MISTRAL_API_KEY

logger = logging.getLogger("core.parameter_extractor")

# Import Mistral API caller
from core.mistral import call_mistral_api


def _safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Ultra-robust JSON parser for LLM parameter extraction
    OPTIMIZED: Faster parsing with early returns
    """
    if not text or not isinstance(text, str):
        return {"parameters": {}}

    text = text.strip()
    
    # Quick check for empty/invalid response
    if len(text) < 10 or text.lower() in ["none", "n/a", "null", "{}"]:
        return {"parameters": {}}
    
    # Helper to clean JSON
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

    # Step 1: Try direct parse (fastest path)
    try:
        cleaned = clean_json_string(text)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            if "parameters" in data:
                return data
            else:
                return {"parameters": data}
    except json.JSONDecodeError:
        pass

    # Step 2: Extract JSON blocks
    json_blocks = re.findall(r'\{.*?\}', text, re.DOTALL)
    for block in json_blocks[:3]:  # Only try first 3 blocks for speed
        try:
            cleaned_block = clean_json_string(block)
            data = json.loads(cleaned_block)
            if isinstance(data, dict) and len(data) > 0:
                if "parameters" in data:
                    return data
                elif any(isinstance(v, dict) for v in data.values()):
                    return {"parameters": data}
        except:
            continue

    # Step 3: Aggressive key-value fallback
    params = {}
    
    # Pattern: "key": {...}
    complex_matches = re.findall(r'"([^"]+)"\s*:\s*\{([^}]+)\}', text)
    for key, content in complex_matches[:10]:  # Limit to 10 for speed
        param = {"confidence": 0.8}
        
        val_match = re.search(r'"?value"?\s*:\s*([^,}\]]+)', content)
        if val_match:
            val_str = val_match.group(1).strip().strip('"')
            try:
                param["value"] = json.loads(val_str)
            except:
                param["value"] = val_str
        
        unit_match = re.search(r'"?unit"?\s*:\s*([^,}\]]+)', content)
        if unit_match:
            unit = unit_match.group(1).strip().strip('"')
            if unit.lower() != "null":
                param["unit"] = unit
        
        raw_match = re.search(r'"?raw_text"?\s*:\s*([^,}\]]+)', content)
        if raw_match:
            param["raw_text"] = raw_match.group(1).strip().strip('"')
        else:
            param["raw_text"] = param.get("value", "")
        
        if param.get("value") is not None:
            params[key.strip()] = param

    if params:
        return {"parameters": params}

    # Final fallback: empty
    logger.warning(f"Failed to extract parameters from: {text[:100]}...")
    return {"parameters": {}}


# In parameter_extractor.py - Update to use chat format:

async def extract_parameters(query: str, domain: str = "biomed") -> Dict[str, Any]:
    """Extract parameters using Qwen Transformers"""
    
    # === PRIMARY: Use Qwen Transformers ===
    try:
        from core.model_loader import model_loader
        from core.config import QWEN_USE_TRANSFORMERS
        
        logger.info("Using Qwen for parameter extraction...")
        
        if QWEN_USE_TRANSFORMERS:
            # Chat format for Transformers
            messages = [
                {
                    "role": "system",
                    "content": """You are a parameter extraction assistant. 
Return ONLY valid JSON with extracted parameters."""
                },
                {
                    "role": "user", 
                    "content": f"""Extract ALL experimental parameters from this query:

Query: {query}

Return ONLY valid JSON with this structure:
{{
  "parameters": {{
    "parameter_name": {{
      "value": "value",
      "unit": "unit_if_any",
      "raw_text": "original_text",
      "confidence": 0.9
    }}
  }}
}}"""
                }
            ]
            
            result = await model_loader.generate_with_qwen_transformers(
                messages=messages,
                max_tokens=400
            )
        else:
            # Plain prompt for llama.cpp
            prompt = f"""Extract ALL experimental parameters from this query:

Query: {query}

Return ONLY valid JSON with this structure:
{{
  "parameters": {{
    "parameter_name": {{
      "value": "value",
      "unit": "unit_if_any",
      "raw_text": "original_text",
      "confidence": 0.9
    }}
  }}
}}"""
            
            result = await model_loader.generate_with_qwen(
                prompt=prompt,
                max_tokens=400,
                temperature=0.05
            )
        
        if result:
            params = _safe_json_parse(result)
            if params.get("parameters"):
                params["_metadata"] = {
                    "method": "qwen_transformers" if QWEN_USE_TRANSFORMERS else "qwen_llamacpp",
                    "model": "Qwen2.5-7B-Instruct",
                    "success": True,
                    "param_count": len(params["parameters"])
                }
                logger.info(f"Qwen extracted {len(params['parameters'])} parameters")
                return params
                
    except Exception as e:
        logger.warning(f"Qwen extraction failed: {e}")
    
    # Fallback to Mistral API...
    # === Mistral-Large API with TIGHT TIMEOUT ===
    if MISTRAL_USE_API and MISTRAL_API_KEY:
        try:
            logger.info("Using Mistral-Large API for parameter extraction (15s timeout)")
            
            # Call with timeout
            api_response = await asyncio.wait_for(
                call_mistral_api(
                    prompt=prompt,
                    max_tokens=400,  # Reduced from 500
                    temperature=0.05  # Very low for consistency
                ),
                timeout=15.0  # TIGHTENED from 30s to 15s
            )
            
            params = _safe_json_parse(api_response)
            if params and isinstance(params.get("parameters"), dict):
                params["_metadata"] = {
                    "method": "mistral_api_optimized",
                    "model": "mistral-large-latest",
                    "success": True,
                    "param_count": len(params["parameters"])
                }
                logger.info(f"Extracted {len(params['parameters'])} parameters in <15s")
                return params
                
        except asyncio.TimeoutError:
            logger.warning("Mistral API timed out after 15s")
        except Exception as e:
            logger.error(f"Mistral API failed: {e}")

    # === FINAL FALLBACK ===
    logger.warning("Parameter extraction failed — returning empty")
    return {
        "parameters": {},
        "_metadata": {
            "method": "failed",
            "reason": "mistral_api_unavailable_or_timeout",
            "success": False
        }
    }