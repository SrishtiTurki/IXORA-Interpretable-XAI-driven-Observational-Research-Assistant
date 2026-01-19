# core/mistral.py - FIXED VERSION
import asyncio
import logging
import json
import re
import os
import torch
import aiohttp
from typing import Dict, Any, Tuple, List
import sys

logger = logging.getLogger("core.mistral")

# Globals
mistral_pipeline = None
mistral_tokenizer = None
mistral_model = None

async def load_mistral() -> Any:
    global mistral_pipeline, mistral_tokenizer, mistral_model
    if mistral_pipeline or (mistral_model and mistral_tokenizer):
        logger.info("Mistral already loaded (cached)")
        return mistral_pipeline or mistral_model
    
    from core.config import MISTRAL_MODEL_NAME, MISTRAL_USE_API, MISTRAL_API_KEY, MISTRAL_DEVICE
    
    try:
        if MISTRAL_USE_API:
            if not MISTRAL_API_KEY:
                raise ValueError("MISTRAL_API_KEY required for API mode")
            
            logger.info("SUCCESS: Mistral-Large API mode configured")
            return "api_mode"
            
        else:
            logger.info(f"Loading local Mistral model: {MISTRAL_MODEL_NAME}")
            # Local model loading code...
            return "local_mode"
            
    except Exception as e:
        logger.error(f"Mistral load failed: {e}", exc_info=True)
        return "fallback"

async def call_mistral_api(prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
    """Direct API call with error handling"""
    from core.config import MISTRAL_API_KEY
    
    if not MISTRAL_API_KEY:
        logger.error("No Mistral API key configured!")
        return ""
    
    url = "https://api.mistral.ai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Simplified prompt for debugging
    system_msg = "You are a helpful biomedical research assistant. Provide detailed, comprehensive responses."
    
    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        logger.info(f"Sending request to Mistral API for {len(prompt)} chars...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=120) as response:
                logger.info(f"Response status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text[:200]}")
                    return ""
                
                result = await response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    logger.info(f"API response received: {len(content)} chars")
                    return content
                else:
                    logger.error(f"No choices in response: {result}")
                    return ""
                    
    except asyncio.TimeoutError:
        logger.error("API request timeout (30s)")
        return ""
    except aiohttp.ClientError as e:
        logger.error(f"HTTP client error: {e}")
        return ""
    except Exception as e:
        logger.error(f"API call exception: {e}")
        return ""

async def generate_with_mistral(
    prompt: str, 
    max_tokens: int = 1200, 
    temperature: float = 0.7
) -> Tuple[str, List[str]]:
    """Generate with error handling and fallback"""
    
    logger.info(f"Generating with Mistral: {len(prompt)} chars")
    
    model = await load_mistral()
    if not model:
        logger.error("Mistral model not loaded!")
        return "Mistral service unavailable", []
    
    logger.info(f"Model type: {model}")
    
    content = ""
    cot_steps = []
    
    try:
        if model == "api_mode":
            logger.info("Using API mode...")
            content = await call_mistral_api(prompt, max_tokens, temperature)
            
            if not content:
                logger.warning("API returned empty content, using fallback...")
                # Create a simple fallback response
                content = f"I've analyzed your query. This appears to be about biomedical research involving {prompt[:100]}... For detailed analysis, please ensure your Mistral API key is properly configured."
            
        elif model == "fallback":
            logger.info("Using fallback mode...")
            content = f"Analysis for: {prompt[:200]}... [Fallback mode active - configure Mistral API for full analysis]"
            cot_steps = ["Using fallback generation"]
            
        else:
            logger.info("Using local model...")
            content = "Local model generation not implemented in this version"
        
        logger.info(f"Generated content length: {len(content)}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        content = f"Error during generation: {str(e)[:100]}"
        cot_steps = []
    
    return content, cot_steps

def enforce_xml_structure(content: str, user_query: str, domain: str = "biomed") -> str:
    """Add basic XML structure if missing (domain-aware for CS/Biomed)"""
    if not content:
        if domain == "cs":
            return f"""<enthusiasm>Oh, that's excellent! Great computer science research question!</enthusiasm>
<clarify>What specific algorithmic approach or constraints should I consider?</clarify>
<explanation>Your query about "{user_query[:100]}" seems computational. I couldn't generate a full analysis due to a technical issue, but typical CS investigations consider algorithmic complexity, data structures, baselines, and performance metrics.</explanation>
<hypothesis>Computational parameters will significantly influence algorithmic performance metrics.</hypothesis>
<followup>Could you rephrase your question or specify algorithm, dataset size, and constraints?</followup>"""
        else:
            return f"""<enthusiasm>Exciting research question!</enthusiasm>
<explanation>Your query about "{user_query[:100]}" is interesting for biomedical research. While I couldn't generate a full analysis due to a technical issue, I can tell you that yeast biomass studies typically investigate factors like pH, temperature, nutrients, and incubation time.</explanation>
<hypothesis>The optimal pH and temperature combination will maximize yeast biomass production while maintaining metabolic activity.</hypothesis>
<followup>What specific measurements are you planning to take? Do you have access to spectrophotometry for biomass quantification?</followup>"""
    
    # Only add tags if completely missing
    if '<enthusiasm>' not in content:
        enthusiasm_text = "Oh, that's excellent! Great CS research question!" if domain == "cs" else "Great research question!"
        content = f"<enthusiasm>{enthusiasm_text}</enthusiasm>\n\n{content}"
    
    if domain == "cs" and '<clarify>' not in content:
        if '</enthusiasm>' in content:
            parts = content.split('</enthusiasm>', 1)
            content = f"{parts[0]}</enthusiasm>\n\n<clarify>What algorithmic approach and constraints should I consider?</clarify>\n\n{parts[1]}"
        else:
            content = f"<clarify>What algorithmic approach and constraints should I consider?</clarify>\n\n{content}"

    if '<explanation>' not in content:
        lines = content.split('\n')
        explanation_start = 0
        for i, line in enumerate(lines):
            if '</clarify>' in line:
                explanation_start = i + 1
                break
            if '</enthusiasm>' in line:
                explanation_start = i + 1
                break
        
        if explanation_start < len(lines):
            lines[explanation_start] = f"<explanation>\n{lines[explanation_start]}"
            lines[-1] = f"{lines[-1]}\n</explanation>"
            content = '\n'.join(lines)
    
    if '<hypothesis>' not in content:
        if domain == "cs":
            hypothesis_text = "Computational parameters will significantly affect algorithmic performance metrics."
        else:
            hypothesis_text = "Optimal conditions within pH 3-8 and 20-37°C will significantly affect yeast biomass yield and metabolic activity."
        content = content.replace('</explanation>', f'</explanation>\n\n<hypothesis>{hypothesis_text}</hypothesis>')
    
    if '<followup>' not in content:
        if domain == "cs":
            followup_text = "What performance metrics should we optimize? Do you have baseline implementations?"
        else:
            followup_text = "What's your primary outcome measure? How many replicates are you planning?"
        content = f"{content}\n\n<followup>{followup_text}</followup>"
    
    return content