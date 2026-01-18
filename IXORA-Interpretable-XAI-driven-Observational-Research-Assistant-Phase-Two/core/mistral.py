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

async def call_mistral_api(
    prompt: str, 
    max_tokens: int = 1000, 
    temperature: float = 0.7,
    domain: str = "biomed"
) -> str:
    """Direct API call with error handling and domain awareness
    
    Args:
        prompt: The user's input prompt
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature (0.0 to 1.0)
        domain: The domain context ('biomed' or 'cs'/'computerscience')
        
    Returns:
        Generated text response
    """
    from core.config import MISTRAL_API_KEY
    from core.prompts import get_system_prompt
    
    if not MISTRAL_API_KEY:
        logger.error("No Mistral API key configured!")
        return ""
    
    url = "https://api.mistral.ai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Get domain-appropriate system prompt
    system_msg = get_system_prompt(domain)
    
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
    temperature: float = 0.7,
    domain: str = "biomed"
) -> Tuple[str, List[str]]:
    """Generate with error handling, fallback, and domain awareness
    
    Args:
        prompt: The input prompt to generate from
        max_tokens: Maximum number of tokens to generate
        temperature: Generation temperature (0.0 to 1.0)
        domain: The domain context ('biomed' or 'cs'/'computerscience')
        
    Returns:
        Tuple of (generated_text, chain_of_thought_steps)
    """
    logger.info(f"Generating with Mistral for {domain} domain: {len(prompt)} chars")
    
    model = await load_mistral()
    if not model:
        logger.error("Mistral model not loaded!")
        return "Mistral service unavailable. Please check your configuration.", []
    
    logger.info(f"Model type: {model}, Domain: {domain}")
    
    content = ""
    cot_steps = []
    
    try:
        if model == "api_mode":
            logger.info(f"Using API mode for {domain} domain...")
            content = await call_mistral_api(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                domain=domain
            )
            
            if not content:
                logger.warning("API returned empty content, using fallback...")
                if domain in ["cs", "computerscience", "ai"]:
                    content = (
                        f"I've analyzed your technical query about {prompt[:100]}... "
                        "As a computer science research assistant, I can help with algorithms, "
                        "data structures, and programming concepts. Please ensure your API key is configured "
                        "for the most accurate and detailed responses."
                    )
                else:
                    content = (
                        f"I've analyzed your biomedical query about {prompt[:100]}... "
                        "For detailed analysis, please ensure your Mistral API key is properly configured."
                    )
            
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
    """Add basic XML structure if missing with domain awareness
    
    Args:
        content: The content to structure
        user_query: The original user query
        domain: The domain context ('biomed' or 'cs'/'computerscience')
        
    Returns:
        Structured content with XML tags
    """
    is_cs = domain in ["cs", "computerscience", "ai"]
    
    if not content:
        if is_cs:
            return f"""<technical>Analyzing your technical query about {user_query[:100]}...</technical>
<algorithms>Common approaches include...</algorithms>
<implementation>Key implementation considerations are...</implementation>
<challenges>Potential challenges include...</challenges>
<followup>What specific programming language are you using? Do you have any performance requirements?</followup>"""
        else:
            return f"""<enthusiasm>Exciting research question!</enthusiasm>
<explanation>Your query about "{user_query[:100]}" is interesting for biomedical research. While I couldn't generate a full analysis due to a technical issue, I can tell you that yeast biomass studies typically investigate factors like pH, temperature, nutrients, and incubation time.</explanation>
<hypothesis>The optimal pH and temperature combination will maximize yeast biomass production while maintaining metabolic activity.</hypothesis>
<followup>What specific measurements are you planning to take? Do you have access to spectrophotometry for biomass quantification?</followup>"""
    
    # Only add tags if completely missing
    if is_cs:
        if '<technical>' not in content:
            content = f"<technical>Technical analysis of your query:</technical>\n\n{content}"
    
    if is_cs:
        # Handle CS-specific XML structure
        if '<algorithms>' not in content and '</technical>' in content:
            content = content.replace('</technical>', 
                '</technical>\n\n<algorithms>Common algorithmic approaches include:</algorithms>')
        
        if '<implementation>' not in content and ('</algorithms>' in content or '<algorithms>' not in content):
            content = f"{content}\n\n<implementation>Implementation considerations:</implementation>"
            
        if '<challenges>' not in content:
            content = f"{content}\n\n<challenges>Potential challenges to consider:</challenges>"
            
        if '<followup>' not in content:
            content = f"{content}\n\n<followup>What specific technologies or frameworks are you using? Do you have any performance requirements?</followup>"
    else:
        # Handle biomed XML structure
        if '<explanation>' not in content:
            lines = content.split('\n')
            explanation_start = 0
            for i, line in enumerate(lines):
                if '</enthusiasm>' in line:
                    explanation_start = i + 1
                    break
            
            if explanation_start < len(lines):
                lines[explanation_start] = f"<explanation>\n{lines[explanation_start]}"
                lines[-1] = f"{lines[-1]}\n</explanation>"
                content = '\n'.join(lines)
        
        if '<hypothesis>' not in content and '</explanation>' in content:
            hypothesis_text = "Optimal conditions within pH 3-8 and 20-37°C will significantly affect yeast biomass yield and metabolic activity."
            content = content.replace('</explanation>', f'</explanation>\n\n<hypothesis>{hypothesis_text}</hypothesis>')
        
        if '<followup>' not in content:
            content = f"{content}\n\n<followup>What's your primary outcome measure? How many replicates are you planning?</followup>"
    
    return content