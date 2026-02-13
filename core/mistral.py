"""
core/mistral.py - COMPLETE OPTIMIZED VERSION (updated Feb 2026)
Mistral API integration with strict domain specialization & refusal
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import re

from core.config import (
    MISTRAL_API_KEY,
    MISTRAL_MODEL_NAME,
    MISTRAL_MAX_TOKENS,
    MISTRAL_TEMPERATURE,
    MISTRAL_TIMEOUT
)

# Import strict domain prefixes
from core.prompts import BIOMED_SYSTEM_PREFIX, CS_SYSTEM_PREFIX

logger = logging.getLogger("core.mistral")

# ==================== CONFIGURATION ====================
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7
EXPLANATION_TEMPERATURE = 0.8
RESEARCH_TEMPERATURE = 0.7
ANALYSIS_TEMPERATURE = 0.7

EXPLANATION_CONFIG = {
    "max_tokens": 4098,
    "temperature": 0.8,
    "top_p": 0.95,
    "presence_penalty": 0.1,
    "frequency_penalty": 0.1
}

RESEARCH_CONFIG = {
    "max_tokens": 1200,
    "temperature": 0.7,
    "top_p": 0.9,
    "presence_penalty": 0.05,
    "frequency_penalty": 0.05
}

# ==================== HELPER FUNCTIONS ====================

def is_explanation_query(query: str) -> bool:
    """Check if a query is asking for an explanation"""
    explanation_keywords = [
        "explain", "what is", "what are", "how does", "describe", "tell me about",
        "define", "meaning of", "understanding", "can you explain", "could you explain",
        "elaborate on", "break down", "walk me through", "help me understand",
        "what does it mean", "how it works", "can you describe", "explain to me",
        "teach me about", "clarify", "what's the difference", "compare"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in explanation_keywords)


def extract_explanation_topic(query: str) -> str:
    """Extract main topic from explanation-style query"""
    explanation_keywords = [
        "explain", "what is", "what are", "how does", "describe", "tell me about",
        "define", "meaning of", "understanding", "can you explain", "could you explain",
        "elaborate on", "break down", "walk me through", "help me understand"
    ]
    query_lower = query.lower()
    for keyword in explanation_keywords:
        if keyword in query_lower:
            parts = query_lower.split(keyword, 1)
            if len(parts) > 1:
                topic = parts[1].strip("?:. ")
                if topic:
                    return topic.capitalize()
    return query.strip("?:. ").capitalize()


# ==================== STRICT DOMAIN-AWARE EXPLANATION PROMPTS ====================

# In mistral.py, update the build_explanation_prompt function for CS:

def build_explanation_prompt(topic: str, domain: str = "general") -> str:
    """
    Return structured explanation prompt + strict domain guardrails.
    """
    refusal_instruction = """
IMPORTANT RULES:
- You are ONLY allowed to answer questions clearly within your domain.
- If this topic is outside your specialization, respond ONLY with:
  "This question is outside my specialization in {domain}. I cannot provide a reliable answer."
  Do NOT attempt to answer anyway.
"""

    if domain == "biomed":
        return f"""{BIOMED_SYSTEM_PREFIX}

{refusal_instruction.format(domain="biomedical science")}

Provide a comprehensive explanation of:

TOPIC: {topic}

STRUCTURE:
1. Introduction & Significance
2. Core Concepts & Definitions
3. Biological Mechanisms
4. Experimental Context
5. Clinical/Research Applications
6. Current Research & Future Directions

TONE: Clear, engaging, thorough. Use analogies where helpful.
DEPTH: 7-9 paragraphs. Comprehensive but accessible.
AUDIENCE: Researcher seeking deep understanding.
"""

    elif domain == "cs":
        return f"""{CS_SYSTEM_PREFIX}

{refusal_instruction.format(domain="computer science")}

Provide a comprehensive explanation of:

TOPIC: {topic}

CRITICAL RESPONSE FORMAT - YOU MUST USE THESE EXACT XML TAGS:

<enthusiasm>
[Brief enthusiastic greeting about the topic - 1-2 sentences.
Example: "Great question about {topic}! This is a fundamental concept in computer science."]
</enthusiasm>

<clarify>
[Ask 1-2 specific, practical clarifying questions. Examples:
 • What programming language or framework are you working with?
 • Do you have specific performance requirements (time/space complexity)?
 • What's your use case or application for this?

Then state: "I'll provide a comprehensive general explanation that should help across different contexts."]
</clarify>

<explanation>
[Comprehensive explanation covering all these sections:]

**Problem Context & Significance**
[2-3 sentences: Why this topic matters in CS, where it's used]

**Core Concepts & Definitions**
[3-4 sentences: Technical definitions with precision]

**Technical Details**
[Main section with:
 - Algorithmic approach or system design
 - Time/Space complexity analysis (Big O notation)
 - Code example or pseudo-code
 - Relevant data structures or design patterns]

**Implementation Considerations**
[3-4 sentences: Trade-offs, best practices, common pitfalls]

**Real-world Applications**
[2-3 sentences: Where and how this is used in practice]

**Current State & Trends**
[2-3 sentences: Latest developments, modern approaches]

TOTAL: 5-9 well-developed paragraphs with concrete examples.
</explanation>

<followup>
[List 2-3 follow-up questions to deepen understanding. Format as numbered list:
1. [Question about advanced topic or optimization]
2. [Question about related concepts or trade-offs]
3. [Question about practical implementation or use cases]]
</followup>

CRITICAL REMINDERS:
- You MUST wrap your response in the XML tags shown above
- Do NOT use markdown headers (##, ###) inside explanation - use **bold** for subsections
- Include actual code snippets where helpful
- Provide Big O analysis for algorithmic topics
- Be implementation-focused and practical

TONE: Technical but accessible. Precise, practical, code-oriented.
DEPTH: 7-9 paragraphs within <explanation> tag.
AUDIENCE: Developer/Researcher with some CS background.
"""

    else:
        return f"""
You are a general research assistant.

Provide a comprehensive explanation of:

TOPIC: {topic}

STRUCTURE:
1. Introduction & Context
2. Core Concepts
3. Detailed Analysis
4. Applications & Examples
5. Key Insights
6. Further Exploration

TONE: Clear, thorough, engaging.
DEPTH: 7-9 paragraphs.
"""
    
# ==================== MAIN API CALL ====================

async def call_mistral_api(
    prompt: str,
    max_tokens: int = 1500,
    temperature: float = None,
    system_prompt: str = None,
    explanation_mode: bool = False,
    domain: str = "general",
    **kwargs
) -> str:
    """
    Call Mistral chat completions endpoint with strict domain handling.
    """
    
    # Track if we're using a fallback
    using_fallback = False
    original_system_prompt = system_prompt
    
    # ── Force a safe default if system_prompt is None or empty ───────────────
    if not system_prompt or not isinstance(system_prompt, str) or system_prompt.strip() == "":
        # Only use fallback if we're NOT in explanation mode
        # (in explanation mode, we'll build a structured prompt anyway)
        if not explanation_mode and not is_explanation_query(prompt):
            using_fallback = True
            system_prompt = "You are a helpful research assistant with expertise in science and technology."

            # Optional: domain-specific minimal fallbacks
            if domain == "biomed":
                system_prompt = "You are a biomedical research assistant. Focus on biology, medicine, pharmacology, and related experimental sciences."
            elif domain == "cs":
                system_prompt = "You are an expert computer science assistant. Focus on algorithms, complexity, systems, machine learning, and software engineering."

    # ── Choose / override system prompt based on mode ─────────────────────────
    if explanation_mode or is_explanation_query(prompt):
        # Use structured explanation + strict guardrails
        topic = extract_explanation_topic(prompt)
        structured_prompt = build_explanation_prompt(topic, domain)
        # Prefer the structured one if we are in explanation mode
        system_prompt = structured_prompt  # ← overrides the fallback above
        
        # Reset using_fallback since we're now using a structured prompt
        using_fallback = False

    # ── Log warning only if we're actually using a fallback ─────────────────
    if using_fallback and original_system_prompt is None:
        logger.debug(
            f"System prompt was None/empty for domain={domain} → using fallback: "
            f"{system_prompt[:60]}..."
        )
    
    # ── Log what we're actually sending (very helpful for debugging) ─────────
    logger.info(
        f"System prompt | domain={domain} | mode={'explanation' if explanation_mode else 'normal'} | "
        f"length={len(system_prompt)} chars | preview: {system_prompt[:80].replace('\n', ' ')}..."
    )
    
    # ── Parameter tuning ─────────────────────────────────────────────────────
    if explanation_mode:
        temperature = temperature if temperature is not None else EXPLANATION_CONFIG["temperature"]
        max_tokens = max(max_tokens, EXPLANATION_CONFIG["max_tokens"])
    elif temperature is None:
        temperature = DEFAULT_TEMPERATURE

    # ── Build payload ────────────────────────────────────────────────────────
    payload = {
        "model": MISTRAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": min(max_tokens, MISTRAL_MAX_TOKENS),
        "temperature": max(0.1, min(1.0, temperature)),
        "top_p": kwargs.get("top_p", 0.9),
        "stream": False
    }

    if "presence_penalty" in kwargs:
        payload["presence_penalty"] = kwargs["presence_penalty"]
    if "frequency_penalty" in kwargs:
        payload["frequency_penalty"] = kwargs["frequency_penalty"]

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    logger.info(
        f"→ Mistral API call | domain={domain} | mode={'explanation' if explanation_mode else 'normal'} | "
        f"temp={payload['temperature']} | tokens={payload['max_tokens']}"
    )

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=MISTRAL_TIMEOUT)) as session:
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload
            ) as resp:

                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    usage = data.get("usage", {})
                    logger.info(
                        f"← Mistral success | prompt={usage.get('prompt_tokens')} → "
                        f"completion={usage.get('completion_tokens')}"
                    )
                    return content

                else:
                    error_text = await resp.text()
                    logger.error(f"Mistral HTTP {resp.status}: {error_text}")
                    if resp.status == 401:
                        return "Invalid API key."
                    if resp.status == 429:
                        return "Rate limit exceeded."
                    if resp.status == 422:
                        return f"MISTRAL API validation error (422): {error_text[:200]}"
                    return f"API error {resp.status}: {error_text[:200]}"

    except asyncio.TimeoutError:
        logger.error("Mistral timeout")
        return "Request timed out."
    except Exception as e:
        logger.exception("Mistral exception")
        return f"Connection error: {str(e)}"


async def generate_with_mistral(
    prompt: str,
    max_tokens: int = 2500,
    temperature: float = None,
    system_prompt: str = None,
    explanation_mode: bool = False,
    domain: str = "general",
    include_cot: bool = True,
    **kwargs
) -> str:
    """
    Enhanced Mistral generation with explanation optimization and chain-of-thought.
    
    Args:
        prompt: The user prompt
        max_tokens: Maximum tokens to generate
        temperature: Creativity level (0.0-1.0)
        system_prompt: Optional system prompt override
        explanation_mode: Whether to optimize for explanations
        domain: Domain context
        include_cot: Whether to include chain-of-thought reasoning
        **kwargs: Additional parameters
    
    Returns:
        Generated text (string)
    """
    
    cot_steps = []
    
    # Detect if this is an explanation request
    is_explanation = explanation_mode or is_explanation_query(prompt)
    
    if is_explanation and include_cot:
        # For explanations, use a two-step approach: reasoning then final answer
        reasoning_prompt = f"""First, think through how to explain this clearly:

Topic/Query: {prompt}

Think step by step:
1. What are the key concepts that need to be explained?
2. How can I structure this for maximum clarity?
3. What examples or analogies would be helpful?
4. What common misunderstandings should I address?
5. How can I make this both comprehensive and accessible?

Provide your reasoning:"""
        
        try:
            # Get reasoning
            reasoning = await call_mistral_api(
                prompt=reasoning_prompt,
                max_tokens=4098,
                temperature=0.7,  # Low temperature for focused reasoning
                system_prompt="You are a meticulous thinker. Break down the explanation step by step.",
                explanation_mode=False,
                domain=domain
            )
            
            cot_steps.append({
                "step": "explanation_planning",
                "reasoning": reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
            })
            
            # Build final prompt with reasoning
            enhanced_prompt = f"""Based on this reasoning plan:
{reasoning}

Now provide the complete, polished explanation for: {prompt}

Structure it clearly and comprehensively."""
            
            # Generate final explanation with optimized parameters
            final_response = await call_mistral_api(
                prompt=enhanced_prompt,
                max_tokens=max_tokens,
                temperature=temperature if temperature is not None else EXPLANATION_TEMPERATURE,
                system_prompt=system_prompt,
                explanation_mode=True,
                domain=domain,
                **kwargs
            )
            
            # Apply XML structure if needed
            if domain in ["cs", "biomed"]:
                final_response = enforce_xml_structure(final_response, prompt, domain)
            
            return final_response
            
        except Exception as e:
            logger.warning(f"Chain-of-thought explanation failed: {e}, falling back to direct generation")
            # Fall back to direct generation
    
    # Standard generation (with or without enhanced parameters)
    response = await call_mistral_api(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        explanation_mode=explanation_mode,
        domain=domain,
        **kwargs
    )
    
    # Apply XML structure if needed
    if domain in ["cs", "biomed"]:
        response = enforce_xml_structure(response, prompt, domain)
        logger.info(f"Applied XML structure enforcement for {domain} domain")
    
    return response


# Simplified version that returns just a string for backward compatibility
async def simple_generate_with_mistral(
    prompt: str,
    max_tokens: int = 1500,
    temperature: float = None,
    system_prompt: str = None,
    explanation_mode: bool = False,
    domain: str = "general",
    **kwargs
) -> str:
    """
    Simple wrapper for generate_with_mistral that returns only the text.
    For backward compatibility.
    """
    return await generate_with_mistral(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        explanation_mode=explanation_mode,
        domain=domain,
        include_cot=False,  # Disable CoT for simplicity
        **kwargs
    )

async def generate_with_mistral(
    prompt: str,
    max_tokens: int = 2500,
    temperature: float = None,
    system_prompt: str = None,
    explanation_mode: bool = False,
    domain: str = "general",
    include_cot: bool = True,
    **kwargs
) -> Tuple[str, List[Dict]]:
    """
    Enhanced Mistral generation with explanation optimization and chain-of-thought.
    
    Args:
        prompt: The user prompt
        max_tokens: Maximum tokens to generate
        temperature: Creativity level (0.0-1.0)
        system_prompt: Optional system prompt override
        explanation_mode: Whether to optimize for explanations
        domain: Domain context
        include_cot: Whether to include chain-of-thought reasoning
        **kwargs: Additional parameters
    
    Returns:
        Tuple of (generated_text, chain_of_thought_steps)
    """
    
    cot_steps = []
    
    # Detect if this is an explanation request
    is_explanation = explanation_mode or is_explanation_query(prompt)
    
    if is_explanation and include_cot:
        # For explanations, use a two-step approach: reasoning then final answer
        reasoning_prompt = f"""First, think through how to explain this clearly:

Topic/Query: {prompt}

Think step by step:
1. What are the key concepts that need to be explained?
2. How can I structure this for maximum clarity?
3. What examples or analogies would be helpful?
4. What common misunderstandings should I address?
5. How can I make this both comprehensive and accessible?

Provide your reasoning:"""
        
        try:
            # Get reasoning
            reasoning = await call_mistral_api(
                prompt=reasoning_prompt,
                max_tokens=2500,
                temperature=0.7,  # Low temperature for focused reasoning
                system_prompt="You are a meticulous thinker. Break down the explanation step by step.",
                explanation_mode=False
            )
            
            cot_steps.append({
                "step": "explanation_planning",
                "reasoning": reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
            })
            
            # Build final prompt with reasoning
            enhanced_prompt = f"""Based on this reasoning plan:
{reasoning}

Now provide the complete, polished explanation for: {prompt}

Structure it clearly and comprehensively."""
            
            # Generate final explanation with optimized parameters
            final_response = await call_mistral_api(
                prompt=enhanced_prompt,
                max_tokens=max_tokens,
                temperature=temperature if temperature is not None else EXPLANATION_TEMPERATURE,
                system_prompt=system_prompt,
                explanation_mode=True,
                domain=domain,
                **kwargs
            )
            
            cot_steps.append({
                "step": "final_explanation",
                "length": len(final_response)
            })
            
            return final_response, cot_steps
            
        except Exception as e:
            logger.warning(f"Chain-of-thought explanation failed: {e}, falling back to direct generation")
            # Fall back to direct generation
    
    # Standard generation (with or without enhanced parameters)
    response = await call_mistral_api(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        explanation_mode=explanation_mode,
        domain=domain,
        **kwargs
    )
    
    # Assign response to response_text first
    response_text = response
    
    if domain in ["cs", "biomed"]:
        response_text = enforce_xml_structure(response_text, prompt, domain)
        logger.info(f"Applied XML structure enforcement for {domain} domain")
    
    if include_cot:
        return response_text, cot_steps
    return response_text

async def generate_detailed_explanation(
    topic: str,
    domain: str = "general",
    audience_level: str = "intermediate",
    use_examples: bool = True,
    use_analogies: bool = True,
    include_structure: bool = True,
    max_tokens: int = 2000
) -> str:
    """
    Generate a detailed, comprehensive explanation.
    
    Args:
        topic: Topic to explain
        domain: Domain context
        audience_level: Target audience knowledge level
        use_examples: Whether to include examples
        use_analogies: Whether to use analogies
        include_structure: Whether to include explicit structure
        max_tokens: Maximum response length
    
    Returns:
        Detailed explanation string
    """
    
    # Build audience-specific guidance
    audience_guidance = {
        "beginner": "Assume the reader has little prior knowledge. Start with basics and build up gradually.",
        "intermediate": "Assume the reader has some background knowledge but wants deeper understanding.",
        "expert": "Assume the reader is knowledgeable but wants comprehensive technical details."
    }.get(audience_level, "Assume the reader has some background knowledge.")
    
    # Build domain-specific prompt
    if domain == "biomed":
        prompt = f"""You are a world-class biomedical educator. Provide a comprehensive explanation of:

TOPIC: {topic}

AUDIENCE: {audience_guidance}

STRUCTURE (please follow):
1. **Introduction & Significance**: Why this matters in biology/medicine
2. **Core Concepts**: Key terms, principles, and relationships
3. **Biological Mechanisms**: Step-by-step processes, pathways, interactions
4. **Experimental Context**: How this is studied, key methods
5. **Applications**: Medical, research, or practical applications
6. **Current Research**: Recent findings and future directions

{"7. **Examples**: Include concrete examples from research" if use_examples else ""}
{"8. **Analogies**: Use helpful analogies to clarify complex concepts" if use_analogies else ""}

TONE: Clear, engaging, thorough. Be comprehensive but accessible.
DEPTH: Aim for 8-10 detailed paragraphs.
"""
    
    elif domain == "cs":
        prompt = f"""You are an expert computer science educator. Provide a comprehensive explanation of:

TOPIC: {topic}

AUDIENCE: {audience_guidance}

STRUCTURE (please follow):
1. **Problem Context**: What computational problem this addresses
2. **Core Concepts**: Key terms, algorithms, data structures
3. **Technical Details**: Mechanisms, implementations, complexity
4. **Practical Considerations**: Implementation tips, trade-offs
5. **Applications**: Real-world use cases and impact
6. **Comparisons**: How this compares to alternatives
7. **Current State**: Recent developments and future directions

{"8. **Examples**: Include code examples or conceptual examples" if use_examples else ""}
{"9. **Analogies**: Use helpful analogies to explain abstract concepts" if use_analogies else ""}

TONE: Clear, technical but accessible. Be comprehensive and precise.
DEPTH: Aim for 8-10 detailed paragraphs.
"""
    
    else:
        prompt = f"""You are a research assistant and educator. Provide a comprehensive explanation of:

TOPIC: {topic}

AUDIENCE: {audience_guidance}

STRUCTURE (please follow):
1. **Introduction**: Context and importance
2. **Core Concepts**: Key terms and fundamental principles
3. **Detailed Analysis**: Mechanisms, relationships, evidence
4. **Applications**: Practical uses and implications
5. **Key Insights**: Most important takeaways
6. **Further Exploration**: Where to learn more

{"7. **Examples**: Include concrete examples to illustrate concepts" if use_examples else ""}
{"8. **Analogies**: Use helpful analogies for clarity" if use_analogies else ""}

TONE: Clear, thorough, engaging. Balance depth with accessibility.
DEPTH: Aim for 8-10 detailed paragraphs.
"""
    
    if not include_structure:
        # Remove explicit structure instructions but keep content guidance
        prompt = prompt.replace("STRUCTURE (please follow):", "Provide a comprehensive explanation that covers:")
    
    # Generate the explanation
    explanation, cot_steps = await generate_with_mistral(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=EXPLANATION_TEMPERATURE,
        explanation_mode=True,
        domain=domain,
        include_cot=True
    )
    
    logger.info(f"✅ Generated detailed explanation for '{topic[:50]}...' ({len(explanation)} chars)")
    return explanation

# ==================== XML STRUCTURE ENFORCEMENT ====================

def enforce_xml_structure(text: str, query: str = "", domain: str = "biomed") -> str:
    """
    Ensure response has proper XML structure.
    Enhanced to handle CS domain with clarifying questions.
    """
    
    # Check if text already has XML tags
    has_enthusiasm = "<enthusiasm>" in text and "</enthusiasm>" in text
    has_explanation = "<explanation>" in text and "</explanation>" in text
    has_hypothesis = "<hypothesis>" in text and "</hypothesis>" in text
    has_followup = "<followup>" in text and "</followup>" in text
    has_clarify = "<clarify>" in text and "</clarify>" in text
    
    # If it already has all needed tags for the domain, return as-is
    if domain == "cs":
        # CS needs: enthusiasm, clarify, explanation, followup
        if has_enthusiasm and has_clarify and has_explanation and has_followup:
            logger.info("✅ CS response has complete XML structure")
            return text
    else:
        # Biomed needs: enthusiasm, explanation, (hypothesis OR followup)
        if has_enthusiasm and has_explanation and (has_hypothesis or has_followup):
            logger.info("✅ Biomed response has complete XML structure")
            return text
    
    logger.warning(f"⚠️ Response missing XML tags for {domain} domain - adding structure")
    
    # Check if this looks like an explanation response
    is_explanation_response = is_explanation_query(query) or "explanation" in text.lower() or "explain" in text.lower()
    
    # Extract or create sections
    enthusiasm_text = ""
    explanation_text = ""
    hypothesis_text = ""
    followup_text = ""
    clarify_text = ""
    
    # Extract existing sections if present
    if has_enthusiasm:
        start = text.find("<enthusiasm>") + len("<enthusiasm>")
        end = text.find("</enthusiasm>")
        if start > len("<enthusiasm>") - 1 and end > start:
            enthusiasm_text = text[start:end].strip()
    
    if has_explanation:
        start = text.find("<explanation>") + len("<explanation>")
        end = text.find("</explanation>")
        if start > len("<explanation>") - 1 and end > start:
            explanation_text = text[start:end].strip()
    
    if has_hypothesis:
        start = text.find("<hypothesis>") + len("<hypothesis>")
        end = text.find("</hypothesis>")
        if start > len("<hypothesis>") - 1 and end > start:
            hypothesis_text = text[start:end].strip()
    
    if has_followup:
        start = text.find("<followup>") + len("<followup>")
        end = text.find("</followup>")
        if start > len("<followup>") - 1 and end > start:
            followup_text = text[start:end].strip()
    
    if has_clarify:
        start = text.find("<clarify>") + len("<clarify>")
        end = text.find("</clarify>")
        if start > len("<clarify>") - 1 and end > start:
            clarify_text = text[start:end].strip()
    
    # If no XML found, use the entire text as explanation
    if not any([has_enthusiasm, has_explanation, has_hypothesis, has_followup, has_clarify]):
        if is_explanation_response:
            explanation_text = text.strip()
        else:
            # Split intelligently
            paragraphs = text.strip().split('\n\n')
            if paragraphs:
                if len(paragraphs[0]) < 200 and any(word in paragraphs[0].lower() for word in ['great', 'interesting', 'fascinating', 'excellent', 'wonderful']):
                    enthusiasm_text = paragraphs[0]
                    explanation_text = '\n\n'.join(paragraphs[1:])
                else:
                    explanation_text = text.strip()
    
    # Ensure we have enthusiasm text
    if not enthusiasm_text:
        if is_explanation_response:
            topic = extract_explanation_topic(query)
            if domain == "cs":
                enthusiasm_text = f"Excellent question about {topic}! This is an important concept in computer science."
            else:
                enthusiasm_text = f"Great question about {topic}! I'd be happy to provide a comprehensive explanation."
        else:
            enthusiasm_text = "Great research question! Let me analyze this thoroughly."
    
    # Ensure we have explanation text
    if not explanation_text:
        explanation_text = text.strip() if text.strip() else "I've analyzed your query and here are the key insights..."
    
    # Ensure we have clarify text for CS domain
    if domain == "cs" and not clarify_text:
        clarify_text = """Before diving deep, I'd like to clarify a few things to give you the most relevant answer:

1. What programming language or framework are you working with?
2. Do you have any specific performance requirements or constraints?
3. What's your intended use case or application?

I'll provide a comprehensive general explanation that should be helpful across different contexts."""
    
    # Ensure we have hypothesis (for biomed research queries) or followup
    if not hypothesis_text and domain == "biomed" and not is_explanation_response:
        hypothesis_text = "Based on this analysis, we can hypothesize that the key parameters will significantly influence the experimental outcomes."
    
    if not followup_text:
        if domain == "cs":
            followup_text = """1. Would you like to see a complete implementation example in a specific language?
2. Are you interested in optimization techniques or alternative approaches?
3. How does this compare to related algorithms or patterns in terms of performance?"""
        elif is_explanation_response:
            followup_text = """1. Would you like me to go deeper into any specific aspect?
2. How do you plan to apply this understanding in your work?
3. Are there related topics you'd like me to explain?"""
        else:
            followup_text = """1. What specific measurements are you planning?
2. How many replicates will you run?
3. What is your primary outcome measure?"""
    
    # Build structured response
    structured_response = f"<enthusiasm>{enthusiasm_text}</enthusiasm>\n\n"
    
    # Add clarify section for CS domain
    if clarify_text and domain == "cs":
        structured_response += f"<clarify>{clarify_text}</clarify>\n\n"
    
    structured_response += f"<explanation>{explanation_text}</explanation>\n\n"
    
    # Add hypothesis for biomed (but not for explanations)
    if hypothesis_text and domain == "biomed" and not is_explanation_response:
        structured_response += f"<hypothesis>{hypothesis_text}</hypothesis>\n\n"
    
    structured_response += f"<followup>{followup_text}</followup>"
    
    logger.info(f"✅ Added XML structure for {domain} domain")
    return structured_response
# ==================== QUICK EXPLANATION ENDPOINT ====================

async def quick_explanation(query: str, domain: str = "general") -> str:
    """
    Generate a quick explanation for simple queries.
    Optimized for speed and clarity.
    """
    
    # Simple prompt for quick explanations
    prompt = f"""Provide a clear, concise explanation of: {query}

Keep it to 2-3 paragraphs. Focus on:
1. Key definition or concept
2. How it works or what it means
3. Why it's important or relevant

Be direct and to the point."""
    
    try:
        explanation = await call_mistral_api(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            system_prompt="You are a helpful assistant who provides clear, concise explanations.",
            explanation_mode=True,
            domain=domain
        )
        return explanation.strip()
    except Exception as e:
        logger.error(f"Quick explanation failed: {e}")
        return f"I'll explain {query}: [Explanation generation failed]"

# ==================== HEALTH CHECK ====================

async def check_mistral_health() -> Dict[str, Any]:
    """Check if Mistral API is working"""
    try:
        test_response = await call_mistral_api(
            prompt="test",
            max_tokens=5,
            temperature=0.1
        )
        
        return {
            "status": "healthy" if test_response else "unhealthy",
            "response_received": bool(test_response),
            "test_response": test_response[:100] if test_response else None,
            "api_key_configured": bool(MISTRAL_API_KEY),
            "model": MISTRAL_MODEL_NAME
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "api_key_configured": bool(MISTRAL_API_KEY),
            "model": MISTRAL_MODEL_NAME
        }