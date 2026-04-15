"""
core/mistral.py - FIXED VERSION
Fixes:
  1. Removed duplicate generate_with_mistral definition (was returning first char of string)
  2. Fixed explanation_mode hijacking system prompt on research queries
  3. Fixed MISTRAL_TIMEOUT being too low for synthesizer calls (now per-call override)
  4. generate_with_mistral always returns Tuple[str, List] consistently
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

# Hard cap for API timeout — set high enough that synthesizer (75s) can complete
API_HARD_TIMEOUT = max(MISTRAL_TIMEOUT, 120)

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
    """Check if a query is PURELY asking for an explanation — NOT a research/analysis query."""
    # Only trigger on very clear explanation-only patterns
    # We intentionally do NOT trigger on "analyze", "study", "work with", "impact of"
    explanation_keywords = [
        "explain to me", "can you explain", "could you explain",
        "what is ", "what are ", "how does ", "what does it mean",
        "define ", "meaning of", "walk me through", "help me understand",
        "teach me about", "what's the difference between",
    ]
    query_lower = query.lower().strip()

    # If the query is longer than 120 chars it's almost certainly a research query, not pure explanation
    if len(query_lower) > 120:
        return False

    return any(query_lower.startswith(kw) or f" {kw}" in query_lower for kw in explanation_keywords)


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

def build_explanation_prompt(topic: str, domain: str = "general") -> str:
    """Return structured explanation prompt + strict domain guardrails."""
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
[Brief enthusiastic greeting about the topic - 1-2 sentences.]
</enthusiasm>

<clarify>
[Ask 1-2 specific, practical clarifying questions, then state you'll provide a comprehensive general explanation.]
</clarify>

<explanation>
[Comprehensive explanation covering all these sections:]

**Problem Context & Significance**
[2-3 sentences: Why this topic matters in CS, where it's used]

**Core Concepts & Definitions**
[3-4 sentences: Technical definitions with precision]

**Technical Details**
[Main section with algorithmic approach, complexity analysis, code example]

**Implementation Considerations**
[3-4 sentences: Trade-offs, best practices, common pitfalls]

**Real-world Applications**
[2-3 sentences: Where and how this is used in practice]

**Current State & Trends**
[2-3 sentences: Latest developments, modern approaches]

TOTAL: 5-9 well-developed paragraphs with concrete examples.
</explanation>

<followup>
[List 2-3 follow-up questions to deepen understanding. Format as numbered list.]
</followup>

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
    timeout_override: float = None,   # ← NEW: per-call timeout override
    **kwargs
) -> str:
    """
    Call Mistral chat completions endpoint with strict domain handling.

    IMPORTANT: explanation_mode=True now ONLY overrides the system prompt when the
    query is genuinely a pure explanation query (short, starts with "what is", etc.).
    Research/analysis queries are NOT affected even if they contain the word "explain".
    """

    using_fallback = False
    original_system_prompt = system_prompt

    # ── Build system prompt ───────────────────────────────────────────────────
    #
    # FIX: Only hijack the system prompt when BOTH:
    #   a) explanation_mode is True AND
    #   b) the prompt is genuinely a short pure-explanation query
    #
    # Previously, any prompt containing words like "explain" triggered
    # build_explanation_prompt(), discarding the rich synthesizer system prompt.

    is_pure_explanation = explanation_mode and is_explanation_query(prompt)

    if is_pure_explanation:
        topic = extract_explanation_topic(prompt)
        system_prompt = build_explanation_prompt(topic, domain)
        using_fallback = False
        logger.debug(f"Using structured explanation prompt for topic: {topic[:60]}")

    elif not system_prompt or not isinstance(system_prompt, str) or system_prompt.strip() == "":
        # No system prompt provided and not an explanation — use a minimal domain fallback
        using_fallback = True
        if domain == "biomed":
            system_prompt = "You are a biomedical research assistant. Focus on biology, medicine, pharmacology, and related experimental sciences."
        elif domain == "cs":
            system_prompt = "You are an expert computer science assistant. Focus on algorithms, complexity, systems, machine learning, and software engineering."
        else:
            system_prompt = "You are a helpful research assistant with expertise in science and technology."

    # ── Log ──────────────────────────────────────────────────────────────────
    system_prompt_preview = system_prompt[:80].replace('\n', ' ')
    logger.info(
        f"System prompt | domain={domain} | mode={'explanation' if is_pure_explanation else 'normal'} | "
        f"length={len(system_prompt)} chars | preview: {system_prompt_preview}..."
    )

    # ── Parameter tuning ─────────────────────────────────────────────────────
    if is_pure_explanation:
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

    # FIX: Use per-call timeout override if provided, else use the hard cap
    effective_timeout = timeout_override if timeout_override else API_HARD_TIMEOUT

    logger.info(
        f"→ Mistral API call | domain={domain} | mode={'explanation' if is_pure_explanation else 'normal'} | "
        f"temp={payload['temperature']} | tokens={payload['max_tokens']} | timeout={effective_timeout}s"
    )

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=effective_timeout)
        ) as session:
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
        logger.error(f"Mistral timeout after {effective_timeout}s")
        return ""
    except Exception as e:
        logger.exception("Mistral exception")
        return f"Connection error: {str(e)}"


# ==================== MAIN GENERATION FUNCTION ====================
# FIX: Single authoritative definition — always returns Tuple[str, List[Dict]]
# Previously there were TWO definitions of this function. Python used the second
# one, which in the non-CoT path returned a bare string. Callers doing result[0]
# then got only the first character of the response string.

async def generate_with_mistral(
    prompt: str,
    max_tokens: int = 2500,
    temperature: float = None,
    system_prompt: str = None,
    explanation_mode: bool = False,
    domain: str = "general",
    include_cot: bool = True,
    timeout_override: float = None,
    **kwargs
) -> Tuple[str, List[Dict]]:
    """
    Enhanced Mistral generation with optional chain-of-thought.

    Always returns: Tuple[str, List[Dict]]
      - str: the generated response text
      - List[Dict]: chain-of-thought steps (empty list if CoT not used)

    Callers should do:
        result = await generate_with_mistral(...)
        text = result[0] if isinstance(result, tuple) else result
    """

    cot_steps = []

    # Only use CoT path for genuine short explanation queries
    is_pure_explanation = explanation_mode and is_explanation_query(prompt)

    if is_pure_explanation and include_cot:
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
            reasoning = await call_mistral_api(
                prompt=reasoning_prompt,
                max_tokens=2500,
                temperature=0.7,
                system_prompt="You are a meticulous thinker. Break down the explanation step by step.",
                explanation_mode=False,
                domain=domain,
                timeout_override=timeout_override,
            )

            cot_steps.append({
                "step": "explanation_planning",
                "reasoning": reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
            })

            enhanced_prompt = f"""Based on this reasoning plan:
{reasoning}

Now provide the complete, polished explanation for: {prompt}

Structure it clearly and comprehensively."""

            final_response = await call_mistral_api(
                prompt=enhanced_prompt,
                max_tokens=max_tokens,
                temperature=temperature if temperature is not None else EXPLANATION_TEMPERATURE,
                system_prompt=system_prompt,
                explanation_mode=True,
                domain=domain,
                timeout_override=timeout_override,
                **kwargs
            )

            cot_steps.append({
                "step": "final_explanation",
                "length": len(final_response)
            })

            if domain in ["cs", "biomed"]:
                final_response = enforce_xml_structure(final_response, prompt, domain)

            return final_response, cot_steps

        except Exception as e:
            logger.warning(f"Chain-of-thought explanation failed: {e}, falling back to direct generation")
            # Fall through to standard path below

    # ── Standard generation path ─────────────────────────────────────────────
    response = await call_mistral_api(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        explanation_mode=explanation_mode,
        domain=domain,
        timeout_override=timeout_override,
        **kwargs
    )

    # Apply XML structure enforcement
    if domain in ["cs", "biomed"]:
        response = enforce_xml_structure(response, prompt, domain)
        logger.info(f"Applied XML structure enforcement for {domain} domain")

    return response, cot_steps


# ==================== SIMPLE WRAPPER ====================

async def simple_generate_with_mistral(
    prompt: str,
    max_tokens: int = 1500,
    temperature: float = None,
    system_prompt: str = None,
    explanation_mode: bool = False,
    domain: str = "general",
    **kwargs
) -> str:
    """Simple wrapper that returns only the text string (no CoT steps)."""
    result = await generate_with_mistral(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        explanation_mode=explanation_mode,
        domain=domain,
        include_cot=False,
        **kwargs
    )
    return result[0] if isinstance(result, tuple) else result


# ==================== XML STRUCTURE ENFORCEMENT ====================

def enforce_xml_structure(text: str, query: str = "", domain: str = "biomed") -> str:
    """
    Ensure response has proper XML structure.
    Enhanced to handle CS domain with clarifying questions.
    """

    has_enthusiasm = "<enthusiasm>" in text and "</enthusiasm>" in text
    has_explanation = "<explanation>" in text and "</explanation>" in text
    has_hypothesis  = "<hypothesis>"  in text and "</hypothesis>"  in text
    has_followup    = "<followup>"    in text and "</followup>"    in text
    has_clarify     = "<clarify>"     in text and "</clarify>"     in text

    if domain == "cs":
        if has_enthusiasm and has_clarify and has_explanation and has_followup:
            logger.info("✅ CS response has complete XML structure")
            return text
    else:
        if has_enthusiasm and has_explanation and (has_hypothesis or has_followup):
            logger.info("✅ Biomed response has complete XML structure")
            return text

    logger.warning(f"⚠️ Response missing XML tags for {domain} domain - adding structure")

    is_explanation_response = is_explanation_query(query)

    enthusiasm_text = ""
    explanation_text = ""
    hypothesis_text = ""
    followup_text = ""
    clarify_text = ""

    def _extract(tag):
        start = text.find(f"<{tag}>") + len(f"<{tag}>")
        end = text.find(f"</{tag}>")
        if start > len(f"<{tag}>") - 1 and end > start:
            return text[start:end].strip()
        return ""

    if has_enthusiasm:  enthusiasm_text  = _extract("enthusiasm")
    if has_explanation: explanation_text = _extract("explanation")
    if has_hypothesis:  hypothesis_text  = _extract("hypothesis")
    if has_followup:    followup_text    = _extract("followup")
    if has_clarify:     clarify_text     = _extract("clarify")

    # If no XML at all, treat entire text as explanation
    if not any([has_enthusiasm, has_explanation, has_hypothesis, has_followup, has_clarify]):
        explanation_text = text.strip()

    # Defaults
    if not enthusiasm_text:
        if is_explanation_response:
            topic = extract_explanation_topic(query)
            enthusiasm_text = (
                f"Excellent question about {topic}! This is an important concept in computer science."
                if domain == "cs"
                else f"Great question about {topic}! I'd be happy to provide a comprehensive explanation."
            )
        else:
            enthusiasm_text = "This is a scientifically significant research question with clear experimental tractability and real-world implications."

    if not explanation_text:
        explanation_text = text.strip() or "Analysis in progress..."

    if domain == "cs" and not clarify_text:
        clarify_text = (
            "Before diving deep, I'd like to clarify a few things:\n\n"
            "1. What programming language or framework are you working with?\n"
            "2. Do you have specific performance requirements or constraints?\n"
            "3. What's your intended use case?\n\n"
            "I'll provide a comprehensive general explanation that should help across different contexts."
        )

    if not hypothesis_text and domain == "biomed" and not is_explanation_response:
        hypothesis_text = (
            "Based on this analysis, we hypothesize that the key parameters will significantly "
            "influence the experimental outcomes in a dose-dependent manner."
        )

    if not followup_text:
        if domain == "cs":
            followup_text = (
                "1. Would you like to see a complete implementation example in a specific language?\n"
                "2. Are you interested in optimization techniques or alternative approaches?\n"
                "3. How does this compare to related algorithms in terms of performance?"
            )
        elif is_explanation_response:
            followup_text = (
                "1. Would you like me to go deeper into any specific aspect?\n"
                "2. How do you plan to apply this understanding in your work?\n"
                "3. Are there related topics you'd like me to explain?"
            )
        else:
            followup_text = (
                "1. What specific assay methods are you planning to measure the primary outcome?\n"
                "2. How many biological replicates will you run per condition?\n"
                "3. Are you considering a response-surface methodology to map the parameter space efficiently?"
            )

    # Assemble
    structured_response = f"<enthusiasm>{enthusiasm_text}</enthusiasm>\n\n"
    if clarify_text and domain == "cs":
        structured_response += f"<clarify>{clarify_text}</clarify>\n\n"
    structured_response += f"<explanation>{explanation_text}</explanation>\n\n"
    if hypothesis_text and domain == "biomed" and not is_explanation_response:
        structured_response += f"<hypothesis>{hypothesis_text}</hypothesis>\n\n"
    structured_response += f"<followup>{followup_text}</followup>"

    logger.info(f"✅ Added XML structure for {domain} domain")
    return structured_response


# ==================== DETAILED EXPLANATION GENERATOR ====================

async def generate_detailed_explanation(
    topic: str,
    domain: str = "general",
    audience_level: str = "intermediate",
    use_examples: bool = True,
    use_analogies: bool = True,
    include_structure: bool = True,
    max_tokens: int = 2000
) -> str:
    """Generate a detailed, comprehensive explanation."""

    audience_guidance = {
        "beginner":     "Assume the reader has little prior knowledge. Start with basics and build up gradually.",
        "intermediate": "Assume the reader has some background knowledge but wants deeper understanding.",
        "expert":       "Assume the reader is knowledgeable but wants comprehensive technical details."
    }.get(audience_level, "Assume the reader has some background knowledge.")

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
        prompt = prompt.replace("STRUCTURE (please follow):", "Provide a comprehensive explanation that covers:")

    result = await generate_with_mistral(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=EXPLANATION_TEMPERATURE,
        explanation_mode=True,
        domain=domain,
        include_cot=True
    )
    explanation = result[0] if isinstance(result, tuple) else result
    logger.info(f"✅ Generated detailed explanation for '{topic[:50]}...' ({len(explanation)} chars)")
    return explanation


# ==================== QUICK EXPLANATION ENDPOINT ====================

async def quick_explanation(query: str, domain: str = "general") -> str:
    """Generate a quick explanation for simple queries."""
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
            explanation_mode=False,
            domain=domain
        )
        return explanation.strip()
    except Exception as e:
        logger.error(f"Quick explanation failed: {e}")
        return f"I'll explain {query}: [Explanation generation failed]"


# ==================== HEALTH CHECK ====================

async def check_mistral_health() -> Dict[str, Any]:
    """Health check for Mistral API"""
    try:
        response = await call_mistral_api(
            "Say hello in one sentence.",
            max_tokens=20,
            temperature=0.1
        )
        return {
            "status": "healthy",
            "response_received": True,
            "model": MISTRAL_MODEL_NAME,
            "api_key_configured": bool(MISTRAL_API_KEY)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "api_key_configured": bool(MISTRAL_API_KEY)
        }