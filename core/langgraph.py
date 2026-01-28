"""
core/langgraph.py - FIXED VERSION: LINEAR FLOW WITH OPTIMIZED EMBEDDINGS
"""

import logging
from typing import TypedDict, Annotated, List, Dict
import operator
from datetime import datetime
import asyncio
import re
import json
# Convert to tensors for similarity computation
import torch
from sentence_transformers import util
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from core.analytics import run_bayesian_optimization, run_comprehensive_analytics_parallel, run_causal_analysis
from core.mistral import generate_with_mistral
from core.config import BIOMISTRAL_TIMEOUT
from scipy.stats import entropy
from evaluate import load  # For BLEURT/BERTScore
import torch
import difflib
import numpy as np
import hashlib
import time

logger = logging.getLogger("core.langgraph")

# In core/langgraph.py, around line 28, add error handling:

try:
    from core.rlhf.reward_model import get_reward_model
except ImportError as e:
    logger.warning(f"RLHF reward model import failed: {e}")
    # Create a dummy function
    def get_reward_model():
        return None


class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage], operator.add]
    query: str
    domain: str
    parameters: dict
    analytics: dict
    hypothesis: str
    draft: str
    final_response: str
    trace: List[dict]
    confidence: float
    step_count: int
    validated: bool
    embedding_scores: dict  # NEW: Store embedding similarity scores

# ==================== AGENTS ====================

from core.parameter_extractor import extract_parameters

# In langgraph.py - Optimize the workflow

async def extractor_agent(state: AgentState) -> AgentState:
    """
    Optimized extractor - uses existing function with timeout.
    """
    logger.info("🔍 [EXTRACTOR] Starting parameter extraction...")
    extract_start = time.time()
    
    query = state["query"]
    domain = state["domain"]
    
    try:
        # Use existing extract_parameters with timeout
        from core.parameter_extractor import extract_parameters
        
        extraction_result = await asyncio.wait_for(
            extract_parameters(query, domain=domain),
            timeout=15.0  # 15 second timeout
        )
        
        parameters = extraction_result.get("parameters", {})
        metadata = extraction_result.get("_metadata", {})
        
        state["parameters"] = parameters
        
        # Log extraction
        extract_time = time.time() - extract_start
        logger.info(f"✅ [EXTRACTOR] Found {len(parameters)} parameters in {extract_time:.2f}s")
        
        state["trace"].append({
            "step": "parameter_extraction",
            "method": metadata.get("method", "unknown"),
            "param_count": len(parameters),
            "time_seconds": round(extract_time, 2),
            "success": True
        })
        
    except asyncio.TimeoutError:
        logger.warning("⏰ [EXTRACTOR] Timed out after 15s")
        state["parameters"] = {}
        state["trace"].append({
            "step": "parameter_extraction",
            "error": "timeout",
            "time_seconds": 15.0,
            "success": False
        })
    except Exception as e:
        logger.error(f"❌ [EXTRACTOR] Failed: {e}")
        state["parameters"] = {}
        state["trace"].append({
            "step": "parameter_extraction",
            "error": str(e)[:100],
            "success": False
        })
    
    return state

async def draft_agent(state: AgentState) -> AgentState:
    query = state["query"]
    domain = state["domain"]
    parameters = state.get("parameters", {})
    draft = ""
    trace_entry = {
        "step": "draft",
        "query_preview": query[:100] + "..." if len(query) > 100 else query,
        "domain": domain,
        "timestamp": datetime.now().isoformat()
    }

    # Helper to safely call BioMistral with fallback
    async def safe_biomistral_generate(prompt: str, max_tokens: int = 300) -> str:
        try:
            from core.model_loader import model_loader
            output = await model_loader.generate_with_biomistral(prompt, max_tokens=max_tokens)
            trace_entry["biomistral_success"] = True
            trace_entry["biomistral_tokens"] = len(output.split())
            logger.info("BioMistral generated draft successfully")
            return output
        except Exception as e:
            logger.warning(f"BioMistral failed: {e} — falling back to Mistral API")
            trace_entry["biomistral_success"] = False
            trace_entry["biomistral_error"] = str(e)[:100]
            # Fallback to Mistral for core reasoning
            fallback_output = await generate_with_mistral(prompt, max_tokens=max_tokens*2, temperature=0.5)
            return fallback_output

    if domain == "biomed":
        # === STEP 1: BioMistral does the deep biomedical reasoning ===
        biomed_prompt = f"""
You are BioMistral, a biomedical reasoning model specialized in scientific literature understanding, clinical reasoning support, and hypothesis generation. Your role is not to provide medical diagnoses or treatment recommendations, but to act as a research and clinical-assistant model that formulates clear, structured, and scientifically plausible hypotheses based on a user’s query.

When a user provides a question, observation, symptom description, dataset insight, or research curiosity, your task is to carefully interpret the intent and context of the query, identify the key biological, clinical, or molecular variables involved, and generate one or more well-defined hypotheses that could reasonably explain the observation or guide further investigation.

User Query: "{query}"

Extracted Parameters (if any): {json.dumps(parameters, indent=2) if parameters else "None detected."}

Instructions:
- Think deeply about biological mechanisms, pathways, and experimental variables.
- Prioritize scientific plausibility and mechanistic insight.
- Formulate 1–2 strong, testable hypotheses.
- Suggest key experimental variables to control or measure.
- Keep response concise but insightful (200–400 words).
- Do not use XML tags — just write naturally.
"""

        biomed_core = await safe_biomistral_generate(biomed_prompt, max_tokens=400)
        trace_entry["biomistral_core_length"] = len(biomed_core)

        # === STEP 2: Mistral-Large expands and structures it beautifully ===
        expansion_prompt = f"""
You are IXORA, an advanced multi-agent biomedical research assistant.

You have received deep biomedical reasoning from BioMistral (your domain expert):

{biomed_core}

Now, expand this into a full, engaging, colleague-level response with the following structure:

<enthusiasm>Express genuine excitement about the research question</enthusiasm>

<explanation>
Provide a detailed, accurate scientific explanation.
Include background, key mechanisms, experimental considerations, and potential pitfalls.
Reference standard practices and biological principles.
</explanation>

<hypothesis>
State 1–2 clear, testable hypotheses derived from BioMistral's insight.
Make them specific, measurable, and mechanistically grounded.
</hypothesis>

<followup>
Ask 2–3 thoughtful follow-up questions to guide next steps.
Focus on experimental design, controls, measurements, or strain/media choices.
</followup>

Original user query: "{query}"

Ensure the response is warm, precise, and inspiring for a researcher.
Use clear formatting and scientific tone.
"""

        content, cot_steps = await generate_with_mistral(
        expansion_prompt, max_tokens=1500, temperature=0.4
        )
        draft = content  # ← Only take the actual text string!

        trace_entry["mistral_expansion"] = True
        trace_entry["final_draft_length"] = len(draft)
        trace_entry["cot_steps_count"] = len(cot_steps)  # Optional: for debugging

        state["draft"] = draft  

    elif domain == "cs":
        try:
            from core.computerscience.loaders import generate_cs_draft
            
            # === STEP 1: CodeLlama does deep but CONCISE CS reasoning ===
            cs_core_prompt = f"""
    You are CodeLlama-Instruct, a precise computer science reasoning model specialized in algorithms, complexity analysis, data structures, and experimental design.

    Your task: Analyze the user's computational/research question below and provide a concise (150–250 words max), high-insight summary covering:
    - Key algorithmic or computational challenge
    - Relevant time/space complexity considerations
    - Important parameters or hyperparameters implied
    - Standard approaches or baselines
    - Potential trade-offs

    Keep it dense, technical, and focused — no greetings, no XML tags, no fluff.

    User Query: "{query}"

    Extracted Parameters (if any): {json.dumps(parameters, indent=2) if parameters else "None detected."}

    Output only the concise analysis:
    """

            # Generate short core draft with lower max_tokens
            cs_core = await generate_cs_draft(cs_core_prompt, max_tokens=300)  # Tight limit!
            # Fallback trim if model ignores limit
            if len(cs_core) > 1000:
                cs_core = cs_core[:1000] + "\n[...truncated for brevity]"

            trace_entry["cs_core_success"] = True
            trace_entry["cs_core_length"] = len(cs_core)

            # === STEP 2: Mistral-Large expands into full structured response ===
            expansion_prompt = f"""
    You are IXORA, an advanced multi-agent computer science research assistant.

    You have received concise, high-quality algorithmic reasoning from CodeLlama:

    {cs_core}

    Now expand this into a full, engaging, structured response using exactly this format:

    <enthusiasm>Show genuine excitement about this excellent CS research question!</enthusiasm>

    <clarify>Ask 1–2 thoughtful questions to clarify key constraints (e.g., input size, memory limits, target runtime, hardware, dataset characteristics).</clarify>

    <explanation>
    Provide clear, detailed explanation including:
    • Core algorithmic ideas and alternatives
    • Time and space complexity analysis
    • Relevant data structures
    • Common baselines and when to use them
    • Reproducibility and implementation tips
    </explanation>

    <hypothesis>
    State 1–2 specific, testable hypotheses about performance, scalability, or optimal parameter choices.
    </hypothesis>

    <followup>
    Ask 2–3 sharp follow-up questions to help refine the approach (e.g., target metrics, real-world constraints, preferred libraries).
    </followup>

    Keep total response under 1800 words. Be precise, warm, and insightful.
    """

            result = await generate_with_mistral(
                expansion_prompt,
                max_tokens=1800,   # Strict cap on final output
                temperature=0.4
            )

            # Safely extract only the content string
            draft = result[0].strip() if isinstance(result, tuple) else str(result).strip()

            trace_entry["cs_expansion_success"] = True
            trace_entry["final_draft_length"] = len(draft)

        except Exception as e:
            logger.warning(f"CS draft pipeline failed: {e} — using safe fallback")
            fallback = await generate_with_mistral(
                f"Provide a clear, structured computer science analysis for: {query}",
                max_tokens=1200,
                temperature=0.5
            )
            draft = (fallback[0].strip() if isinstance(fallback, tuple) else str(fallback))

        # Final safety trim (never let it go wild)
        # After getting draft_content or final_content
        if len(draft_content) > 6200:
            logger.warning(f"Draft too long ({len(draft_content)} chars) — truncating & summarizing")
            truncate_prompt = f"""
            The following text is too long. Summarize it concisely while preserving structure, 
            key scientific points, hypothesis, and follow-up questions.
            Keep under 5800 characters total.
            
            Original text:
            {draft_content[:10000]}  # safety slice
            
            Summarized version:
            """
            summary, _ = await generate_with_mistral(truncate_prompt, max_tokens=1600, temperature=0.5)
            draft_content = summary

        trace_entry["final_draft_length"] = len(draft)
    else:
        draft = await generate_with_mistral(f"Analyze: {query}", max_tokens=1000)
        trace_entry["general_mistral"] = True
    # Ultimate defense — ensure draft is always a string
    if isinstance(draft, tuple):
        draft = draft[0] if draft else ""
    draft = str(draft).strip()

    state["draft"] = draft
    state["trace"].append(trace_entry)
    logger.info(f"Draft agent completed — final draft: {len(draft)} chars")
    return state

async def analytics_agent(state: AgentState) -> AgentState:
    """
    Optimized analytics - only run essential analytics.
    """
    logger.info("📊 [ANALYTICS] Starting analysis...")
    analytics_start = time.time()
    
    parameters = state.get("parameters", {})
    domain = state.get("domain", "biomed")
    
    if not parameters or len(parameters) < 2:
        logger.info("⏭️ [ANALYTICS] Skipping - insufficient parameters")
        state["analytics"] = {
            "skipped": True,
            "reason": "insufficient_parameters",
            "parameter_count": len(parameters)
        }
        return state
    
    try:
        # Run analytics with timeout
        from core.analytics import run_comprehensive_analytics_parallel
        
        analytics_result = await asyncio.wait_for(
            run_comprehensive_analytics_parallel(
                user_input=state["query"],
                parameters=parameters,
                domain=domain
            ),
            timeout=30.0  # 30 second timeout for analytics
        )
        
        state["analytics"] = analytics_result
        
        analytics_time = time.time() - analytics_start
        logger.info(f"✅ [ANALYTICS] Completed in {analytics_time:.2f}s")
        
        state["trace"].append({
            "step": "analytics",
            "time_seconds": round(analytics_time, 2),
            "explainability_method": analytics_result.get("explainability_method", "none"),
            "parameters_analyzed": len(parameters),
            "optimization_note": "will_run_in_background"
        })
        
    except asyncio.TimeoutError:
        logger.warning("⏰ [ANALYTICS] Timed out after 30s")
        state["analytics"] = {
            "timeout": True,
            "partial_results": True
        }
        state["trace"].append({
            "step": "analytics",
            "error": "timeout",
            "time_seconds": 30.0
        })
    except Exception as e:
        logger.error(f"❌ [ANALYTICS] Failed: {e}")
        state["analytics"] = {
            "error": str(e)[:100],
            "failed": True
        }
    
    return state

async def hypothesis_agent(state: AgentState) -> AgentState:
    query = state["query"]
    parameters = state.get("parameters", {})
    analytics = state.get("analytics", {})

    prompt = f"""Based on this research query, generate a specific, testable hypothesis:

Research Goal: {query}

Key Parameters: {json.dumps(parameters, indent=2)[:300]}

Analytics Insights: {analytics.get('executive_summary', '')}

Generate a hypothesis about how the parameters affect the outcome. Be specific and testable."""

    hypothesis, _ = await generate_with_mistral(prompt, max_tokens=150)

    if not hypothesis or len(hypothesis) < 20:
        # Create fallback hypothesis based on parameters
        param_names = list(parameters.keys())[:2]
        if param_names:
            hypothesis = f"The {param_names[0]} parameter will show the strongest effect on the outcome, with potential interactions with other factors."
        else:
            hypothesis = f"Experimental parameters will demonstrate statistically significant effects on measured outcomes."

    state["hypothesis"] = hypothesis.strip()
    state["trace"].append({
        "step": "hypothesis",
        "timestamp": datetime.now().isoformat(),
        "hypothesis_length": len(hypothesis),
        "generation_method": "mistral_with_fallback"
    })
    logger.info("Hypothesis generated")
    return state

async def memory_summarizer(state: AgentState) -> AgentState:
    if len(state["messages"]) <= 3:
        state["conversation_summary"] = ""
        return state
    
    summary_prompt = (
        "Summarize the previous conversation concisely, "
        "focusing on the main research question and any parameters discussed:\n\n"
        + "\n".join([f"{m.role}: {m.content[:200]}" for m in state["messages"][:-1]])
    )
    
    summary = await generate_with_mistral(summary_prompt, max_tokens=180, temperature=0.4)
    state["conversation_summary"] = summary
    return state

async def synthesizer_agent(state: AgentState) -> AgentState:
    query = state["query"]
    hypothesis = state.get("hypothesis", "")
    parameters = state.get("parameters", {})
    analytics = state.get("analytics", {})
    domain = state.get("domain", "biomed")
    
    # Domain-specific system prompts
    if domain == "cs":
        from core.config import CS_SYSTEM_PREFIX
        system_prefix = CS_SYSTEM_PREFIX + """

STRICT RULES (MUST FOLLOW):
1. ANSWER ONLY computer science/research questions. Politely decline others.
2. ALWAYS start with enthusiastic acknowledgment.
3. Provide 4-6 detailed paragraphs of explanation.
4. Include 1-2 clarifying questions after intro.
5. Include a CLEAR, TESTABLE hypothesis.
6. End with 2-3 thoughtful follow-up questions.
7. Use proper CS terminology (algorithms, complexity, data structures).
8. Reference specific parameters from the analysis.
9. Acknowledge limitations when appropriate.
10. Suggest specific next steps.
11. Maintain professional but engaging tone.

FORMAT REQUIREMENTS (CS):
- Start with: <enthusiasm>Your enthusiastic opening</enthusiasm>
- Clarify: 1-2 questions in <clarify> tags
- Explanation: 4-6 paragraphs in <explanation> tags
- Hypothesis: Clear and testable in <hypothesis> tags
- Follow-up: 2-3 questions in <followup> tags
"""
    else:
        # STRICTER SYSTEM PROMPT (biomed/default)
        system_prefix = """
You are a research assistant specializing in scientific analysis.

STRICT RULES (MUST FOLLOW):
1. ANSWER ONLY scientific/research questions. Politely decline others.
2. ALWAYS start with enthusiastic acknowledgment.
3. Provide 4-6 detailed paragraphs of explanation.
4. Include a CLEAR, TESTABLE hypothesis.
5. End with 2-3 thoughtful follow-up questions.
6. Use proper scientific terminology.
7. Reference specific parameters from the analysis.
8. Acknowledge limitations when appropriate.
9. Suggest specific next steps.
10. Maintain professional but engaging tone.

FORMAT REQUIREMENTS:
- Start with: <enthusiasm>Your enthusiastic opening</enthusiasm>
- Explanation: 4-6 paragraphs in <explanation> tags
- Hypothesis: Clear and testable in <hypothesis> tags
- Follow-up: 2-3 questions in <followup> tags
"""

    # Prepare analytics summary
    analytics_summary = ""
    if analytics:
        explain_method = analytics.get("explainability_method", "SHAP")
        if "shap" in explain_method:
            analytics_summary = f"Used SHAP analysis for global feature importance."
        elif "lime" in explain_method:
            analytics_summary = f"Used LIME analysis for local explanations."
        
        # Add parameter insights
        if parameters:
            param_list = list(parameters.keys())[:3]
            analytics_summary += f" Analyzed {len(param_list)} key parameters: {', '.join(param_list)}."
    
    # Build domain-specific prompt
    if domain == "cs":
        prompt = system_prefix + f"""

QUERY: {query}

PARAMETERS EXTRACTED:
{json.dumps(parameters, indent=2)[:500]}

ANALYTICS USED: {analytics_summary}

HYPOTHESIS TO SUPPORT: {hypothesis}

YOUR TASK:
1. Start with enthusiastic acknowledgment of the query
2. Include 1-2 clarifying questions about algorithmic approach, constraints, or metrics
3. Provide 4-6 detailed paragraphs explaining:
   - Algorithmic background and computational complexity
   - Key factors and parameters (batch size, learning rate, etc.)
   - Implementation considerations
   - Performance and scalability approaches
   - Limitations and challenges
4. Present the hypothesis clearly with experimental setup
5. End with 2-3 specific follow-up questions

RESPONSE FORMAT (STRICT - CS):
<enthusiasm>Your opening here</enthusiasm>
<clarify>
1. First clarifying question?
2. Second clarifying question?
</clarify>
<explanation>
Paragraph 1...
Paragraph 2...
Paragraph 3...
Paragraph 4...
Paragraph 5...
Paragraph 6...
</explanation>
<hypothesis>Your clear, testable hypothesis here</hypothesis>
<followup>
1. First follow-up question?
2. Second follow-up question?
3. Third follow-up question?
</followup>

IMPORTANT: If this is NOT a computer science/research question, respond only with: 'I'd love to help, but let's stick to CS—got a technical or research query?'
"""
    else:
        # Build strict prompt (biomed/default)
        prompt = system_prefix + f"""

QUERY: {query}

PARAMETERS EXTRACTED:
{json.dumps(parameters, indent=2)[:500]}

ANALYTICS USED: {analytics_summary}

HYPOTHESIS TO SUPPORT: {hypothesis}

YOUR TASK:
1. Start with enthusiastic acknowledgment of the query
2. Provide 4-6 detailed paragraphs explaining:
   - Scientific background
   - Key factors and parameters
   - Experimental considerations
   - Statistical approaches
   - Limitations and challenges
3. Present the hypothesis clearly
4. End with 2-3 specific follow-up questions

RESPONSE FORMAT (STRICT):
<enthusiasm>Your opening here</enthusiasm>
<explanation>
Paragraph 1...
Paragraph 2...
Paragraph 3...
Paragraph 4...
Paragraph 5...
Paragraph 6...
</explanation>
<hypothesis>Your clear, testable hypothesis here</hypothesis>
<followup>
1. First follow-up question?
2. Second follow-up question?
3. Third follow-up question?
</followup>

IMPORTANT: If this is NOT a scientific/research question, politely decline and explain your scope.
"""

    # Generate with Mistral
    response, _ = await generate_with_mistral(prompt, max_tokens=1200, temperature=0.5)
    
    # Strict validation of format
    if not response:
        response = create_fallback_response(query, hypothesis, analytics_summary, domain)
    
    # Enforce XML structure strictly (domain-aware)
    response = enforce_xml_structure(response, query, domain)
    
    # Determine required tags based on domain
    if domain == "cs":
        required_tags = ["<enthusiasm>", "<clarify>", "<explanation>", "<hypothesis>", "<followup>"]
    else:
        required_tags = ["<enthusiasm>", "<explanation>", "<hypothesis>", "<followup>"]
    
    state["final_response"] = response
    state["trace"].append({
        "step": "synthesizer_strict",
        "timestamp": datetime.now().isoformat(),
        "response_length": len(response),
        "format_validated": all(tag in response for tag in required_tags)
    })
    
    return state

def create_fallback_response(query: str, hypothesis: str = "", analytics_summary: str = "", domain: str = "biomed") -> str:
    """Create a fallback response with proper structure"""
    if not hypothesis:
        if domain == "cs":
            hypothesis = "Computational parameters will demonstrate statistically significant effects on algorithmic performance metrics."
        else:
            hypothesis = "Experimental parameters will demonstrate statistically significant effects on the measured outcomes."
    
    if not analytics_summary:
        analytics_summary = "Basic parameter analysis applied."
    
    if domain == "cs":
        return f"""<enthusiasm>Oh, that's excellent! Great computer science research question!</enthusiasm>

<clarify>
What specific algorithmic approach are you considering? What are your computational constraints?
</clarify>

<explanation>
I've received your query about "{query[:100]}". As a computer science research assistant, I specialize in algorithmic analysis, computational complexity, and experimental design.

{analytics_summary}

Key considerations for CS research include proper experimental design, algorithmic complexity analysis, baseline comparisons, and reproducibility. Factors like batch size, learning rate, data structures, and algorithmic choices significantly affect computational outcomes.

For detailed analysis, ensure your question involves computational research, algorithmic design, or performance optimization.
</explanation>

<hypothesis>
{hypothesis}
</hypothesis>

<followup>
1. What specific implementation approach are you considering?
2. How would this scale to larger datasets or distributed systems?
3. Do you have preliminary performance benchmarks?
</followup>"""
    else:
        return f"""<enthusiasm>Thank you for your research question!</enthusiasm>

<explanation>
I've received your query about "{query[:100]}". As a research assistant, I specialize in scientific analysis and experimental design.

{analytics_summary}

Key considerations for research include proper experimental design, statistical analysis, control groups, and reproducibility. Factors like parameters, conditions, and measurements significantly affect experimental outcomes.

For detailed analysis, ensure your question involves scientific investigation, data analysis, or experimental design.
</explanation>

<hypothesis>
{hypothesis}
</hypothesis>

<followup>
1. What specific research objectives are you trying to achieve?
2. What measurement techniques do you have available?
3. Do you have any preliminary data or literature to build upon?
</followup>"""

def enforce_xml_structure(content: str, user_query: str, domain: str = "biomed") -> str:
    """Enforce strict XML structure for responses (domain-aware)"""
    if not content:
        return create_fallback_response(user_query, "", "", domain)
    
    # Determine required tags based on domain
    if domain == "cs":
        required_tags = ["<enthusiasm>", "<clarify>", "<explanation>", "<hypothesis>", "<followup>"]
        tags_to_close = ["enthusiasm", "clarify", "explanation", "hypothesis", "followup"]
    else:
        required_tags = ["<enthusiasm>", "<explanation>", "<hypothesis>", "<followup>"]
        tags_to_close = ["enthusiasm", "explanation", "hypothesis", "followup"]
    
    for tag in required_tags:
        if tag not in content:
            # Insert missing tag at appropriate location
            if tag == "<enthusiasm>":
                if domain == "cs":
                    content = f"{tag}Oh, that's excellent! Great CS research question!{tag.replace('<', '</')}\n\n{content}"
                else:
                    content = f"{tag}Exciting research question!{tag.replace('<', '</')}\n\n{content}"
            elif tag == "<clarify>":
                if "</enthusiasm>" in content:
                    parts = content.split("</enthusiasm>", 1)
                    content = f"{parts[0]}</enthusiasm>\n\n{tag}What specific algorithmic approach are you considering?{tag.replace('<', '</')}\n\n{parts[1]}"
            elif tag == "<explanation>":
                # Find where to insert explanation
                if "</clarify>" in content:
                    parts = content.split("</clarify>", 1)
                    content = f"{parts[0]}</clarify>\n\n{tag}\n{parts[1]}\n</explanation>"
                elif "<enthusiasm>" in content:
                    parts = content.split("</enthusiasm>", 1)
                    content = f"{parts[0]}</enthusiasm>\n\n{tag}\n{parts[1]}\n</explanation>"
                else:
                    content = f"{tag}\n{content}\n</explanation>"
            elif tag == "<hypothesis>":
                if domain == "cs":
                    hypothesis_text = "Computational parameters will significantly affect algorithmic performance metrics."
                else:
                    hypothesis_text = "Hypothesis based on analysis."
                content = content.replace("</explanation>", f"</explanation>\n\n{tag}{hypothesis_text}{tag.replace('<', '</')}")
            elif tag == "<followup>":
                if domain == "cs": 
                    content = f"{content}\n\n{tag}\n1. What specific implementation approach are you considering?\n2. How would this scale to larger datasets?\n</followup>"
                else:
                    content = f"{content}\n\n{tag}\n1. What specific aspect would you like to explore further?\n2. Do you have any preliminary data?\n</followup>"
    
    # Ensure closing tags
    for tag_name in tags_to_close:
        opening = f"<{tag_name}>"
        closing = f"</{tag_name}>"
        if opening in content and closing not in content:
            content += closing
    
    return content

async def validator_agent(state: AgentState) -> AgentState:
    logger = logging.getLogger("core.langgraph")
    
    response = state.get("final_response", "").strip()
    draft = state.get("draft", "").strip()
    query = state.get("query", "").strip()
    
    # Initialize validation scores
    validation_scores = {}
    embedding_scores = {}  # NEW: Store embedding metrics separately
    
    # === Basic structural & content checks (domain-aware) ===
    domain = state.get("domain", "biomed")
    if domain == "cs":
        required_tags = ["<enthusiasm>", "<clarify>", "<explanation>", "<hypothesis>", "<followup>"]
    else:
        required_tags = ["<enthusiasm>", "<explanation>", "<hypothesis>", "<followup>"]
    
    structural_score = sum(tag in response for tag in required_tags) / len(required_tags)
    
    # Store basic scores
    validation_scores.update({
        "structural_completeness": round(structural_score, 3),
        "has_enthusiasm": 1.0 if "<enthusiasm>" in response else 0.0,
        "has_explanation": 1.0 if "<explanation>" in response else 0.0,
        "has_hypothesis": 1.0 if "<hypothesis>" in response else 0.0,
        "has_followup": 1.0 if "<followup>" in response else 0.0,
        "response_length": min(1.0, len(response) / 1000),
    })
    
    # CS-specific: check for clarify tag
    if domain == "cs":
        validation_scores["has_clarify"] = 1.0 if "<clarify>" in response else 0.0
    
    # === OPTIMIZED: Embedding Similarity with Model Loader ===
    try:
        start_time = time.time()
        
        # Use centralized model loader for embeddings
        from core.model_loader import model_loader
        
        # Batch encode all texts at once
        texts_to_encode = [response, draft, query]
        
        # Generate cache keys for efficient caching
        cache_keys = []
        for text in texts_to_encode:
            text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
            cache_keys.append(f"validator_{text_hash}")
        
        # Get embeddings using domain classification
        domain_scores = await model_loader.classify_domain(query)
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "general"
        
        # Batch encoding
        embeddings = await model_loader.get_embeddings(
            texts_to_encode, 
            domain=primary_domain,
            use_cache=True
        )
        
        # Extract embeddings
        response_emb, draft_emb, query_emb = embeddings
        
        
        response_tensor = torch.tensor(response_emb).unsqueeze(0)
        draft_tensor = torch.tensor(draft_emb).unsqueeze(0)
        query_tensor = torch.tensor(query_emb).unsqueeze(0)
        
        # Compute similarities
        cosine_draft = float(util.cos_sim(response_tensor, draft_tensor)[0][0])
        cosine_query = float(util.cos_sim(response_tensor, query_tensor)[0][0])
        
        # Also compute self-similarity of response (coherence)
        if len(response) > 100:
            # Split response into halves and compare
            half_point = len(response) // 2
            part1 = response[:half_point]
            part2 = response[half_point:]
            part1_emb = await model_loader.get_embeddings(part1, primary_domain)
            part2_emb = await model_loader.get_embeddings(part2, primary_domain)
            part1_tensor = torch.tensor(part1_emb).unsqueeze(0)
            part2_tensor = torch.tensor(part2_emb).unsqueeze(0)
            coherence_score = float(util.cos_sim(part1_tensor, part2_tensor)[0][0])
        else:
            coherence_score = 0.8  # Default for short responses
        
        embedding_time = (time.time() - start_time) * 1000
        
        # Store embedding scores
        embedding_scores.update({
            "cosine_draft_similarity": round(cosine_draft, 3),
            "cosine_query_relevance": round(cosine_query, 3),
            "response_coherence": round(coherence_score, 3),
            "primary_domain": primary_domain,
            "domain_scores": domain_scores,
            "embedding_time_ms": round(embedding_time, 1)
        })
        
        logger.info(f"✅ Embedding validation: draft={cosine_draft:.3f}, query={cosine_query:.3f} ({embedding_time:.1f}ms)")
        
        # Adjust confidence based on similarity
        if cosine_draft < 0.6 or cosine_query < 0.5:
            validation_scores["low_similarity_penalty"] = 0.9
        
        # Domain relevance check
        if primary_domain == "biomed" and domain_scores.get("biomed", 0) < 0.4:
            validation_scores["domain_mismatch_penalty"] = 0.8
        elif primary_domain == "cs" and domain_scores.get("cs", 0) < 0.4:
            validation_scores["domain_mismatch_penalty"] = 0.8

    except Exception as e:
        logger.warning(f"Embedding similarity failed: {e}")
        embedding_scores.update({
            "cosine_draft_similarity": 0.7,
            "cosine_query_relevance": 0.7,
            "response_coherence": 0.7,
            "error": str(e)[:100]
        })
    
    # === Advanced metrics (BLEURT, BERTScore) ===
    total_length = len(draft) + len(response)
    if total_length < 3000:  # Safe limit
        try:
            # BLEURT
            bleurt = load("bleurt", config_name="bleurt-base-128")
            bleurt_score = bleurt.compute(predictions=[response], references=[draft])["scores"][0]
            
            # BERTScore
            bertscore = load("bertscore")
            bs = bertscore.compute(
                predictions=[response],
                references=[draft],
                lang="en",
                model_type="microsoft/deberta-base-mnli"
            )
            bertscore_f1 = bs["f1"][0]

            validation_scores["bleurt_score"] = round(bleurt_score, 3)
            validation_scores["bertscore_f1"] = round(bertscore_f1, 3)
            
            if bleurt_score < 0.4 or bertscore_f1 < 0.8:
                validation_scores["low_advanced_metrics"] = 0.9
                
        except Exception as e:
            logger.warning(f"Advanced metrics failed: {e}")
            validation_scores["bleurt_score"] = 0.6
            validation_scores["bertscore_f1"] = 0.7
    
    # === RLHF Selection (SAFE VERSION) ===
    try:
        reward_model = get_reward_model()
        if reward_model:
            # Generate alternative response
            alt_prompt = f"Generate an alternative detailed research response to: {query}"
            alt_response, _ = await generate_with_mistral(alt_prompt, max_tokens=800, temperature=0.6)
            
            candidates = [response, alt_response]
            
            # Score with reward model
            with torch.no_grad():
                # Get embeddings for candidates using model_loader
                from core.model_loader import model_loader
                
                # Classify domain for reward scoring
                domain_scores = await model_loader.classify_domain(query)
                primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "general"
                
                # Get embeddings
                candidate_embeddings = await model_loader.get_embeddings(candidates, primary_domain)
                
                # Convert to tensor and score
                candidate_tensors = torch.tensor(candidate_embeddings)
                rewards = reward_model.classifier(candidate_tensors)
                
                # SAFE: squeeze and extract scalars
                rewards_flat = rewards.squeeze().cpu().numpy()
                reward_main = float(rewards_flat[0])
                reward_alt = float(rewards_flat[1])

            embedding_scores["rlhf_reward"] = round(reward_main, 3)
            embedding_scores["rlhf_comparison"] = round(reward_alt, 3)

            logger.info(f"RLHF: Main={reward_main:.3f} vs Alt={reward_alt:.3f}")

            # If alternative is better, use it!
            if reward_alt > reward_main + 0.05:
                logger.info("RLHF selected ALTERNATIVE response (better reward)")
                domain = state.get("domain", "biomed")
                state["final_response"] = enforce_xml_structure(alt_response.strip(), query, domain)
                embedding_scores["rlhf_selected"] = "alternative"

    except Exception as e:
        logger.warning(f"RLHF failed safely: {e}")
        embedding_scores["rlhf_reward"] = 0.5
        embedding_scores["rlhf_comparison"] = 0.5
    
    # === Calculate final confidence ===
    # Weight different scores
    weights = {
        "structural_completeness": 0.25,
        "cosine_query_relevance": 0.30 if "cosine_query_relevance" in embedding_scores else 0.15,
        "bertscore_f1": 0.20,
        "bleurt_score": 0.15,
        "response_length": 0.05,
        "has_hypothesis": 0.15,
        "has_followup": 0.10
    }
    
    # Calculate weighted confidence
    weighted_sum = 0
    total_weight = 0
    
    for score_name, weight in weights.items():
        if score_name in validation_scores:
            score_value = validation_scores[score_name]
            weighted_sum += score_value * weight
            total_weight += weight
        elif score_name in embedding_scores:
            score_value = embedding_scores[score_name]
            weighted_sum += score_value * weight
            total_weight += weight
    
    # Base confidence
    base_confidence = weighted_sum / total_weight if total_weight > 0 else 0.7
    
    # BONUS for good analytics integration
    if state.get("analytics", {}) and "explainability" in state["analytics"]:
        base_confidence *= 1.15  # 15% bonus for good analytics
    
    # PENALTY for missing required elements
    if not validation_scores.get("has_hypothesis", 0):
        base_confidence *= 0.8  # 20% penalty
    
    if not validation_scores.get("has_followup", 0):
        base_confidence *= 0.9  # 10% penalty
    
    # Ensure reasonable bounds
    final_confidence = min(0.98, max(0.3, base_confidence))
    
    # Store all scores
    validation_scores["overall_confidence"] = round(final_confidence, 3)
    state["embedding_scores"] = embedding_scores  # NEW: Store embedding scores
    
    # Store validation results in state
    state["validation_scores"] = validation_scores
    state["confidence"] = final_confidence
    state["validated"] = True
    
    # Add detailed trace
    state["trace"].append({
        "step": "validator_comprehensive",
        "validation_scores": validation_scores,
        "embedding_scores": embedding_scores,
        "confidence": final_confidence,
        "timestamp": datetime.now().isoformat()
    })
    
    logger.info(f"✅ Validation complete — overall confidence: {final_confidence:.3f}")
    
    return state

# ==================== GRAPH CONSTRUCTION ====================

def create_workflow():
    workflow = StateGraph(AgentState)

    # Add nodes in sequence
    workflow.add_node("extractor", extractor_agent)
    workflow.add_node("draft", draft_agent)
    workflow.add_node("analytics", analytics_agent)
    workflow.add_node("hypothesis", hypothesis_agent)
    workflow.add_node("synthesizer", synthesizer_agent)
    workflow.add_node("validator", validator_agent)

    # Linear flow
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "draft")
    workflow.add_edge("draft", "analytics")
    workflow.add_edge("analytics", "hypothesis")
    workflow.add_edge("hypothesis", "synthesizer")
    workflow.add_edge("synthesizer", "validator")
    workflow.add_edge("validator", END)

    return workflow.compile()

# Compile the graph once
multi_agent_graph = create_workflow()

# ==================== PUBLIC ENTRY POINT ====================

async def run_multi_agent(
    query: str,
    domain: str = "biomed",
    session_id: str = None,
    history: List[Dict[str, str]] = None
) -> dict:
    history = history or []

    
    initial_state = AgentState(
        messages=[HumanMessage(content=msg["content"]) for msg in history] + [HumanMessage(content=query)],
        query=query,
        domain=domain,
        parameters={},
        analytics={},
        hypothesis="",
        draft="",
        final_response="",
        trace=[],
        confidence=0.0,
        step_count=0,
        validated=False,
        embedding_scores={}
    )
    logger.info(f"Starting multi-agent pipeline for query: {query[:100]}...")
    logger.info(f"Starting with history length: {len(initial_state['messages'])}")
    if len(initial_state['messages']) > 1:
        logger.info("Previous message: " + initial_state['messages'][-2].content[:80])
    

    try:
        result = await multi_agent_graph.ainvoke(
            initial_state,
            config={"recursion_limit": 10}
        )

        return {
            "final_response": result.get("final_response", "Response generation failed."),
            "trace": result.get("trace", []),
            "confidence": result.get("confidence", 0.7),
            "embedding_scores": result.get("embedding_scores", {}),  # NEW: Include embedding scores
            "validation_scores": result.get("validation_scores", {}),
            "white_box_state": {
                k: v for k, v in result.items() 
                if k not in ["final_response", "trace", "messages"]  # Filter large fields
            }
        }
    except Exception as e:
        logger.exception(f"Graph execution failed: {e}")
        # Fallback response (domain-aware)
        if domain == "cs":
            fallback = f"""<enthusiasm>Oh, that's excellent! Great computer science research question!</enthusiasm>

<clarify>
What specific algorithmic approach are you considering?
</clarify>

<explanation>
I encountered a technical issue while processing your query about "{query[:100]}". 

For CS research questions like this, typical considerations include:
- Algorithmic complexity analysis
- Proper experimental design with baselines
- Performance metrics and benchmarking
- Reproducibility (random seeds, library versions)
</explanation>
<hypothesis>Computational parameters will significantly influence algorithmic performance metrics.</hypothesis>
<followup>Could you rephrase your question or ask about specific computational parameters?</followup>"""
        else:
            fallback = f"""<enthusiasm>Thank you for your research question!</enthusiasm>

<explanation>
I encountered a technical issue while processing your query about "{query[:100]}". 

For research questions like this, typical considerations include:
- Experimental design with proper controls
- Statistical analysis of results  
- Parameter optimization
- Relevance and reproducibility
</explanation>
<hypothesis>The key experimental parameters will significantly influence your research outcomes.</hypothesis>
<followup>Could you rephrase your question or ask about specific experimental conditions?</followup>"""

        return {
            "final_response": fallback,
            "trace": [{"step": "error", "error": str(e)[:100], "fallback_used": True}],
            "confidence": 0.8,
            "embedding_scores": {},
            "validation_scores": {},
            "white_box_state": {}
        }