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
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from core.analytics import run_bayesian_optimization, run_comprehensive_analytics_parallel
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

async def extractor_agent(state: AgentState) -> AgentState:
    query = state["query"]
    parameters = {}

    # Extract pH range
    ph_match = re.search(r'pH\s*([\d\.]+)\s*[–\-]\s*([\d\.]+)', query)
    if ph_match:
        ph_low, ph_high = float(ph_match.group(1)), float(ph_match.group(2))
        parameters["ph_range"] = {
            "value": [ph_low, ph_high],
            "unit": "pH",
            "description": f"pH range from {ph_low} to {ph_high}",
            "method": "regex"
        }
        parameters["ph_center"] = {
            "value": (ph_low + ph_high) / 2,
            "unit": "pH",
            "description": "Midpoint of pH range",
            "method": "calculated"
        }

    # Extract temperature range
    temp_match = re.search(r'(\d+\.?\d*)\s*[–\-]\s*(\d+\.?\d*)\s*°?C', query)
    if temp_match:
        temp_low, temp_high = float(temp_match.group(1)), float(temp_match.group(2))
        parameters["temperature_range"] = {
            "value": [temp_low, temp_high],
            "unit": "°C",
            "description": f"Temperature range from {temp_low} to {temp_high}",
            "method": "regex"
        }
        parameters["temp_center"] = {
            "value": (temp_low + temp_high) / 2,
            "unit": "°C",
            "description": "Midpoint of temperature range",
            "method": "calculated"
        }

    state["parameters"] = parameters
    state["trace"].append({"step": "extractor", "parameters_count": len(parameters)})
    logger.info(f"Extractor found {len(parameters)} parameters")
    return state

async def draft_agent(state: AgentState) -> AgentState:
    query = state["query"]
    domain = state["domain"]
    draft = ""
    trace_entry = {"step": "draft"}

    if domain == "biomed":
        try:
            # Use model_loader for BioMistral generation
            from core.model_loader import model_loader
            
            biomed_draft = await model_loader.generate_with_biomistral(
                f"Summarize biomedical context for: {query}",
                max_tokens=80
            )
            
            trace_entry["biomistral_used"] = True
            trace_entry["biomistral_draft_length"] = len(biomed_draft)
        except Exception as e:
            logger.warning(f"BioMistral generation failed: {e} - using fallback")
            biomed_draft = ""
            trace_entry["biomistral_used"] = False
            trace_entry["fallback_reason"] = str(e)[:100]

        refine_prompt = f"""
You are a biomedical research assistant. Take this concise biomedical draft and expand it into a detailed, engaging, colleague-level response.
Include scientific accuracy, experimental design considerations, a clear testable hypothesis, and 1-2 thoughtful follow-up questions.

Biomedical draft:
{biomed_draft if biomed_draft else "Fallback: Standard biomedical factors for query."}

Original query:
{query}

Respond in XML structure: <enthusiasm>...</enthusiasm> <explanation>...</explanation> <hypothesis>...</hypothesis> <followup>...</followup>
"""

        draft, _ = await generate_with_mistral(
            refine_prompt,
            max_tokens=400,
            temperature=0.7
        )
        trace_entry["mistral_expansion"] = True

    elif domain == "cs":
        try:
            # Use CS model loader for CS draft generation
            from core.computerscience.loaders import generate_cs_draft
            
            cs_draft = await generate_cs_draft(
                query,
                max_tokens=80
            )
            
            trace_entry["csmodel_used"] = True
            trace_entry["csmodel_draft_length"] = len(cs_draft)
        except Exception as e:
            logger.warning(f"CS Model generation failed: {e} - using fallback")
            cs_draft = ""
            trace_entry["csmodel_used"] = False
            trace_entry["fallback_reason"] = str(e)[:100]

        refine_prompt = f"""
You are a computer science research assistant. Take this concise CS draft and expand it into a detailed, engaging, colleague-level response.
Include algorithmic reasoning, computational complexity considerations, a clear testable hypothesis, and 1-2 thoughtful follow-up questions.

CS draft:
{cs_draft if cs_draft else "Fallback: Standard computational factors for query."}

Original query:
{query}

Respond in XML structure: <enthusiasm>...</enthusiasm> <clarify>...</clarify> <explanation>...</explanation> <hypothesis>...</hypothesis> <followup>...</followup>
"""

        draft, _ = await generate_with_mistral(
            refine_prompt,
            max_tokens=400,
            temperature=0.7
        )
        trace_entry["mistral_expansion"] = True

    else:
        # For other domains
        mistral_prompt = f"Generate a detailed analysis for: {query}"
        draft, _ = await generate_with_mistral(mistral_prompt, max_tokens=300)
        trace_entry["mistral_only"] = True

    state["draft"] = draft
    state["trace"].append(trace_entry)
    logger.info(f"Draft generated ({len(draft)} chars)")
    return state

async def analytics_agent(state: AgentState) -> AgentState:
    logger.info("Running analytics agent...")
    
    parameters = state.get("parameters", {})
    domain = state.get("domain", "biomed")
    
    # Run the real analytics (this is where LIME/SHAP selection happens)
    analytics_result = await run_comprehensive_analytics_parallel(
        user_input=state["query"],
        parameters=parameters,
        domain=domain
    )
    
    # Store the real result
    state["analytics"] = analytics_result
    
    # Now we can safely access the dynamic fields
    state["trace"].append({
        "step": "analytics",
        "explainability_method": analytics_result.get("explainability_method", "none"),
        "parameters_count": len(parameters),
        "selection_reason": "embedding similarity",
        "lime_used": "lime" in analytics_result.get("explainability", {}),
        "shap_used": "shap" in analytics_result.get("explainability", {}),
        "execution_time_ms": analytics_result.get("execution_time", 0) * 1000  # optional, if added
    })
    
    logger.info(f"Analytics completed. Method: {analytics_result.get('explainability_method', 'none')}")

    # Optional: conditional Bayesian optimization
    if "optimize" in state.get("intent", "").lower() or len(parameters) >= 3:
        try:
            optimization_result = await run_bayesian_optimization(parameters, domain)
            # Merge or update optimization
            state["analytics"]["optimization"] = optimization_result
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            state["analytics"]["optimization"] = {"error": str(e)}

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
    response, _ = await generate_with_mistral(prompt, max_tokens=1800, temperature=0.7)
    
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
        
        # Convert to tensors for similarity computation
        import torch
        from sentence_transformers import util
        
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
            alt_response, _ = await generate_with_mistral(alt_prompt, max_tokens=800, temperature=0.8)
            
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