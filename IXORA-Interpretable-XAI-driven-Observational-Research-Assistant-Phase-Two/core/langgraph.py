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
from core.config import BIOMISTRAL_TIMEOUT, CODEQWEN_TIMEOUT
from core.cs_parameter_extractor import extract_cs_parameters
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
    # Initialize state with domain-specific settings
    state = {
        "messages": [],
        "query": "",
        "domain": "",
        "parameters": {},
        "analytics": {},
        "hypothesis": "",
        "draft": "",
        "final_response": "",
        "trace": [{"step": "init", "domain": "", "timestamp": datetime.now().isoformat()}],
        "confidence": 1.0,
        "step_count": 0,
        "validated": False,
        "embedding_scores": {},
        "domain_specific": {
            "biomed": {
                "timeout": BIOMISTRAL_TIMEOUT,
                "max_tokens": 800,
                "temperature": 0.7
            },
            "computerscience": {
                "timeout": CODEQWEN_TIMEOUT,
                "max_tokens": 1024,
                "temperature": 0.3
            }
        }.get("", {
            "timeout": 60.0,
            "max_tokens": 512,
            "temperature": 0.5
        })
    }
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
    domain = state.get("domain", "biomed").lower()
    parameters = {}

    if domain in ["biomed", "medical"]:
        # Biomedical parameter extraction
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
    
    elif domain in ["cs", "computerscience", "ai"]:
        # Computer Science parameter extraction
        try:
            cs_params = await extract_cs_parameters(query)
            if cs_params and 'parameters' in cs_params:
                for param_name, param_data in cs_params['parameters'].items():
                    if isinstance(param_data, dict) and 'value' in param_data:
                        parameters[param_name] = {
                            "value": param_data['value'],
                            "unit": param_data.get('unit', ''),
                            "description": param_data.get('description', ''),
                            "method": "cs_extractor",
                            "confidence": param_data.get('confidence', 0.8)
                        }
        except Exception as e:
            logger.warning(f"CS parameter extraction failed: {e}")
            # Fallback to basic regex patterns
            lr_match = re.search(r'(?:learning[-\s]rate|lr)[=:]?\s*([\d\.e-]+)', query, re.IGNORECASE)
            if lr_match:
                parameters["learning_rate"] = {
                    "value": float(lr_match.group(1)),
                    "unit": "",
                    "description": "Learning rate for optimization",
                    "method": "regex"
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
        
    elif domain in ["cs", "computerscience", "ai"]:
        try:
            # Use CodeQwen for CS domain
            from core.model_loader import model_loader
            
            cs_prompt = f"""You are an expert computer science research assistant. Analyze this query and provide a detailed response:
            
            Query: {query}
            
            Include:
            1. Technical explanation
            2. Relevant algorithms/approaches
            3. Implementation considerations
            4. Potential challenges and solutions
            5. 1-2 follow-up questions
            
            Format with XML tags: <technical>...</technical> <algorithms>...</algorithms> <implementation>...</implementation> <challenges>...</challenges> <followup>...</followup>"""
            
            draft, _ = await model_loader.generate_with_codeqwen(
                cs_prompt,
                max_tokens=600,
                temperature=0.3
            )
            trace_entry["codeqwen_used"] = True
            trace_entry["codeqwen_draft_length"] = len(draft)
            
        except Exception as e:
            logger.warning(f"CodeQwen generation failed: {e} - using fallback")
            draft = f"Analysis for computer science query: {query}"
            trace_entry["codeqwen_used"] = False
            trace_entry["fallback_reason"] = str(e)[:100]

    else:
        # For non-biomedical domains
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
    
    # STRICTER SYSTEM PROMPT
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
    
    # Build strict prompt
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
        response = create_fallback_response(query, hypothesis, analytics_summary)
    
    # Enforce XML structure strictly
    response = enforce_xml_structure(response, query)
    
    state["final_response"] = response
    state["trace"].append({
        "step": "synthesizer_strict",
        "timestamp": datetime.now().isoformat(),
        "response_length": len(response),
        "format_validated": all(tag in response for tag in ["<enthusiasm>", "<explanation>", "<hypothesis>", "<followup>"])
    })
    
    return state

def create_fallback_response(query: str, hypothesis: str = "", analytics_summary: str = "") -> str:
    """Create a fallback response with proper structure"""
    if not hypothesis:
        hypothesis = "Experimental parameters will demonstrate statistically significant effects on the measured outcomes."
    
    if not analytics_summary:
        analytics_summary = "Basic parameter analysis applied."
    
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

def enforce_xml_structure(content: str, query: str) -> str:
    """Enforce strict XML structure for responses"""
    if not content:
        return create_fallback_response(query)
    
    # Ensure all required tags are present
    required_tags = ["<enthusiasm>", "<explanation>", "<hypothesis>", "<followup>"]
    
    for tag in required_tags:
        if tag not in content:
            # Insert missing tag at appropriate location
            if tag == "<enthusiasm>":
                content = f"{tag}Exciting research question!{tag.replace('<', '</')}\n\n{content}"
            elif tag == "<explanation>":
                # Find where to insert explanation
                if "<enthusiasm>" in content:
                    parts = content.split("</enthusiasm>", 1)
                    content = f"{parts[0]}</enthusiasm>\n\n{tag}\n{parts[1]}\n</explanation>"
                else:
                    content = f"{tag}\n{content}\n</explanation>"
            elif tag == "<hypothesis>":
                content = content.replace("</explanation>", f"</explanation>\n\n{tag}Hypothesis based on analysis.</hypothesis>")
            elif tag == "<followup>":
                content = f"{content}\n\n{tag}\n1. What specific aspect would you like to explore further?\n2. Do you have any preliminary data?\n</followup>"
    
    # Ensure closing tags
    tags_to_close = ["enthusiasm", "explanation", "hypothesis", "followup"]
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
    
    # === Basic structural & content checks ===
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
                state["final_response"] = enforce_xml_structure(alt_response.strip(), query)
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
):
    """Run the multi-agent workflow with domain-specific optimizations.
    
    Args:
        query: User query or input text
        domain: Domain for processing (biomed, computerscience, etc.)
        session_id: Optional session ID for tracking
        history: Conversation history
        
    Returns:
        Dict containing the final response and metadata
    """
    # Normalize domain
    domain = domain.lower()
    if domain in ["cs", "ai", "ml"]:
        domain = "computerscience"
    elif domain in ["bio", "medical"]:
        domain = "biomed"
    
    history = history or []
    
    # Import the appropriate pipeline based on domain
    if domain == "computerscience":
        from core.computerscience.pipeline import ComputerSciencePipeline
        pipeline = ComputerSciencePipeline()
        logger.info(f"Using Computer Science pipeline for domain: {domain}")
    else:  # Default to biomedical
        from core.medicalscience.pipeline import BiomedicalPipeline
        pipeline = BiomedicalPipeline()
        logger.info(f"Using Biomedical pipeline for domain: {domain}")
    
    try:
        # Run the appropriate pipeline
        logger.info(f"Starting {domain} pipeline for query: {query[:100]}...")
        result = await pipeline.run(query, session_id)
        
        # Format the response
        response = {
            "final_response": result.response,
            "trace": result.trace,
            "confidence": result.confidence,
            "embedding_scores": {},
            "validation_scores": {
                "overall_confidence": result.confidence,
                "domain": domain
            },
            "white_box_state": {
                "domain": domain,
                "parameters": result.parameters,
                "analytics": result.analytics,
                "hypothesis": result.hypothesis,
                "evidence": [e.to_dict() for e in result.evidence] if hasattr(result, 'evidence') else []
            }
        }
        
        return response
    except Exception as e:
        logger.exception(f"{domain.capitalize()} pipeline execution failed: {e}")
        
        # Domain-specific fallback responses
        if domain == "computerscience":
            fallback = f"""<response>
<error>I encountered an issue processing your computer science question.</error>
<explanation>
I couldn't process your query about "{query[:100]}" due to a technical issue.

For computer science questions, consider:
- The specific programming language or framework
- Algorithm complexity and efficiency
- System design considerations
- Security implications if applicable
</explanation>
<followup>Could you rephrase your question or provide more specific details?</followup>
</response>"""
        else:  # Default to biomedical fallback
            fallback = f"""<response>
<error>I encountered an issue processing your research question.</error>
<explanation>
I couldn't process your query about "{query[:100]}" due to a technical issue.

For research questions, consider:
- Experimental design with controls
- Statistical analysis methods
- Parameter optimization
- Relevance and reproducibility
</explanation>
<followup>Could you rephrase your question or provide more specific details?</followup>
</response>"""

        return {
            "final_response": fallback,
            "trace": [{"step": "error", "error": str(e)[:100], "fallback_used": True}],
            "confidence": 0.5,  # Lower confidence for fallback responses
            "embedding_scores": {},
            "validation_scores": {"overall_confidence": 0.5},
            "white_box_state": {
                "domain": domain,
                "error": str(e)[:500]  # Include error in white box state
            }
        }