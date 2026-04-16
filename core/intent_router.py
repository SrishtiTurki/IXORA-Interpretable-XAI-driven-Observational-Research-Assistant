# core/intent_router.py - FIXED for CS domain (preserves biomedical functionality)

import json
import logging
import asyncio
import re
from typing import Dict, Any, Optional, Tuple
from core.model_loader import generate_with_qwen

logger = logging.getLogger("core.intent_router")

# ========== DOMAIN KEYWORDS ==========
# These are STRONG indicators of domain, not weak hints

BIOMED_STRONG_KEYWORDS = {
    # Experimental biology
    "ph", "temperature", "concentration", "dosage", "cell culture", "enzyme", 
    "protein", "dna", "rna", "gene", "mutation", "bacteria", "virus", 
    "antibiotic", "drug", "pharmacology", "clinical trial", "patient",
    
    # Biochemistry
    "biochemical", "metabolic", "pathway", "receptor", "ligand", "substrate",
    "inhibitor", "catalyst", "reaction rate", "buffer", "molarity",
    
    # Lab techniques
    "pcr", "elisa", "western blot", "gel electrophoresis", "chromatography",
    "microscopy", "culture media", "incubation", "centrifuge", "pipette",
    
    # Biological systems
    "cell line", "tissue", "organ", "in vivo", "in vitro", "animal model",
    "knockout", "transgenic", "stem cell", "differentiation", "apoptosis"
}

CS_STRONG_KEYWORDS = {
    # Algorithms & theory
    "algorithm", "complexity", "time complexity", "space complexity", "big o",
    "dynamic programming", "greedy", "divide and conquer", "recursion",
    "sorting", "searching", "graph algorithm", "tree traversal",
    
    # Data structures
    "linked list", "hash table", "binary tree", "heap", "stack", "queue",
    "graph", "array", "matrix", "data structure",
    
    # ML/AI theory
    "neural network", "training", "backpropagation", "gradient descent",
    "overfitting", "regularization", "cross-validation", "loss function",
    "optimizer", "adversarial training", "malware detection", "model robustness",
    "random forest", "xgboost", "ensemble", "bagging", "boosting", "stacking",
    "precision", "recall", "f1", "f1-score", "accuracy", "roc", "auc",
    "hyperparameter", "fine-tuning", "inference",
    
    # Systems
    "operating system", "kernel", "thread", "process", "memory management",
    "cache", "compiler", "interpreter", "database", "query optimization",
    
    # Security/crypto
    "encryption", "decryption", "hash function", "cryptography", "malware",
    "vulnerability", "exploit", "security analysis", "attack surface",
    
    # Data/ML operations
    "dataset", "resampling", "smote", "undersampling", "oversampling",
    "feature selection", "dimensionality reduction", "pca"
}

# ========== RESEARCH QUERY DETECTION (NEW) ==========
# These patterns indicate a research query that needs full pipeline
RESEARCH_QUERY_PATTERNS = [
    # Performance improvement patterns
    r"improved \w+ from [\d\.]+ to [\d\.]+",
    r"increased \w+ from \d+ to \d+",
    r"boosted \w+ by [\d\.]+%",
    r"reduced \w+ from [\d\.]+ to [\d\.]+",
    r"decreased \w+ by [\d\.]+%",
    r"[\d\.]+% (improvement|increase|gain|reduction|decrease)",
    
    # Comparison patterns
    r"compare[s]? .+ (with|to|against)",
    r"benchmark",
    r"evaluat(e|ing|ion)",
    
    # Parameter/optimization patterns
    r"parameter[s]? (optimization|tuning|selection)",
    r"hyperparameter",
    r"cross-?validation",
    
    # Experimental patterns
    r"experiment",
    r"analysis of",
    r"effect of .+ on",
]

CASUAL_CHAT_PATTERNS = [
    r"^(hi|hello|hey|greetings)[\s!]*$",
    r"^(good morning|good afternoon|good evening)[\s!]*$",
    r"^(thanks|thank you|ty)[\s!]*$",
    r"^how are you[\s?]*$",
    r"^what('s| is) up[\s?]*$",
    r"^bye|goodbye|see you[\s!]*$",
]

# ========== FALLBACK CLASSIFICATION (KEYWORD-BASED) ==========

def classify_by_keywords(query: str, forced_domain: Optional[str] = None) -> Tuple[str, float, bool]:
    """
    Fallback classification using keyword matching.
    Returns: (domain, confidence, is_research)
    """
    query_lower = query.lower()
    
    # FIRST: Check if this is a research query (needs full pipeline)
    is_research = False
    research_score = 0
    
    # Check research patterns
    for pattern in RESEARCH_QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            research_score += 25
            logger.debug(f"Research pattern matched: {pattern}")
    
    # Check for CS research keywords
    cs_research_keywords = [
        "random forest", "xgboost", "ensemble", "bagging", "boosting",
        "precision", "recall", "f1", "accuracy", "roc", "auc",
        "hyperparameter", "cross-validation", "training", "inference",
        "dataset", "resampling", "feature", "model stability"
    ]
    for kw in cs_research_keywords:
        if kw in query_lower:
            research_score += 10
    
    # Check for numbers + metrics (strong indicator of research)
    if re.search(r"\d+\.?\d*%", query_lower):
        research_score += 15
    if re.search(r"from \d+ to \d+", query_lower):
        research_score += 20
    
    # Check query length (longer queries are more likely research)
    word_count = len(query.split())
    if word_count > 12:
        research_score += 10
    if word_count > 20:
        research_score += 10
    
    is_research = research_score >= 25
    
    # Check for casual chat (overrides research if clear)
    for pattern in CASUAL_CHAT_PATTERNS:
        if re.match(pattern, query_lower, re.IGNORECASE):
            return "casual_chat", 0.9, False
    
    # Short greetings
    if word_count <= 3 and query_lower in ["hi", "hello", "hey", "thanks", "bye"]:
        return "casual_chat", 0.95, False
    
    # If domain is forced, check for strong violations
    if forced_domain == "biomed":
        cs_score = sum(2 for kw in CS_STRONG_KEYWORDS if kw in query_lower)
        biomed_score = sum(2 for kw in BIOMED_STRONG_KEYWORDS if kw in query_lower)
        
        # Also check for research indicators in CS context
        if is_research and cs_score >= 2:
            # This is a CS research query - not out of domain, but needs CS pipeline
            return "cs", 0.85, True
        
        # If CS keywords dominate, it's out-of-domain
        if cs_score >= 6 and cs_score > biomed_score * 2:
            return "out_of_domain_cs", 0.9, False
        
        # Check for specific CS patterns
        cs_patterns = ["adversarial", "malware", "algorithm", "complexity", 
                       "neural network training", "model robustness", "random forest",
                       "ensemble", "precision", "recall", "f1"]
        if any(pattern in query_lower for pattern in cs_patterns):
            if is_research:
                return "cs", 0.85, True
            return "out_of_domain_cs", 0.85, False
    
    elif forced_domain == "cs":
        biomed_score = sum(2 for kw in BIOMED_STRONG_KEYWORDS if kw in query_lower)
        cs_score = sum(2 for kw in CS_STRONG_KEYWORDS if kw in query_lower)
        
        # If biomed keywords dominate, it's out-of-domain
        if biomed_score >= 6 and biomed_score > cs_score * 2:
            return "out_of_domain_biomed", 0.9, False
        
        # Check for specific biomed patterns
        biomed_patterns = ["ph", "cell culture", "enzyme", "protein expression",
                           "clinical trial", "drug dosage"]
        if any(pattern in query_lower for pattern in biomed_patterns):
            return "out_of_domain_biomed", 0.85, False
    
    # No forced domain or within domain - classify normally
    biomed_score = sum(1 for kw in BIOMED_STRONG_KEYWORDS if kw in query_lower)
    cs_score = sum(1 for kw in CS_STRONG_KEYWORDS if kw in query_lower)
    
    # Research query takes precedence
    if is_research:
        if cs_score >= 2:
            return "cs", 0.8, True
        elif biomed_score >= 2:
            return "biomed", 0.8, True
        else:
            # Research query but unclear domain - use forced or default to cs
            return forced_domain if forced_domain in ["cs", "biomed"] else "cs", 0.7, True
    
    # Non-research classification
    if biomed_score > cs_score and biomed_score >= 2:
        return "biomed", 0.7, False
    elif cs_score > biomed_score and cs_score >= 2:
        return "cs", 0.7, False
    else:
        return "casual_chat", 0.5, False


# ========== LLM-BASED CLASSIFICATION ==========

def build_classification_prompt(query: str, forced_domain: Optional[str] = None) -> str:
    """Build classification prompt for Qwen"""
    
    base_prompt = f"""Classify the following query into ONE category. Return ONLY valid JSON.

Query: "{query}"

Categories:
1. "research_query" - User asks about experimental results, parameter optimization, performance metrics, or wants to design a study/experiment
2. "explanation" - User wants to understand a concept, mechanism, or how something works
3. "casual_chat" - General conversation, greetings, thanks, or unclear intent
"""
    
    if forced_domain == "biomed":
        base_prompt += """4. "out_of_domain_cs" - Query is about computer science (algorithms, ML training, malware, complexity theory, random forests, precision/recall, F1 scores, etc.)

IMPORTANT RULES:
- If the query mentions performance metrics (accuracy, precision, recall, F1, ROC, AUC) → "research_query"
- If the query mentions model names (Random Forest, XGBoost, Neural Network, Ensemble) → "research_query"  
- If the query mentions changes in numbers (improved from X to Y, increased by Z%) → "research_query"
- If query is about algorithms, data structures, complexity, systems → "out_of_domain_cs"
"""
    elif forced_domain == "cs":
        base_prompt += """4. "out_of_domain_biomed" - Query is about biology/medicine (cell culture, pH, enzymes, clinical trials, drug dosages, etc.)

IMPORTANT RULES:
- If the query mentions CS metrics (accuracy, precision, recall, F1, runtime, complexity) → "research_query"
- If the query mentions algorithms, models, training, inference → "research_query"
- If the query mentions parameter changes or improvements → "research_query"
- If query is about biological experiments, wet-lab protocols, clinical research → "out_of_domain_biomed"
"""
    
    base_prompt += """
Return ONLY this JSON format (no other text):
{
  "intent": "<category>",
  "confidence": 0.85,
  "reasoning": "Brief explanation"
}"""
    
    return base_prompt


async def classify_with_qwen(query: str, forced_domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Use Qwen to classify intent with domain boundary detection.
    Returns dict with: intent, confidence, needs_pipeline, task
    """
    
    try:
        prompt = build_classification_prompt(query, forced_domain)
        
        # Generate with Qwen
        response = await asyncio.wait_for(
            generate_with_qwen(prompt, max_tokens=200, temperature=0.3),
            timeout=8.0
        )
        
        logger.debug(f"Qwen raw response: {response[:200]}")
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            
            intent = result.get("intent", "casual_chat")
            confidence = float(result.get("confidence", 0.5))
            
            # Determine if pipeline is needed - FIXED: research_query needs pipeline
            needs_pipeline = intent in ["research_query", "explanation"]
            
            # For explanation, sometimes we still want pipeline for depth
            if intent == "explanation" and len(query.split()) > 10:
                needs_pipeline = True
            
            # Map intent to task
            task_map = {
                "research_query": "full_pipeline",
                "explanation": "explanation",
                "casual_chat": None,
                "out_of_domain_cs": None,
                "out_of_domain_biomed": None
            }
            
            return {
                "intent": intent,
                "confidence": confidence,
                "needs_pipeline": needs_pipeline,
                "task": task_map.get(intent),
                "reasoning": result.get("reasoning", ""),
                "method": "qwen_llm"
            }
        else:
            logger.warning("No JSON found in Qwen response, using keyword fallback")
            raise ValueError("No JSON in response")
    
    except Exception as e:
        logger.warning(f"Qwen classification failed: {e} → using keyword fallback")
        
        # Fallback to keyword-based classification
        domain, confidence, is_research = classify_by_keywords(query, forced_domain)
        
        # Handle out-of-domain cases
        if domain.startswith("out_of_domain"):
            return {
                "intent": domain,
                "confidence": confidence,
                "needs_pipeline": False,
                "task": None,
                "reasoning": "Detected by keyword analysis",
                "method": "keyword_fallback"
            }
        
        # Handle research queries
        if is_research or domain in ["cs", "biomed"]:
            return {
                "intent": "research_query",
                "confidence": max(confidence, 0.7),
                "needs_pipeline": True,
                "task": "full_pipeline",
                "reasoning": f"Research query detected in {domain} domain",
                "method": "keyword_fallback"
            }
        
        # Map domain to intent for non-research
        intent_map = {
            "biomed": "explanation",
            "cs": "explanation",
            "casual_chat": "casual_chat"
        }
        
        intent = intent_map.get(domain, "casual_chat")
        needs_pipeline = intent != "casual_chat"
        
        return {
            "intent": intent,
            "confidence": confidence,
            "needs_pipeline": needs_pipeline,
            "task": "full_pipeline" if needs_pipeline else None,
            "reasoning": "Keyword-based classification",
            "method": "keyword_fallback"
        }


# ========== MAIN CLASSIFICATION FUNCTION ==========

async def classify_conversation_intent(
    query: str,
    session_state: Optional[Dict[str, Any]] = None,
    forced_domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main classification function with domain boundary enforcement.
    
    Returns:
        {
            "intent": str,
            "confidence": float,
            "needs_pipeline": bool,
            "task": str | None,
            "reasoning": str,
            "method": str
        }
    """
    
    logger.info(f"Classifying query (domain={forced_domain}): {query[:100]}...")
    
    # Quick pre-check for obvious research queries (bypasses LLM for speed)
    query_lower = query.lower()
    
    # Check for CS research patterns (fast path)
    cs_research_indicators = [
        "random forest", "xgboost", "ensemble", "precision", "recall", "f1",
        "accuracy", "improved from", "increased from", "boosted by",
        "model stability", "hyperparameter", "cross-validation"
    ]
    
    for indicator in cs_research_indicators:
        if indicator in query_lower:
            logger.info(f"🔬 Fast-path CS research detected: '{indicator}'")
            return {
                "intent": "research_query",
                "confidence": 0.85,
                "needs_pipeline": True,
                "task": "full_pipeline",
                "reasoning": f"Fast-path: CS research indicator '{indicator}'",
                "method": "fast_path"
            }
    
    # Check for biomed research patterns (fast path)
    biomed_research_indicators = [
        "cell culture", "enzyme activity", "protein expression", "ph",
        "temperature effect", "dosage", "clinical trial", "in vivo", "in vitro"
    ]
    
    for indicator in biomed_research_indicators:
        if indicator in query_lower:
            logger.info(f"🔬 Fast-path biomed research detected: '{indicator}'")
            return {
                "intent": "research_query",
                "confidence": 0.85,
                "needs_pipeline": True,
                "task": "full_pipeline",
                "reasoning": f"Fast-path: Biomed research indicator '{indicator}'",
                "method": "fast_path"
            }
    
    # Check for performance number patterns
    if re.search(r"improved .+ from [\d\.]+ to [\d\.]+", query_lower):
        logger.info("🔬 Fast-path: Performance improvement pattern detected")
        return {
            "intent": "research_query",
            "confidence": 0.90,
            "needs_pipeline": True,
            "task": "full_pipeline",
            "reasoning": "Fast-path: Performance improvement pattern",
            "method": "fast_path"
        }
    
    # Proceed with LLM classification for ambiguous cases
    try:
        result = await classify_with_qwen(query, forced_domain)
        
        # Override: If query has research characteristics but LLM said casual
        if result.get("intent") == "casual_chat" and not result.get("needs_pipeline"):
            # Double-check with keyword analysis
            _, _, is_research = classify_by_keywords(query, forced_domain)
            if is_research:
                logger.warning("LLM misclassified research query as casual - overriding")
                result = {
                    "intent": "research_query",
                    "confidence": 0.75,
                    "needs_pipeline": True,
                    "task": "full_pipeline",
                    "reasoning": "Overriding LLM misclassification",
                    "method": "keyword_override"
                }
        
        logger.info(
            f"🤖 Classification: {result['intent']} "
            f"(conf={result['confidence']:.2f}, method={result['method']}, pipeline={result['needs_pipeline']})"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Classification failed completely: {e}")
        
        # Ultimate fallback - treat as research if it has any substance
        if len(query.split()) > 5:
            return {
                "intent": "research_query",
                "confidence": 0.6,
                "needs_pipeline": True,
                "task": "full_pipeline",
                "reasoning": "Fallback: treating as research due to length",
                "method": "error_fallback"
            }
        
        return {
            "intent": "casual_chat",
            "confidence": 0.3,
            "needs_pipeline": False,
            "task": None,
            "reasoning": "Classification system failure",
            "method": "error_fallback"
        }


# ========== HELPER: CHECK IF OUT OF DOMAIN ==========

def is_out_of_domain(intent: str) -> bool:
    """Check if intent indicates out-of-domain query"""
    return intent.startswith("out_of_domain_")


def get_out_of_domain_message(intent: str, query: str) -> str:
    """Generate appropriate out-of-domain refusal message"""
    
    if intent == "out_of_domain_cs":
        return """I appreciate your question, but I am specifically designed for **computer science research**. 

Your query appears to be about biomedical or biological experiments. For questions about:
- Cell cultures, enzymes, proteins
- pH, temperature, drug dosages
- Clinical trials or wet-lab protocols

Please use a biomedical-focused system.

However, if you have a CS research question (algorithms, ML models, performance optimization, system design), I'd be happy to help!"""
    
    elif intent == "out_of_domain_biomed":
        return """I appreciate your question, but I am specifically designed for **biomedical research**. 

Your query appears to be about computer science. For questions about:
- Algorithms and complexity
- Machine learning models
- Performance optimization
- System design

Please use a CS-focused system.

However, if you have a biomedical research question (experimental design, parameter optimization, biological mechanisms), I'd be happy to help!"""
    
    else:
        return "I'm specialized in research questions. Please ask me about computer science or biomedical research topics."