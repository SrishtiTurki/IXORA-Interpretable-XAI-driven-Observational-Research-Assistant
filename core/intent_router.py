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
    
    # Systems
    "operating system", "kernel", "thread", "process", "memory management",
    "cache", "compiler", "interpreter", "database", "query optimization",
    
    # Security/crypto
    "encryption", "decryption", "hash function", "cryptography", "malware",
    "vulnerability", "exploit", "security analysis", "attack surface"
}

# ========== FALLBACK CLASSIFICATION (KEYWORD-BASED) ==========

def classify_by_keywords(query: str, forced_domain: Optional[str] = None) -> Tuple[str, float]:
    """
    Fallback classification using keyword matching.
    Returns: (domain, confidence)
    """
    query_lower = query.lower()
    
    # If domain is forced, check for strong violations
    if forced_domain == "biomed":
        cs_score = sum(1 for kw in CS_STRONG_KEYWORDS if kw in query_lower)
        biomed_score = sum(1 for kw in BIOMED_STRONG_KEYWORDS if kw in query_lower)
        
        # If CS keywords dominate, it's out-of-domain
        if cs_score >= 3 and cs_score > biomed_score * 2:
            return "out_of_domain_cs", 0.9
        
        # Check for specific CS patterns
        if any(pattern in query_lower for pattern in [
            "adversarial", "malware", "algorithm", "complexity", 
            "neural network training", "model robustness"
        ]):
            return "out_of_domain_cs", 0.85
    
    elif forced_domain == "cs":
        biomed_score = sum(1 for kw in BIOMED_STRONG_KEYWORDS if kw in query_lower)
        cs_score = sum(1 for kw in CS_STRONG_KEYWORDS if kw in query_lower)
        
        # If biomed keywords dominate, it's out-of-domain
        if biomed_score >= 3 and biomed_score > cs_score * 2:
            return "out_of_domain_biomed", 0.9
        
        # Check for specific biomed patterns
        if any(pattern in query_lower for pattern in [
            "ph", "cell culture", "enzyme", "protein expression",
            "clinical trial", "drug dosage"
        ]):
            return "out_of_domain_biomed", 0.85
    
    # No forced domain or within domain - classify normally
    biomed_score = sum(1 for kw in BIOMED_STRONG_KEYWORDS if kw in query_lower)
    cs_score = sum(1 for kw in CS_STRONG_KEYWORDS if kw in query_lower)
    
    if biomed_score > cs_score and biomed_score >= 2:
        return "biomed", 0.7
    elif cs_score > biomed_score and cs_score >= 2:
        return "cs", 0.7
    else:
        return "casual_chat", 0.5


# ========== LLM-BASED CLASSIFICATION ==========

def build_classification_prompt(query: str, forced_domain: Optional[str] = None) -> str:
    """Build classification prompt for Qwen"""
    
    base_prompt = f"""Classify the following query into ONE category. Return ONLY valid JSON.

Query: "{query}"

Categories:
1. "research_planning" - User wants to design an experiment or research study
2. "parameter_extraction" - Query mentions specific experimental parameters (pH, temp, concentration, etc.)
3. "explanation" - User wants to understand a concept or mechanism
4. "casual_chat" - General conversation, greetings, or unclear intent
"""
    
    if forced_domain == "biomed":
        base_prompt += """5. "out_of_domain_cs" - Query is about computer science (algorithms, ML training, malware, complexity theory, etc.)

IMPORTANT: If the query is about:
- Machine learning algorithms or training
- Adversarial models or malware detection
- Algorithm complexity or data structures
- Computer systems or programming
Then classify as "out_of_domain_cs"
"""
    elif forced_domain == "cs":
        base_prompt += """5. "out_of_domain_biomed" - Query is about biology/medicine (cell culture, pH, enzymes, clinical trials, etc.)

IMPORTANT: If the query is about:
- Biological experiments or wet-lab work
- Cell cultures, enzymes, or proteins
- Drug dosages or clinical trials
- pH, temperature, or other lab conditions
Then classify as "out_of_domain_biomed"
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
            
            # Determine if pipeline is needed
            needs_pipeline = intent in ["research_planning", "parameter_extraction"]
            
            # Map intent to task
            task_map = {
                "research_planning": "experimental_design",
                "parameter_extraction": "parameter_analysis",
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
        domain, confidence = classify_by_keywords(query, forced_domain)
        
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
        
        # Map domain to intent
        intent_map = {
            "biomed": "research_planning",
            "cs": "research_planning",
            "casual_chat": "casual_chat"
        }
        
        intent = intent_map.get(domain, "casual_chat")
        needs_pipeline = intent != "casual_chat"
        
        return {
            "intent": intent,
            "confidence": confidence,
            "needs_pipeline": needs_pipeline,
            "task": "experimental_design" if needs_pipeline else None,
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
    
    try:
        result = await classify_with_qwen(query, forced_domain)
        
        logger.info(
            f"🤖 Classification: {result['intent']} "
            f"(conf={result['confidence']:.2f}, method={result['method']})"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Classification failed completely: {e}")
        
        # Ultimate fallback
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
        return """This question appears to be about computer science, algorithms, or machine learning theory. 

I am specialized in **biomedical experimental research** (biology, biochemistry, pharmacology, wet-lab protocols, clinical research) and cannot provide reliable answers on CS topics.

Please either:
1. Rephrase your question in a biomedical context, or
2. Use a computer science-focused system for this query.

I'd be happy to help with any biology or experimental design questions!"""
    
    elif intent == "out_of_domain_biomed":
        return """This question appears to be about biological experiments, wet-lab protocols, or medical research.

I am specialized in **computer science theory** (algorithms, complexity, machine learning theory, systems) and cannot provide reliable answers on biological experiments or life sciences.

Please either:
1. Rephrase your question in a computational/algorithmic context, or
2. Use a biomedical-focused system for this query.

I'd be happy to help with any CS theory or algorithmic questions!"""
    
    else:
        return "I cannot assist with this query as it falls outside my area of specialization."