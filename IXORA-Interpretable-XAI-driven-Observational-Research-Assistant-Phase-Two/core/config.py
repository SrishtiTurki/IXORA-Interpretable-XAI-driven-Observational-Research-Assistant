# core/config.py - ADD CELERY CONFIG & BIOMISTRAL GGUF
import os
import torch
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv() 

# ========== CPU OPTIMIZATION WITH FEATURES ==========
FORCE_CPU = os.getenv("FORCE_CPU", "true").lower() == "true"
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)  # Limit CPU threads
RLHF_ENABLED = os.getenv("RLHF_ENABLED", "true").lower() == "true"
RLHF_MODEL_PATH = os.getenv("RLHF_MODEL_PATH", "models/rlhf")
RLHF_REWARD_MODEL = os.getenv("RLHF_REWARD_MODEL", "OpenAssistant/reward-model-deberta-v3-large-v2")
RLHF_LEARNING_RATE = float(os.getenv("RLHF_LEARNING_RATE", "1.41e-5"))
RLHF_BATCH_SIZE = int(os.getenv("RLHF_BATCH_SIZE", "4"))
RLHF_DOMAIN_SPECIFIC = {
    "biomed": {
        "learning_rate": 5e-6,
        "batch_size": 2,
        "max_length": 1024,
        "reward_model": "OpenAssistant/reward-model-deberta-v3-large-v2"
    },
    "computerscience": {
        "learning_rate": 2e-5,
        "batch_size": 4,
        "max_length": 2048,
        "reward_model": "bigcode/starcoder2-rlhf-reward-model"
    }
}  

EXTRACT_USE_LLM = os.getenv("EXTRACT_USE_LLM", "true").lower() == "true"

# BioMistral GGUF Settings (mandatory for biomed domain)
BIOMISTRAL_GGUF_MODEL = os.getenv("BIOMISTRAL_GGUF_MODEL", "MaziyarPanahi/BioMistral-7B-GGUF")
BIOMISTRAL_GGUF_FILE = os.getenv("BIOMISTRAL_GGUF_FILE", "BioMistral-7B.Q5_K_M.gguf")  # Best balance
BIOMISTRAL_MAX_TOKENS = int(os.getenv("BIOMISTRAL_MAX_TOKENS", "80"))
BIOMISTRAL_TIMEOUT = float(os.getenv("BIOMISTRAL_TIMEOUT", "60.0"))
BIOMISTRAL_CTX_LENGTH = 4096
BIOMISTRAL_N_THREADS = int(os.getenv("BIOMISTRAL_N_THREADS", "4"))  # Adjust to your CPU cores

# CS Domain Configuration
CS_DOMAINS = {
    "ALGORITHMS": "algorithms",
    "COMPLEXITY": "complexity_theory",
    "COMPUTABILITY": "computability",
    "CRYPTOGRAPHY": "cryptography",
    "ML_THEORY": "ml_theory",
    "QUANTUM": "quantum",
    "FORMAL_LANG": "formal_languages",
    "SYSTEMS": "systems",
    "SOFTWARE_ENG": "software_engineering",
    "PRAC_CODING": "practical_coding"
}

# CodeQwen Settings (for computerscience domain)
CODEQWEN_MODEL = os.getenv("CODEQWEN_MODEL", "Qwen/CodeQwen1.5-7B-Chat")
CODEQWEN_MAX_TOKENS = int(os.getenv("CODEQWEN_MAX_TOKENS", "512"))
CODEQWEN_TEMPERATURE = float(os.getenv("CODEQWEN_TEMPERATURE", "0.2"))
CODEQWEN_TOP_P = float(os.getenv("CODEQWEN_TOP_P", "0.95"))
CODEQWEN_TIMEOUT = float(os.getenv("CODEQWEN_TIMEOUT", "120.0"))
CODEQWEN_CTX_LENGTH = 32768  # Larger context for code understanding
CODEQWEN_DEVICE = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
CODEQWEN_LOAD_IN_8BIT = os.getenv("CODEQWEN_LOAD_IN_8BIT", "true").lower() == "true"
CODEQWEN_LOAD_IN_4BIT = os.getenv("CODEQWEN_LOAD_IN_4BIT", "false").lower() == "true"
CODEQWEN_TRUST_REMOTE_CODE = True  # Required for CodeQwen
BIOMISTRAL_N_GPU_LAYERS = int(os.getenv("BIOMISTRAL_N_GPU_LAYERS", "0"))  # 0 = CPU only; -1 = all on GPU

# Mistral - Use API for speed, fallback to local if needed
MISTRAL_USE_API = os.getenv("MISTRAL_USE_API", "true").lower() == "true"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_DEVICE = "cpu"

# Mistral-Large Config (full, no quant)
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistralai/Mistral-Large-Instruct-2407")  
MISTRAL_USE_API = os.getenv("MISTRAL_USE_API", "true").lower() == "true"  # TRUE = API mode
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # Required for API mode
MISTRAL_DEVICE = os.getenv("MISTRAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Feature flags - enable but with limitations
ENABLE_SHAP = True
ENABLE_LIME = True
ENABLE_BIOMEDLM = True
ENABLE_ANALYTICS = True

# CPU-optimized analytics settings
ANALYTICS_SETTINGS = {
    "max_samples": 100,  # Reduced from 1000
    "n_estimators": 20,  # Reduced from 100
    "bootstrap_iterations": 20,  # Reduced from 100
    "optimization_iterations": 10,  # Reduced from 50
    "timeout_per_analytic": 15.0,  # 15 seconds max
    "use_simplified_models": True
}

# Celery Configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
CELERY_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKER_CONCURRENCY", "2"))
CELERY_TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", "300"))  # 5 minutes
CELERY_CS_TASK_TIME_LIMIT = int(os.getenv("CELERY_CS_TASK_TIME_LIMIT", "600"))  # 10 minutes for CS tasks

# Domain Models
DOMAIN_MODELS = {
    "biomed": "biomedical",
    "general": "general",
    "computerscience": "computerscience"
}

# Redis Cache
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = 3600  # 1 hour

# Intents
INTENT_KEYWORDS = {
    "conversational": [
        # basic explanations & understanding
        "what is", "what are", "tell me about", "explain", "explain to me",
        "define", "definition of", "who is", "who are",
        "how does", "how do", "how does it work", "how it works",
        "describe", "give an overview", "overview of",
        "basics of", "introduction to", "intro to",
        "simple explanation", "in simple terms", "layman terms",
        "meaning of", "purpose of", "use of", "uses of",
        "example of", "real world example", "illustrate",
        "difference between", "what does it mean",
        "walk me through", "break it down", "step by step explanation"
    ],

    "research": [
        # academic & scientific inquiry
        "study", "studies", "experiment", "experiments", "determine"
        "investigate", "investigation", "research", "research on",
        "explore", "exploration", "work with",
        "paper on", "research paper", "thesis", "dissertation",
        "findings", "results", "data collection", "dataset",
        "case study", "case studies",
        "observations", "empirical", "scientific",
        "methodology", "methods", "experimental design",
        "publication", "published", "journal",
        "peer review", "peer reviewed",
        "analyze results", "statistical analysis",
        "survey", "questionnaire",
        "conducted by", "authors", "literature review",
        "prior work", "related work", "hypothesis",
        "research gap", "evidence based", "replication study"
    ],

    "analyze": [
        # reasoning, evaluation, causality
        "analyze", "analysis of", "examine", "evaluation",
        "evaluate", "assess", "assessment",
        "compare", "comparison", "contrast",
        "interpret", "interpretation",
        "derive", "derive insights",
        "optimize", "optimization",
        "determine", "estimate", "predict", "forecast",
        "correlation", "relationship", "association",
        "causal", "causality", "cause and effect",
        "why", "why does", "why is",
        "root cause", "explain why",
        "impact of", "effect of", "influence of",
        "reason behind", "underlying reason",
        "how does it affect", "implications of",
        "strengths and weaknesses",
        "trade offs", "limitations",
        "pattern", "trend", "anomaly",
        "what happens if", "scenario analysis"
    ]
}


# Shared System Prefix (override per domain in prompts.py)
SYSTEM_PREFIX = """
1. Do not answer questions that are not related to your field; tell them politely that you only answer domain-related questions.
2. Always start by enthusiastically acknowledging the user's interest.
3. Make the conversation natural and engaging—ask open-ended clarifying questions.
4. Be explicit about techniques you'll use (e.g., causal inference, SHAP/LIME explanations).
5. Explain the user query as clearly as you can, making sure it is lengthy and detailed.
6. Actionable & engaging: Suggest next steps, then end with 1-2 thoughtful follow-up questions.
7. Identify a clear, testable hypothesis for each problem or question.
8. Full depth: Write 4-6 detailed paragraphs explaining concepts thoroughly, with scientific accuracy.
9. Be conversational and dynamic: Respond like a colleague.
10. Guide the user step by step, getting clarity before suggesting ideas.
"""