"""
core/config.py - Cleaned & Optimized Configuration (Feb 2026)
Supports biomedical and computer science domains with Mistral API fallback
No more static intent keywords — using LLM-based router instead
"""

import os
import torch
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()

# ========== CPU OPTIMIZATION ==========
FORCE_CPU = os.getenv("FORCE_CPU", "true").lower() == "true"
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)

EXTRACT_USE_LLM = os.getenv("EXTRACT_USE_LLM", "true").lower() == "true"

# ========== MISTRAL API (PRIMARY for expansion / final response) ==========
MISTRAL_USE_API = os.getenv("MISTRAL_USE_API", "true").lower() == "true"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_DEVICE = "cpu"
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-large-latest")
MISTRAL_MAX_TOKENS = int(os.getenv("MISTRAL_MAX_TOKENS", "4096"))
MISTRAL_TEMPERATURE = float(os.getenv("MISTRAL_TEMPERATURE", "0.7"))
MISTRAL_TIMEOUT = float(os.getenv("MISTRAL_TIMEOUT", "60.0"))

# ========== BIOMEDICAL DOMAIN — Use BioGPT (Transformers) ==========
BIOMISTRAL_USE_TRANSFORMERS = True
BIOMISTRAL_TRANSFORMERS_MODEL = "microsoft/biogpt"
BIOMISTRAL_ENABLED = True
BIOMISTRAL_MAX_TOKENS = int(os.getenv("BIOMISTRAL_MAX_TOKENS", "512"))
BIOMISTRAL_TIMEOUT = float(os.getenv("BIOMISTRAL_TIMEOUT", "60.0"))
BIOMISTRAL_DEVICE = "cpu"
BIOMISTRAL_CTX_LENGTH = 2048
BIOMISTRAL_N_THREADS = int(os.getenv("BIOMISTRAL_N_THREADS", "2"))
BIOMISTRAL_N_GPU_LAYERS = int(os.getenv("BIOMISTRAL_N_GPU_LAYERS", "0"))

# ========== COMPUTER SCIENCE DOMAIN — Using Qwen-Coder (GGUF) ==========
CSMODEL_USE_FROM_PRETRAINED = True
CSMODEL_REPO_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
CSMODEL_FILENAME = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
CSMODEL_ENABLED = True
CSMODEL_MAX_TOKENS = int(os.getenv("CSMODEL_MAX_TOKENS", "512"))
CSMODEL_TIMEOUT = float(os.getenv("CSMODEL_TIMEOUT", "60.0"))
CSMODEL_CTX_LENGTH = 8192
CSMODEL_N_THREADS = int(os.getenv("CSMODEL_N_THREADS", "2"))
CSMODEL_N_GPU_LAYERS = int(os.getenv("CSMODEL_N_GPU_LAYERS", "0"))

# ========== QWEN (parameter extraction & intent routing) ==========
QWEN_USE_FROM_PRETRAINED = True
QWEN_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
QWEN_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
QWEN_ENABLED = True
QWEN_CTX_LENGTH = 8192
QWEN_N_THREADS = int(os.getenv("QWEN_N_THREADS", "3"))
QWEN_N_GPU_LAYERS = int(os.getenv("QWEN_N_GPU_LAYERS", "0"))
QWEN_MAX_TOKENS = 600
QWEN_TIMEOUT = 45.0

# ========== LOCAL MODEL FILES (GGUF) — fallbacks ==========
BIOMISTRAL_GGUF_FILE = os.getenv("BIOMISTRAL_GGUF_FILE", "")
CSMODEL_GGUF_FILE = os.getenv("CSMODEL_GGUF_FILE", "")
QWEN_GGUF_FILE = os.getenv("QWEN_GGUF_FILE", "")

# ========== MEDGEMMA CONFIG (Disabled) ==========
MEDGEMMA_USE_FROM_PRETRAINED = False
MEDGEMMA_REPO_ID = ""
MEDGEMMA_FILENAME = ""
MEDGEMMA_CTX_LENGTH = 2048
MEDGEMMA_N_THREADS = 2
MEDGEMMA_N_GPU_LAYERS = 0

# ========== EMBEDDING MODEL ==========
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_MAX_LENGTH = 512
EMBEDDING_CACHE_SIZE = 1000

# ========== FEATURE FLAGS ==========
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "false").lower() == "true"
ENABLE_LIME = os.getenv("ENABLE_LIME", "false").lower() == "true"
ENABLE_BIOMEDLM = os.getenv("ENABLE_BIOMEDLM", "false").lower() == "true"
ENABLE_ANALYTICS = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"
ENABLE_REWARD_MODEL = os.getenv("ENABLE_REWARD_MODEL", "false").lower() == "true"

# ========== CPU-OPTIMIZED ANALYTICS SETTINGS ==========
ANALYTICS_SETTINGS = {
    "max_samples": 100,
    "n_estimators": 20,
    "bootstrap_iterations": 20,
    "optimization_iterations": 10,
    "timeout_per_analytic": 15.0,
    "use_simplified_models": True
}

# ========== CELERY SETTINGS ==========
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
CELERY_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKER_CONCURRENCY", "1"))
CELERY_TASK_TIME_LIMIT = 300

# ========== REDIS CACHE ==========
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = 3600

# ========== SERVER SETTINGS ==========
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ========== DOMAIN PORT MAPPING ==========
DOMAIN_PORTS = {
    "biomed": int(os.getenv("BIOMED_PORT", "8000")),
    "cs": int(os.getenv("CS_PORT", "8001")),
    "general": int(os.getenv("GENERAL_PORT", "8002"))
}

# ========== DOMAIN-SPECIFIC SYSTEM PREFIXES ==========
SYSTEM_PREFIX = """
1. Do not answer questions unrelated to your field; politely redirect.
2. Start enthusiastically, acknowledge interest.
3. Be natural, engaging — ask clarifying questions.
4. Be explicit about techniques (causal inference, SHAP/LIME, etc.).
5. EXPLANATION RULE: Detailed, 4–6 paragraphs minimum for complex topics.
6. Actionable: Suggest next steps, end with 1–2 thoughtful questions.
7. Identify clear, testable hypothesis when appropriate.
8. Conversational, colleague-like tone.
9. Guide step-by-step, seek clarity before deep suggestions.
"""

BIOMED_SYSTEM_PREFIX = """
You are a biomedical research assistant. Focus on:
- Biology, biochemistry, pharmacology, cell/molecular biology
- Experimental parameters: pH, temperature, concentration, dosage
- Wet-lab protocols, clinical relevance, biomolecular mechanisms
""" + SYSTEM_PREFIX

CS_SYSTEM_PREFIX = """
You are a computer science research assistant. Focus on:
- Algorithms, complexity, data structures, ML theory
- Systems, performance, optimization, benchmarking
- Computational techniques, software/hardware trade-offs
""" + SYSTEM_PREFIX

# ========== HELPER FUNCTIONS ==========
def get_domain_config(domain: str) -> Dict[str, Any]:
    base = {
        "system_prefix": SYSTEM_PREFIX,
        "use_api": MISTRAL_USE_API,
        "api_key_configured": bool(MISTRAL_API_KEY),
        "port": PORT,
        "domain": domain
    }
    if domain == "biomed":
        return {
            **base,
            "system_prefix": BIOMED_SYSTEM_PREFIX,
            "model": "biomistral" if BIOMISTRAL_ENABLED and not MISTRAL_USE_API else "mistral_api",
            "max_tokens": BIOMISTRAL_MAX_TOKENS,
            "timeout": BIOMISTRAL_TIMEOUT,
            "port": DOMAIN_PORTS.get("biomed", PORT)
        }
    elif domain == "cs":
        return {
            **base,
            "system_prefix": CS_SYSTEM_PREFIX,
            "model": "cs_model" if CSMODEL_ENABLED and not MISTRAL_USE_API else "mistral_api",
            "max_tokens": CSMODEL_MAX_TOKENS,
            "timeout": CSMODEL_TIMEOUT,
            "port": DOMAIN_PORTS.get("cs", PORT)
        }
    else:
        return {
            **base,
            "model": "mistral_api",
            "max_tokens": MISTRAL_MAX_TOKENS,
            "timeout": MISTRAL_TIMEOUT,
            "port": DOMAIN_PORTS.get("general", PORT)
        }


def get_model_for_domain(domain: str) -> str:
    if MISTRAL_USE_API and MISTRAL_API_KEY:
        return "mistral_api"
    if domain == "biomed" and BIOMISTRAL_ENABLED:
        return "biomistral"
    if domain == "cs" and CSMODEL_ENABLED:
        return "cs_model"
    return "qwen"


def is_api_available() -> bool:
    return MISTRAL_USE_API and bool(MISTRAL_API_KEY)


def get_available_models() -> Dict[str, bool]:
    return {
        "mistral_api": is_api_available(),
        "biomistral": BIOMISTRAL_ENABLED,
        "cs_model": CSMODEL_ENABLED,
        "qwen": QWEN_ENABLED,
        "embedding": True
    }


def validate_config():
    import logging
    logger = logging.getLogger(__name__)

    if MISTRAL_USE_API and not MISTRAL_API_KEY:
        logger.warning("⚠️  Mistral API enabled but no key configured")

    if not MISTRAL_USE_API:
        enabled = []
        if BIOMISTRAL_ENABLED: enabled.append("BioMistral")
        if CSMODEL_ENABLED:    enabled.append("CS Model")
        if QWEN_ENABLED:       enabled.append("Qwen")
        if not enabled:
            logger.error("❌ No local models enabled and Mistral API disabled!")
        else:
            logger.info(f"✅ Local models: {', '.join(enabled)}")

    return True


# Validate on import
validate_config()