# core/config.py - COMPLETE CONFIG WITH BIOMED + CS DOMAINS
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

EXTRACT_USE_LLM = os.getenv("EXTRACT_USE_LLM", "true").lower() == "true"

# ========== BIOMEDICAL DOMAIN (UNCHANGED) ==========
# BioMistral GGUF Settings (mandatory for biomed domain)
BIOMISTRAL_GGUF_MODEL = os.getenv("BIOMISTRAL_GGUF_MODEL", "MaziyarPanahi/BioMistral-7B-GGUF")
BIOMISTRAL_GGUF_FILE = os.getenv("BIOMISTRAL_GGUF_FILE", "BioMistral-7B.Q5_K_M.gguf")
BIOMISTRAL_MAX_TOKENS = int(os.getenv("BIOMISTRAL_MAX_TOKENS", "80"))
BIOMISTRAL_TIMEOUT = float(os.getenv("BIOMISTRAL_TIMEOUT", "60.0"))
BIOMISTRAL_CTX_LENGTH = 4096
BIOMISTRAL_N_THREADS = int(os.getenv("BIOMISTRAL_N_THREADS", "4"))
BIOMISTRAL_N_GPU_LAYERS = int(os.getenv("BIOMISTRAL_N_GPU_LAYERS", "0"))  # 0 = CPU only


# ========== COMPUTER SCIENCE DOMAIN (NEW) ==========
# CS Model GGUF Settings - Using  CodeLlama
CSMODEL_GGUF_MODEL = os.getenv("CSMODEL_GGUF_MODEL", "TheBloke/CodeLlama-7B-Instruct-GGUF")
CSMODEL_GGUF_FILE = os.getenv("CSMODEL_GGUF_FILE", "codellama-7b-instruct.Q5_K_M.gguf")

CSMODEL_MAX_TOKENS = int(os.getenv("CSMODEL_MAX_TOKENS", "80"))
CSMODEL_TIMEOUT = float(os.getenv("CSMODEL_TIMEOUT", "60.0"))
CSMODEL_CTX_LENGTH = 4096
CSMODEL_N_THREADS = int(os.getenv("CSMODEL_N_THREADS", "4"))
CSMODEL_N_GPU_LAYERS = int(os.getenv("CSMODEL_N_GPU_LAYERS", "0"))  # 0 = CPU only

# ========== MISTRAL API (SHARED BY BOTH DOMAINS) ==========
# Mistral - Use API for speed, fallback to local if needed
MISTRAL_USE_API = os.getenv("MISTRAL_USE_API", "true").lower() == "true"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_DEVICE = "cpu"

# Mistral-Large Config (full, no quant)
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistralai/Mistral-Large-Instruct-2407")  
MISTRAL_USE_API = os.getenv("MISTRAL_USE_API", "true").lower() == "true"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_DEVICE = os.getenv("MISTRAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ========== FEATURE FLAGS (SHARED) ==========
ENABLE_SHAP = True
ENABLE_LIME = True
ENABLE_BIOMEDLM = True  # For biomed domain
ENABLE_ANALYTICS = True

# ========== CPU-OPTIMIZED ANALYTICS SETTINGS (SHARED) ==========
ANALYTICS_SETTINGS = {
    "max_samples": 100,
    "n_estimators": 20,
    "bootstrap_iterations": 20,
    "optimization_iterations": 10,
    "timeout_per_analytic": 15.0,
    "use_simplified_models": True
}

# ========== CELERY SETTINGS (SHARED) ==========
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
CELERY_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKER_CONCURRENCY", "1"))
CELERY_TASK_TIME_LIMIT = 300  # 5 minutes max per task

# ========== DOMAIN MODELS (UPDATED) ==========
DOMAIN_MODELS = {
    "biomed": "biomedical",
    "cs": "computer_science",
    "general": "general"
}

# ========== REDIS CACHE (SHARED) ==========
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = 3600  # 1 hour

# ========== INTENT KEYWORDS (UPDATED WITH CS) ==========
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
        # academic & scientific inquiry (both biomed and CS)
        "study", "studies", "experiment", "experiments", "determine",
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
        "research gap", "evidence based", "replication study",
        # CS-specific research keywords
        "benchmark", "ablation study", "baseline comparison",
        "implementation", "proof of concept", "validation study"
    ],

    "analyze": [
        # reasoning, evaluation, causality (both domains)
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
        "trade offs", "tradeoffs", "limitations",
        "pattern", "trend", "anomaly",
        "what happens if", "scenario analysis",
        # CS-specific analysis keywords
        "complexity analysis", "performance analysis",
        "scalability", "bottleneck", "profiling",
        "time complexity", "space complexity",
        "big o", "algorithm efficiency"
    ]
}

# ========== DOMAIN-SPECIFIC SYSTEM PREFIXES ==========
# Shared System Prefix (general guidelines)
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

# Biomed-specific override
BIOMED_SYSTEM_PREFIX = """
You are a biomedical research assistant. Focus on:
- Medical science, biology, chemistry, pharmacology
- Enzymes, proteins, cellular processes
- pH, temperature, concentration, dosage parameters
- Experimental design in biological systems
- Clinical trials, drug development, biomolecular interactions

""" + SYSTEM_PREFIX

# CS-specific override (NEW)
CS_SYSTEM_PREFIX = """
You are a computer science research assistant. Focus on:
- Algorithms, data structures, computational complexity
- System design, architecture, performance optimization
- Machine learning, AI, deep learning techniques
- Software engineering, programming paradigms
- Distributed systems, databases, networking
- Time/space complexity analysis, benchmarking

""" + SYSTEM_PREFIX

# ========== DOMAIN PORT MAPPING (NEW) ==========
# For running multiple domain servers on different ports
DOMAIN_PORTS = {
    "biomed": int(os.getenv("BIOMED_PORT", "8000")),
    "cs": int(os.getenv("CS_PORT", "8001")),
    "general": int(os.getenv("GENERAL_PORT", "8002"))
}

# ========== CS-SPECIFIC PARAMETER PATTERNS (NEW) ==========
CS_PARAMETER_PATTERNS = {
    "time_complexity": [
        (r'O\(([^)]+)\)', 'complexity', 'big-O'),
        (r'time complexity[:\s]+O\(([^)]+)\)', 'complexity', 'big-O'),
        (r'running time[:\s]+O\(([^)]+)\)', 'complexity', 'big-O')
    ],
    "space_complexity": [
        (r'space[:\s]+O\(([^)]+)\)', 'complexity', 'big-O'),
        (r'memory[:\s]+O\(([^)]+)\)', 'complexity', 'big-O')
    ],
    "dataset_size": [
        (r'(\d+\.?\d*)\s*(MB|GB|TB|KB)', 'size', None),
        (r'(\d+\.?\d*)\s*(million|billion|thousand)\s*(records|rows|samples)', 'size', None)
    ],
    "performance": [
        (r'(\d+\.?\d*)\s*(ms|milliseconds|seconds|fps|qps)', 'performance', None),
        (r'latency[:\s]+(\d+\.?\d*)\s*(ms|seconds)', 'latency', 'ms'),
        (r'throughput[:\s]+(\d+\.?\d*)\s*(qps|rps|ops)', 'throughput', None)
    ],
    "accuracy": [
        (r'(\d+\.?\d*)%?\s*(accuracy|precision|recall|f1)', 'metric', '%'),
        (r'(accuracy|precision|recall|f1)[:\s]+(\d+\.?\d*)%?', 'metric', '%')
    ],
    "batch_size": [
        (r'batch size[:\s]+(\d+)', 'batch', 'samples'),
        (r'(\d+)\s*samples per batch', 'batch', 'samples')
    ],
    "iterations": [
        (r'(\d+)\s*(iterations|epochs|steps)', 'iterations', None),
        (r'train for\s*(\d+)\s*(iterations|epochs)', 'iterations', None)
    ]
}

# ========== DOMAIN-SPECIFIC ONTOLOGIES (NEW) ==========
CS_ONTOLOGY = {
    "time_complexity": {
        "description": "Algorithmic time complexity",
        "common_values": ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)"],
        "units": ["big-O"]
    },
    "space_complexity": {
        "description": "Memory usage complexity",
        "common_values": ["O(1)", "O(n)", "O(n^2)"],
        "units": ["big-O"]
    },
    "accuracy": {
        "description": "Model accuracy metric",
        "normal_range": [0.0, 1.0],
        "units": ["%", "decimal"]
    },
    "latency": {
        "description": "Response time",
        "normal_range": [0, 10000],
        "units": ["ms", "seconds"]
    },
    "throughput": {
        "description": "Operations per second",
        "units": ["qps", "rps", "ops/sec"]
    }
}

BIOMED_ONTOLOGY = {
    "ph": {
        "description": "Hydrogen ion concentration",
        "normal_range": [0, 14],
        "units": ["pH"]
    },
    "temperature": {
        "description": "Experimental temperature",
        "normal_range": [-20, 100],
        "units": ["°C", "C", "K"]
    },
    "concentration": {
        "description": "Substance concentration",
        "units": ["mM", "µM", "nM", "M", "mg/mL", "g/L", "%"]
    }
}

# ========== HELPER FUNCTIONS ==========
def get_domain_config(domain: str) -> Dict[str, Any]:
    """Get configuration for a specific domain"""
    if domain == "biomed":
        return {
            "model": BIOMISTRAL_GGUF_MODEL,
            "model_file": BIOMISTRAL_GGUF_FILE,
            "max_tokens": BIOMISTRAL_MAX_TOKENS,
            "timeout": BIOMISTRAL_TIMEOUT,
            "ctx_length": BIOMISTRAL_CTX_LENGTH,
            "n_threads": BIOMISTRAL_N_THREADS,
            "n_gpu_layers": BIOMISTRAL_N_GPU_LAYERS,
            "system_prefix": BIOMED_SYSTEM_PREFIX,
            "ontology": BIOMED_ONTOLOGY,
            "port": DOMAIN_PORTS["biomed"]
        }
    elif domain == "cs":
        return {
            "model": CSMODEL_GGUF_MODEL,
            "model_file": CSMODEL_GGUF_FILE,
            "max_tokens": CSMODEL_MAX_TOKENS,
            "timeout": CSMODEL_TIMEOUT,
            "ctx_length": CSMODEL_CTX_LENGTH,
            "n_threads": CSMODEL_N_THREADS,
            "n_gpu_layers": CSMODEL_N_GPU_LAYERS,
            "system_prefix": CS_SYSTEM_PREFIX,
            "ontology": CS_ONTOLOGY,
            "parameter_patterns": CS_PARAMETER_PATTERNS,
            "port": DOMAIN_PORTS["cs"]
        }
    else:
        return {
            "system_prefix": SYSTEM_PREFIX,
            "port": DOMAIN_PORTS.get("general", 8002)
        }

def get_model_for_domain(domain: str) -> str:
    """Get the model name for a domain"""
    if domain == "biomed":
        return BIOMISTRAL_GGUF_MODEL
    elif domain == "cs":
        return CSMODEL_GGUF_MODEL
    else:
        return MISTRAL_MODEL_NAME
