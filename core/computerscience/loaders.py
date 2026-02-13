# core/computerscience/loaders.py
# CS MODEL GGUF loading with ctransformers (alternative / legacy path)

import asyncio
import os
import logging
from typing import Optional
from ctransformers import AutoModelForCausalLM
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from core.config import (
    CSMODEL_REPO_ID,
    CSMODEL_FILENAME,
    CSMODEL_GGUF_FILE,      # optional local override
    CSMODEL_CTX_LENGTH,
    CSMODEL_N_THREADS,
    CSMODEL_N_GPU_LAYERS,
    CSMODEL_MAX_TOKENS,
    CSMODEL_TIMEOUT
)

logger = logging.getLogger("cs.loaders")

# Global singleton
_csmodel_llm: Optional[Llama] = None  # Changed from AutoModelForCausalLM to Llama

def _load_csmodel_gguf(
    model_path: Optional[str] = None,
    repo_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
    filename: str = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    ctx_length: int = 8192,
    n_threads: int = 2,
    n_gpu_layers: int = 0,
    max_tokens: int = 512
) -> Llama:
    """Load CS Model - handles both local files and auto-download"""
    global _csmodel_llm
    
    if _csmodel_llm is not None:
        return _csmodel_llm
    
    if model_path and os.path.exists(model_path):
        # Load from local file
        logger.info(f"Loading from: {model_path}")
        _csmodel_llm = Llama(
            model_path=model_path,
            n_ctx=ctx_length,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=True
        )
    else:
        # Auto-download from HuggingFace
        logger.info(f"Downloading: {repo_id}/{filename}")
        _csmodel_llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=ctx_length,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=True
        )
    
    # Test the model
    try:
        test = _csmodel_llm("Hello", max_tokens=5, echo=False)
        if not test['choices'][0]['text'].strip():
            logger.warning("Model test returned empty text")
        else:
            logger.info(f"✅ CS Model loaded successfully! Test output: '{test['choices'][0]['text'].strip()}'")
    except Exception as e:
        logger.error(f"Model test failed: {e}")
    
    return _csmodel_llm

async def generate_cs_draft(user_input: str, max_tokens: int = CSMODEL_MAX_TOKENS) -> str:
    """
    Generate a short CS-focused summary/draft using the loaded model.
    Falls back gracefully on timeout or error.
    """
    try:
        # Load the model if not already loaded
        if _csmodel_llm is None:
            # Run the synchronous loading in a thread
            llm = await asyncio.to_thread(
                _load_csmodel_gguf,
                CSMODEL_GGUF_FILE,
                CSMODEL_REPO_ID,
                CSMODEL_FILENAME,
                CSMODEL_CTX_LENGTH,
                CSMODEL_N_THREADS,
                CSMODEL_N_GPU_LAYERS,
                max_tokens
            )
        else:
            llm = _csmodel_llm

        prompt = f"""You are a computer science research assistant.
Focus on algorithms, complexity, optimization, data structures, ML hyperparameters, etc.

Query: {user_input}

Extract and summarize:
- Key parameters / hyperparameters
- Algorithms or methods mentioned
- Computational considerations (time/space complexity, scalability, etc.)

Summary (2-4 sentences):"""

        # Run the synchronous generation in a thread
        output = await asyncio.to_thread(
            llm,
            prompt,
            max_tokens=max_tokens,
            temperature=0.65,
            top_p=0.92,
            stop=["</s>", "\n\n\n", "User:", "###"],
            echo=False  # Changed from max_new_tokens to max_tokens
        )

        # Extract text from the response
        if isinstance(output, dict) and 'choices' in output and len(output['choices']) > 0:
            draft = output['choices'][0].get('text', '').strip()
        else:
            draft = str(output).strip()
            
        if not draft or len(draft) < 15:
            draft = "No strong computational parameters or context identified."

        logger.info(f"CS draft generated ({len(draft)} chars)")
        return draft

    except asyncio.TimeoutError:
        logger.warning("CS model generation timed out")
        return _fallback_cs_draft(user_input)

    except Exception as e:
        logger.error(f"CS model generation failed: {e}")
        return _fallback_cs_draft(user_input)


def _fallback_cs_draft(user_input: str) -> str:
    """Simple rule-based fallback when model is unavailable"""
    preview = user_input[:90].replace("\n", " ").strip()
    return (
        f"Computer science context for: \"{preview}...\"\n"
        "Likely relevant factors: algorithmic approach, time/space complexity, "
        "data structures, optimization techniques, scalability, implementation trade-offs."
    )


# Optional: expose status helper for debugging/health checks
def get_cs_model_status() -> dict:
    return {
        "loaded": _csmodel_llm is not None,
        "model_type": "Qwen2.5-Coder GGUF (llama-cpp-python)",
        "backend": "llama-cpp-python",
        "path": getattr(_csmodel_llm, "model_path", "not loaded") if _csmodel_llm else None
    }
