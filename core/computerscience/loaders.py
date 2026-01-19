# core/computerscience/loaders.py - CS MODEL GGUF WITH CTRANSFORMERS

import asyncio
import logging
from typing import Optional
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from core.config import (
    CSMODEL_GGUF_MODEL, CSMODEL_GGUF_FILE, CSMODEL_CTX_LENGTH,
    CSMODEL_N_THREADS, CSMODEL_N_GPU_LAYERS, CSMODEL_MAX_TOKENS, CSMODEL_TIMEOUT
)

logger = logging.getLogger("cs.loaders")

# Global cache
csmodel_llm: Optional[AutoModelForCausalLM] = None
load_lock = asyncio.Lock()

async def _load_csmodel_gguf() -> AutoModelForCausalLM:
    """Load CS Model GGUF once and cache it"""
    global csmodel_llm
    if csmodel_llm:
        logger.info("CS Model GGUF already loaded (cached)")
        return csmodel_llm

    async with load_lock:
        if csmodel_llm:
            return csmodel_llm

        try:
            logger.info(f"Downloading/loading CS Model GGUF: {CSMODEL_GGUF_FILE}")
            model_path = hf_hub_download(
                repo_id=CSMODEL_GGUF_MODEL,
                filename=CSMODEL_GGUF_FILE,
                local_dir="./models"
            )

            csmodel_llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="mistral",  # Mistral-7B-Instruct is Mistral-based
                gpu_layers=CSMODEL_N_GPU_LAYERS,
                threads=CSMODEL_N_THREADS,
                context_length=CSMODEL_CTX_LENGTH
            )
            logger.info("✅ CS Model GGUF loaded successfully with ctransformers")
            return csmodel_llm

        except Exception as e:
            logger.error(f"❌ CS Model GGUF load failed: {e}")
            raise RuntimeError("CS Model GGUF failed to load")

async def generate_cs_draft(user_input: str, max_tokens: int = CSMODEL_MAX_TOKENS) -> str:
    try:
        llm = await _load_csmodel_gguf()
        
        prompt = f"""You are a computer science expert. Summarize key parameters, algorithms, and computational context from this query in 2-3 sentences.

Query: {user_input}

Summary:"""

        # This must be awaited properly inside asyncio.to_thread
        def run_sync():
            return llm(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.6,
                top_p=0.9,
                stop=["</s>", "\n\n", "User:"]
            )

        output = await asyncio.wait_for(
            asyncio.to_thread(run_sync),
            timeout=CSMODEL_TIMEOUT
        )

        # Now output is a string
        draft = str(output).strip()
        if not draft or len(draft) < 10:
            draft = "No clear computational parameters detected."

        logger.info(f"CS Model draft generated ({len(draft)} chars)")
        return draft

    except asyncio.TimeoutError:
        logger.warning("CS Model timeout — using fallback")
        return "Computer science analysis: algorithmic complexity, data structures, and computational efficiency considerations."
    except Exception as e:
        logger.error(f"CS Model error: {e}")
        return "Computer science context unavailable due to technical issue."

def _fallback_cs_draft(user_input: str) -> str:
    """Fallback when CS Model fails"""
    return (
        f"Computer science context: {user_input[:80]}...\n"
        "Key factors: algorithmic approach, time/space complexity, data structures, scalability, and implementation constraints."
    )
