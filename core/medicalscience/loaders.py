# core/medicalscience/loaders.py - SWITCHED TO BIOMISTRAL-7B-GGUF WITH CTRANSFORMERS

import asyncio
import logging
from typing import Optional
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from core.config import (
    BIOMISTRAL_GGUF_MODEL, BIOMISTRAL_GGUF_FILE, BIOMISTRAL_CTX_LENGTH,
    BIOMISTRAL_N_THREADS, BIOMISTRAL_N_GPU_LAYERS, BIOMISTRAL_MAX_TOKENS, BIOMISTRAL_TIMEOUT
)

logger = logging.getLogger("biomed.loaders")

# Global cache
biomistral_llm: Optional[AutoModelForCausalLM] = None
load_lock = asyncio.Lock()

async def _load_biomistral_gguf() -> AutoModelForCausalLM:
    """Load BioMistral GGUF model once and cache it"""
    global biomistral_llm
    if biomistral_llm:
        logger.info("BioMistral GGUF already loaded (cached)")
        return biomistral_llm

    async with load_lock:
        if biomistral_llm:
            return biomistral_llm

        try:
            logger.info(f"Downloading/loading BioMistral GGUF: {BIOMISTRAL_GGUF_FILE}")
            model_path = hf_hub_download(
                repo_id=BIOMISTRAL_GGUF_MODEL,
                filename=BIOMISTRAL_GGUF_FILE,
                local_dir="./models"  # or your preferred cache dir
            )

            biomistral_llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="mistral",  # BioMistral is Mistral-based
                gpu_layers=BIOMISTRAL_N_GPU_LAYERS,
                threads=BIOMISTRAL_N_THREADS,
                context_length=BIOMISTRAL_CTX_LENGTH
            )
            logger.info("✅ BioMistral-7B-GGUF loaded successfully with ctransformers")
            return biomistral_llm

        except Exception as e:
            logger.error(f"❌ BioMistral GGUF load failed: {e}")
            raise RuntimeError("BioMistral GGUF failed to load")

async def generate_biomistral_draft(user_input: str, max_tokens: int = BIOMISTRAL_MAX_TOKENS) -> str:
    try:
        llm = await _load_biomistral_gguf()
        
        prompt = f"""You are a biomedical expert. Summarize key parameters and context from this query in 2-3 sentences.

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
            timeout=BIOMISTRAL_TIMEOUT
        )

        # Now output is a string
        draft = str(output).strip()  # Safe
        if not draft or len(draft) < 10:
            draft = "No clear biomedical parameters detected."

        logger.info(f"BioMistral draft generated ({len(draft)} chars)")
        return draft

    except asyncio.TimeoutError:
        logger.warning("BioMistral timeout — using fallback")
        return "Biomedical analysis: standard CBC and vital signs monitoring recommended."
    except Exception as e:
        logger.error(f"BioMistral error: {e}")
        return "Biomedical context unavailable due to technical issue."

def _fallback_biomed_draft(user_input: str) -> str:
    """Fallback when BioMistral fails"""
    return (
        f"Biomedical context: {user_input[:80]}...\n"
        "Key factors: pH, temperature, nutrients, aeration, incubation time, and strain-specific responses."
    )