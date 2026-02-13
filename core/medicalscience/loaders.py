# core/medicalscience/loaders.py
# BioMistral / biomedical GGUF loading with ctransformers (alternative / legacy path)

import asyncio
import logging
import os
from typing import Optional
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

from core.config import (
    # Main loading path variables (we'll use these)
    BIOMISTRAL_TRANSFORMERS_MODEL,  # for reference / logging
    # GGUF fallback / alternative path
    BIOMISTRAL_GGUF_FILE,           # optional local file override
    # If you want to standardize on repo + filename like CS model:
    # (add these to config.py if not present)
    # BIOMISTRAL_REPO_ID = "some/repo/biomistral-gguf"   # ← add if needed
    # BIOMISTRAL_FILENAME = "biomistral-7b.gguf"         # ← add if needed
    BIOMISTRAL_CTX_LENGTH,
    BIOMISTRAL_N_THREADS,
    BIOMISTRAL_N_GPU_LAYERS,
    BIOMISTRAL_MAX_TOKENS,
    BIOMISTRAL_TIMEOUT
)

logger = logging.getLogger("biomed.loaders")

# Global singleton + lock
_biomistral_llm: Optional[AutoModelForCausalLM] = None
_load_lock = asyncio.Lock()


async def _load_biomistral_gguf() -> AutoModelForCausalLM:
    """
    Load BioMistral (or similar biomedical Mistral variant) in GGUF format using ctransformers.
    Caches globally after first successful load.
    """
    global _biomistral_llm

    if _biomistral_llm is not None:
        logger.debug("BioMistral GGUF already loaded (cached)")
        return _biomistral_llm

    async with _load_lock:
        if _biomistral_llm is not None:
            return _biomistral_llm

        logger.info("Starting BioMistral GGUF load...")

        try:
            # 1. Check for explicit local GGUF file override
            if BIOMISTRAL_GGUF_FILE and os.path.isfile(BIOMISTRAL_GGUF_FILE):
                model_path = BIOMISTRAL_GGUF_FILE
                logger.info(f"Using explicit local GGUF override: {model_path}")
            else:
                # 2. Standard Hugging Face download (you'll need to define repo & filename)
                #    Currently many BioMistral GGUF conversions are scattered — pick one reliable source
                #    Example popular one (update as needed):
                repo_id = "TheBloke/BioMistral-7B-GGUF"           # ← very common one
                filename = "biomistral-7b.Q4_K_M.gguf"            # ← good balance quality/size

                logger.info(f"Downloading BioMistral GGUF from: {repo_id}")
                logger.info(f"Target file: {filename}")

                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir="./models",
                    local_dir_use_symlinks=False
                )
                logger.info(f"Model downloaded/available at: {model_path}")

            # Load model
            _biomistral_llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="mistral",           # BioMistral is based on Mistral-7B
                gpu_layers=BIOMISTRAL_N_GPU_LAYERS,
                threads=BIOMISTRAL_N_THREADS,
                context_length=BIOMISTRAL_CTX_LENGTH,
                max_new_tokens=BIOMISTRAL_MAX_TOKENS,
            )

            logger.info("✅ BioMistral-7B GGUF loaded successfully with ctransformers")
            return _biomistral_llm

        except Exception as e:
            logger.exception(f"❌ BioMistral GGUF loading failed: {e}")
            raise RuntimeError(f"BioMistral GGUF failed to load: {str(e)}")


async def generate_biomistral_draft(user_input: str, max_tokens: int = BIOMISTRAL_MAX_TOKENS) -> str:
    """
    Generate a short biomedical-focused summary/draft.
    Falls back gracefully on errors or timeout.
    """
    try:
        llm = await _load_biomistral_gguf()

        prompt = f"""You are a biomedical research assistant.
Focus on experimental parameters, biological processes, pharmacology, molecular interactions, etc.

Query: {user_input}

Extract and summarize concisely (2–4 sentences):
- Key experimental/biological parameters (pH, temperature, concentration, dose, time, etc.)
- Biological context or mechanisms
- Potential implications or variables to consider

Summary:"""

        def run_sync_generation():
            return llm(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.65,
                top_p=0.92,
                repetition_penalty=1.12,
                stop=["</s>", "\n\n\n", "User:", "###", "Summary:"],
            )

        output = await asyncio.wait_for(
            asyncio.to_thread(run_sync_generation),
            timeout=BIOMISTRAL_TIMEOUT
        )

        draft = str(output).strip()
        if not draft or len(draft) < 15:
            draft = "No strong biomedical parameters or context identified."

        logger.info(f"BioMistral biomedical draft generated ({len(draft)} chars)")
        return draft

    except asyncio.TimeoutError:
        logger.warning("BioMistral generation timed out")
        return _fallback_biomed_draft(user_input)

    except Exception as e:
        logger.error(f"BioMistral generation failed: {e}")
        return _fallback_biomed_draft(user_input)


def _fallback_biomed_draft(user_input: str) -> str:
    """Rule-based fallback when model is unavailable"""
    preview = user_input[:90].replace("\n", " ").strip()
    return (
        f"Biomedical context for: \"{preview}...\"\n"
        "Likely relevant factors: pH, temperature, concentration, dose, incubation time, "
        "cell line / organism, nutrient availability, oxygen levels, and strain-specific responses."
    )


# Optional: status helper for health checks / debugging
def get_biomistral_status() -> dict:
    return {
        "loaded": _biomistral_llm is not None,
        "model_type": "BioMistral-7B GGUF (ctransformers)",
        "backend": "ctransformers",
        "path": getattr(_biomistral_llm, "model_path", "not loaded") if _biomistral_llm else None
    }