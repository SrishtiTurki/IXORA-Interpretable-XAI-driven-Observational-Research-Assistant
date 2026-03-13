"""
core/model_loader.py - COMPLETE OPTIMIZED Model Loader
Optimizations:
- Better caching for embeddings
- Batch processing
- Faster domain classification
- Lazy loading for domain-specific models
- Comprehensive logging
"""

import asyncio
import logging
import time
import os
import hashlib
from typing import Dict, Any, Optional, List, Union
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from transformers import pipeline
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger("core.model_loader")

from core.config import (
    # Qwen config
    QWEN_USE_FROM_PRETRAINED,
    QWEN_REPO_ID,
    QWEN_FILENAME,
    QWEN_CTX_LENGTH,
    QWEN_N_THREADS,
    QWEN_N_GPU_LAYERS,
    QWEN_MAX_TOKENS,
    QWEN_TIMEOUT,
    
    # BioMistral config
    BIOMISTRAL_USE_TRANSFORMERS,
    BIOMISTRAL_TRANSFORMERS_MODEL,
    BIOMISTRAL_MAX_TOKENS,
    BIOMISTRAL_TIMEOUT,
    BIOMISTRAL_DEVICE,
    
    # CS Model config
    CSMODEL_USE_FROM_PRETRAINED,
    CSMODEL_REPO_ID,
    CSMODEL_FILENAME,
    CSMODEL_CTX_LENGTH,
    CSMODEL_N_THREADS,
    CSMODEL_N_GPU_LAYERS,
    CSMODEL_MAX_TOKENS,
    CSMODEL_TIMEOUT,
    
    # Mistral API config
    MISTRAL_USE_API,
    MISTRAL_API_KEY
)

import os
print(f"CSMODEL_ENABLED: {os.getenv('CSMODEL_ENABLED')}")
print(f"CSMODEL_USE_FROM_PRETRAINED: {os.getenv('CSMODEL_USE_FROM_PRETRAINED')}")

# ==================== CONFIGURATION ====================
class ModelConfig:
    """Configuration for all models"""
    
    LOAD_TIMEOUTS = {
        "biomistral": 30.0,
        "cs_model": 30.0,
        "qwen": 30.0,
        "mistral_api": 10.0,
        "embedding": 15.0,
        "reward": 5.0
    }
    
    MAX_MEMORY_MB = {
        "biomistral": 4000,
        "cs_model": 4000,
        "qwen": 3000,
        "embedding": 500,
        "total": 8000
    }
    
    DOMAIN_PRIORITIES = {
        "biomed": ["embedding", "biomistral", "qwen", "mistral_api", "reward"],
        "cs": ["embedding", "cs_model", "qwen", "mistral_api", "reward"],
        "general": ["embedding", "mistral_api", "reward"]
    }


# ==================== MODEL STATUS ====================
class ModelStatus:
    """Track model loading status"""
    
    def __init__(self, name: str):
        self.name = name
        self.loaded = False
        self.loading = False
        self.load_time = 0.0
        self.last_used = 0.0
        self.memory_mb = 0.0
        self.error: Optional[str] = None
        self.instance: Any = None


# ==================== OPTIMIZED MODEL LOADER ====================
class OptimizedModelLoader:
    """
    Optimized model loader with:
    - Fast embedding caching
    - Batch processing
    - Lazy loading
    - Domain-aware classification
    - Comprehensive logging
    """
    
    def __init__(self):
        # Core models
        self.embedding_model = None
        self.biomistral_model = None
        self.cs_model = None
        self.qwen_model = None
        self.reward_model = None
        
        # Mistral API status
        self.mistral_api_available = False
        self.mistral_api_checked = False
        
        # Caches
        self.embedding_cache = {}
        self.domain_cache = {}
        
        # Status tracking
        self.model_status = {}
        
        # Domain configs for embeddings
        self.domain_configs = {
            "biomed": {
                "keywords": ["ph", "temperature", "biomass", "yeast", "enzyme", "protein",
                           "cell", "bacteria", "fermentation", "growth", "culture"],
                "dimension": 384
            },
            "cs": {
                "keywords": ["algorithm", "complexity", "learning rate", "batch size",
                           "neural network", "optimization", "gradient", "model"],
                "dimension": 384
            },
            "general": {
                "keywords": [],
                "dimension": 384
            }
        }
        
        logger.info("🚀 OptimizedModelLoader initialized")
    
    def _get_cache_key(self, text: str, domain: str = "") -> str:
        """Generate cache key"""
        combined = f"{domain}:{text[:200]}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    # ==================== EMBEDDING SERVICE ====================
    
    async def load_embedding_model(self):
        """Load embedding model lazily"""
        if self.embedding_model is not None:
            return self.embedding_model
        
        logger.info("📥 Loading embedding model (all-MiniLM-L6-v2)...")
        start_time = time.time()
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            load_time = time.time() - start_time
            logger.info(f"✅ Embedding model loaded in {load_time:.2f}s")
            return self.embedding_model
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.embedding_model = None
            return None
    
    async def get_embeddings(
        self, 
        texts: Union[str, List[str]], 
        domain: str = "general",
        use_cache: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """
        OPTIMIZED: Get embeddings with caching and batch processing
        """
        model = await self.load_embedding_model()
        if model is None:
            # Return dummy embeddings
            if isinstance(texts, str):
                return [0.0] * 384
            return [[0.0] * 384 for _ in texts]
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # Check cache
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, domain)
                if cache_key in self.embedding_cache:
                    embeddings.append(self.embedding_cache[cache_key])
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Compute uncached embeddings in batch (FASTER!)
        if uncached_texts:
            try:
                # Run in thread pool
                new_embeddings = await asyncio.to_thread(
                    model.encode,
                    uncached_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Update cache and results
                for i, text_idx in enumerate(uncached_indices):
                    emb = new_embeddings[i].tolist()
                    embeddings[text_idx] = emb
                    
                    if use_cache:
                        cache_key = self._get_cache_key(texts[text_idx], domain)
                        self.embedding_cache[cache_key] = emb
                        
                        # Limit cache size
                        if len(self.embedding_cache) > 1000:
                            keys_to_remove = list(self.embedding_cache.keys())[:100]
                            for k in keys_to_remove:
                                del self.embedding_cache[k]
                
            except Exception as e:
                logger.error(f"❌ Embedding generation failed: {e}")
                for text_idx in uncached_indices:
                    embeddings[text_idx] = [0.0] * 384
        
        # Return single embedding or list
        if single_input:
            return embeddings[0]
        return embeddings
    
    # ==================== BIOMISTRAL/MEDGEMMA ====================
    
    async def load_biomistral(self):
        """Load BioGPT using transformers"""
        if self.biomistral_model is not None:
            return self.biomistral_model
        
        if not BIOMISTRAL_USE_TRANSFORMERS or not BIOMISTRAL_TRANSFORMERS_MODEL:
            logger.warning("⚠️ BioMistral transformers disabled in config")
            return None
        
        logger.info(f"📥 Loading BioGPT: {BIOMISTRAL_TRANSFORMERS_MODEL}")
        start_time = time.time()
        
        try:
            # Set device
            device = 0 if torch.cuda.is_available() and BIOMISTRAL_DEVICE != "cpu" else -1
            
            # Load with transformers pipeline
            self.biomistral_model = await asyncio.to_thread(
                pipeline,
                "text-generation",
                model=BIOMISTRAL_TRANSFORMERS_MODEL,
                device=device,
                max_length=BIOMISTRAL_MAX_TOKENS,
                truncation=True
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ BioGPT loaded via transformers in {load_time:.2f}s")
            return self.biomistral_model
            
        except Exception as e:
            logger.error(f"❌ BioGPT loading failed: {e}", exc_info=True)
            self.biomistral_model = None
            return None
    # In model_loader.py - Update the generate_with_biomistral function

    async def generate_with_biomistral(self, prompt: str, max_new_tokens: int = 350) -> str:
        model_pipe = await self.load_biomistral()
        if model_pipe is None:
            logger.info("No local BioMistral → falling back to API")
            from core.mistral import call_mistral_api
            try:
                return await call_mistral_api(prompt, max_tokens=max_new_tokens + 80)
            except Exception as e:
                return f"[Generation failed] {str(e)}"

        try:
            # Safety truncation
            max_prompt_chars = 1500  # Reduced from 2200
            if len(prompt) > max_prompt_chars:
                prompt = prompt[-max_prompt_chars:]

            # FIX: Use simpler generation parameters
            output = await asyncio.to_thread(
                model_pipe,
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,  # Reduced from 1.15-1.2
                num_return_sequences=1,
                truncation=True
            )

            # Extract text safely
            if isinstance(output, list) and len(output) > 0:
                generated = output[0].get("generated_text", "").strip()
            elif isinstance(output, dict):
                generated = output.get("generated_text", "").strip()
            else:
                generated = str(output).strip()

            # Check if generation is reasonable
            if not generated or len(generated) < 10:
                logger.warning(f"Short generation ({len(generated)} chars)")
                # Fallback immediately to API
                from core.mistral import call_mistral_api
                try:
                    return await call_mistral_api(prompt, max_tokens=max_new_tokens + 100)
                except Exception as e:
                    return f"[Fallback also failed] {str(e)}"

            logger.info(f"✅ BioMistral generated {len(generated)} chars")
            return generated

        except Exception as e:
            logger.error(f"❌ BioMistral generation failed: {e}")
            from core.mistral import call_mistral_api
            try:
                return await call_mistral_api(prompt, max_tokens=max_new_tokens + 120)
            except Exception as e2:
                return f"[Both local and API failed] {str(e2)}"

    # ==================== CS MODEL ====================
    
    async def load_cs_model(self):
        """Load CS model with proper async handling"""
        if self.cs_model is not None:
            logger.info("♻️ CS model already loaded, returning cached instance")
            return self.cs_model
        
        logger.info(f"📥 Loading CS model: {CSMODEL_REPO_ID} / {CSMODEL_FILENAME}")
        
        try:
            start_time = time.time()
            
            # Load model in thread pool
            logger.info("⏳ Downloading/loading model file...")
            self.cs_model = await asyncio.to_thread(
                Llama.from_pretrained,
                repo_id=CSMODEL_REPO_ID,
                filename=CSMODEL_FILENAME,
                n_ctx=CSMODEL_CTX_LENGTH,
                n_threads=CSMODEL_N_THREADS,
                n_gpu_layers=CSMODEL_N_GPU_LAYERS,
                verbose=True,
                logits_all=False,
                embedding=False
            )
            
            logger.info("✅ Model file loaded, running smoke test...")
            
            # Quick smoke test - run model inference in thread pool
            # Note: we need to call the model as a function with proper parameters
            def run_smoke_test():
                """Synchronous smoke test function"""
                try:
                    result = self.cs_model(
                        prompt="Hello,",
                        max_tokens=5,
                        echo=False,
                        temperature=0.7,
                        stop=None
                    )
                    return result
                except Exception as e:
                    logger.error(f"Smoke test execution failed: {e}")
                    raise
            
            # Run smoke test in thread pool
            test_output = await asyncio.to_thread(run_smoke_test)
            
            # Validate smoke test output
            if not test_output:
                raise ValueError("CS model smoke test returned None")
            
            if 'choices' not in test_output or len(test_output['choices']) == 0:
                raise ValueError("CS model smoke test returned invalid structure")
            
            generated_text = test_output['choices'][0].get('text', '').strip()
            
            if not generated_text:
                logger.warning("⚠️ CS model smoke test produced empty output (might be normal for some models)")
                # Don't fail here - some models might produce empty output for "Hello,"
                # Instead, we'll mark it as loaded but warn the user
            
            load_time = time.time() - start_time
            logger.info(f"✅ CS model loaded successfully in {load_time:.2f}s")
            
            if generated_text:
                logger.info(f"✅ Smoke test output: '{generated_text}'")
            
            return self.cs_model
            
        except Exception as e:
            logger.error(f"❌ CS model loading failed: {str(e)}", exc_info=True)
            self.cs_model = None
            
            # Provide helpful error messages based on error type
            if "Connection" in str(e) or "timeout" in str(e).lower():
                logger.error("💡 Hint: Check your internet connection or try again later")
            elif "disk" in str(e).lower() or "space" in str(e).lower():
                logger.error("💡 Hint: Check available disk space")
            elif "memory" in str(e).lower() or "RAM" in str(e).lower():
                logger.error("💡 Hint: Try reducing n_ctx or close other applications")
            elif "model" in str(e).lower() and "not found" in str(e).lower():
                logger.error(f"💡 Hint: Model {CSMODEL_REPO_ID}/{CSMODEL_FILENAME} may not exist")
            
            return None


    # Alternative version with timeout protection
    async def load_cs_model_with_timeout(self, timeout_seconds=300):
        """
        Load CS model with timeout protection
        Use this if loading sometimes hangs
        """
        try:
            return await asyncio.wait_for(
                self.load_cs_model(),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ CS model loading timed out after {timeout_seconds}s")
            self.cs_model = None
            return None

    async def generate_with_cs_model(self, prompt: str, max_tokens: int = 300) -> str:
        """
        Generate text using the CS model (GGUF via llama-cpp-python).
        Includes safeguards against long prompts and empty/short outputs.
        Falls back to Mistral API on failure or poor generation quality.
        """
        model = await self.load_cs_model()
        
        if model is None:
            logger.info("CS model not loaded → falling back to Mistral API")
            from core.mistral import call_mistral_api
            try:
                return await call_mistral_api(prompt, max_tokens=max_tokens + 100)
            except Exception as e:
                logger.error(f"Mistral API fallback failed: {e}")
                return f"Unable to generate response. Error: {str(e)}"

        try:
            # ── Safeguard: truncate very long prompts ─────────────────────────────
            # llama.cpp has strict context limits — better to cut early than crash
            max_input_chars = 4000  # conservative ~3000–3500 tokens depending on tokenizer
            if len(prompt) > max_input_chars:
                prompt = prompt[-max_input_chars:]
                logger.warning(f"CS prompt truncated to ~{max_input_chars} chars to avoid context overflow")

            # ── Generate ──────────────────────────────────────────────────────────
            result = await asyncio.to_thread(
                model,
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                stop=["</s>", "\n\n\n", "<|endoftext|>"],
                echo=False,
                # Optional: can add more control if needed
                # repeat_penalty=1.1,
                # mirostat_mode=2, mirostat_tau=5.0, mirostat_eta=0.1
            )

            generated_text = result['choices'][0]['text'].strip()

            # ── Quality check (very important for GGUF models) ───────────────────
            if not generated_text or len(generated_text) < 20:
                logger.warning(f"CS model produced empty or very short output ({len(generated_text)} chars) → falling back")
                raise ValueError("Empty or too short generation from CS model")

            logger.info(f"✅ CS model generated {len(generated_text)} chars successfully")
            return generated_text

        except Exception as e:
            logger.error(f"❌ CS generation failed: {str(e)}", exc_info=True)
            
            # Fallback to Mistral API
            from core.mistral import call_mistral_api
            try:
                logger.info("Falling back to Mistral API after CS model failure")
                api_result = await call_mistral_api(
                    prompt=prompt,
                    max_tokens=max_tokens + 150,  # give a bit more room
                    temperature=0.7
                )
                return api_result.strip()
            except Exception as e2:
                logger.error(f"❌ Mistral API fallback also failed: {e2}")
                return (
                    "Generation failed on both local CS model and API fallback.\n"
                    f"Local error: {str(e)}\n"
                    f"API error: {str(e2)}"
                )
    
    # ==================== QWEN MODEL ====================
    
    async def load_qwen(self):
        """Load Qwen using llama-cpp-python's from_pretrained"""
        if self.qwen_model is not None:
            return self.qwen_model
        
        if not QWEN_USE_FROM_PRETRAINED:
            logger.info("ℹ️ Qwen from_pretrained disabled")
            self.qwen_model = "api_fallback"
            return "api_fallback"
        
        logger.info(f"📥 Downloading Qwen: {QWEN_REPO_ID}/{QWEN_FILENAME}")
        start_time = time.time()
        
        try:
            self.qwen_model = await asyncio.to_thread(
                Llama.from_pretrained,
                repo_id=QWEN_REPO_ID,
                filename=QWEN_FILENAME,
                n_ctx=QWEN_CTX_LENGTH,
                n_threads=QWEN_N_THREADS,
                n_gpu_layers=QWEN_N_GPU_LAYERS,
                verbose=True
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Qwen loaded via from_pretrained in {load_time:.2f}s")
            return self.qwen_model
            
        except Exception as e:
            logger.error(f"❌ Qwen from_pretrained failed: {e}", exc_info=True)
            self.qwen_model = "api_fallback"
            return "api_fallback"
    
    async def generate_with_qwen(self, prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> str:
        """
        Generate with Qwen or fallback to Mistral API
        """
        logger.info(f"🤖 Qwen generation (or fallback): {len(prompt)} chars")
        
        # Try to load Qwen
        model = await self.load_qwen()
        
        # If Qwen is not available (returns "api_fallback"), use Mistral API directly
        if model == "api_fallback" or model is None:
            logger.info("Using Mistral API fallback for generation")
            from core.mistral import call_mistral_api
            try:
                return await call_mistral_api(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            except Exception as e:
                logger.error(f"❌ Mistral API fallback failed: {e}")
                return ""
        
        # If Qwen is loaded, use it
        try:
            logger.info("Using local Qwen GGUF...")
            output = await asyncio.to_thread(
                model,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                repeat_penalty=1.1,
                stop=["</s>", "<|im_end|>", "<|endoftext|>"],
                echo=False
            )
            generated = output["choices"][0]["text"].strip()
            logger.info(f"✅ Local Qwen generated {len(generated)} chars")
            return generated
            
        except Exception as e:
            logger.error(f"❌ Local Qwen generation failed: {e}")
            # Fallback to Mistral API
            from core.mistral import call_mistral_api
            try:
                return await call_mistral_api(prompt, max_tokens=max_tokens)
            except Exception as e2:
                return f"Unable to generate response. Error: {str(e2)}"
    
    # ==================== MISTRAL API ====================
    
    async def check_mistral_api(self):
        """Check if Mistral API is available"""
        if self.mistral_api_checked:
            return self.mistral_api_available
        
        from core.mistral import call_mistral_api
        
        logger.info("🔍 Checking Mistral API availability...")
        
        if not MISTRAL_USE_API:
            logger.info("ℹ️ Mistral API disabled in config")
            self.mistral_api_available = False
            self.mistral_api_checked = True
            return False
        
        if not MISTRAL_API_KEY:
            logger.warning("⚠️ Mistral API key not configured")
            self.mistral_api_available = False
            self.mistral_api_checked = True
            return False
        
        try:
            # Try a simple API call
            test_response = await call_mistral_api("test", max_tokens=5)
            if test_response:
                logger.info("✅ Mistral API is available")
                self.mistral_api_available = True
            else:
                logger.warning("⚠️ Mistral API returned empty response")
                self.mistral_api_available = False
        except Exception as e:
            logger.error(f"❌ Mistral API check failed: {e}")
            self.mistral_api_available = False
        
        self.mistral_api_checked = True
        return self.mistral_api_available
    
    async def get_mistral_api_status(self):
        """Get Mistral API status with detailed info"""
        status = {
            "configured": MISTRAL_USE_API and bool(MISTRAL_API_KEY),
            "use_api": MISTRAL_USE_API,
            "api_key_exists": bool(MISTRAL_API_KEY),
            "tested": self.mistral_api_checked,
            "available": self.mistral_api_available
        }
        
        return status
    
    # ==================== REWARD MODEL ====================
    
    async def load_reward_model(self):
        """Load reward model if available"""
        if self.reward_model is not None:
            return self.reward_model
        
        logger.info("📥 Loading reward model...")
        start_time = time.time()
        
        try:
            from core.rlhf.reward_model import load_reward_model
            self.reward_model = await asyncio.to_thread(load_reward_model)
            
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded reward model in {load_time:.2f}s")
            return self.reward_model
        except Exception as e:
            logger.warning(f"⚠️ Reward model not available: {e}")
            return None
    
    # ==================== DOMAIN CLASSIFICATION ====================
    
    async def classify_domain(self, text: str) -> Dict[str, float]:
        """
        FAST domain classification using keywords AND intent analysis.
        """
        # Check cache
        cache_key = hashlib.md5(text[:200].encode()).hexdigest()
        if cache_key in self.domain_cache:
            return self.domain_cache[cache_key]
        
        text_lower = text.lower()
        
        # Count keyword matches
        scores = {}
        for domain, config in self.domain_configs.items():
            keywords = config.get("keywords", [])
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[domain] = score
        
        # SPECIAL CASE: If all scores are 0, use intent-based classification
        if all(score == 0 for score in scores.values()):
            # Analyze query intent for domain hints
            if any(word in text_lower for word in ["experiment", "study", "research", "trial", "lab"]):
                # Could be biomedical or general research
                scores["biomed"] = 0.4
                scores["general"] = 0.3
            elif any(word in text_lower for word in ["code", "program", "algorithm", "software", "model", "neural", "network"]):
                # Computer science
                scores["cs"] = 0.6
                scores["general"] = 0.2
            elif any(word in text_lower for word in ["ph", "temperature", "cell", "protein", "enzyme", "biomass", "yeast"]):
                # Definitely biomedical
                scores["biomed"] = 0.8
            elif any(word in text_lower for word in ["optimize", "parameter", "setting", "condition", "variable"]):
                # Could be any domain - default to general with slight biomedical bias
                scores["general"] = 0.5
                scores["biomed"] = 0.3
                scores["cs"] = 0.2
            else:
                # Generic query - default to general
                scores["general"] = 0.6
                scores["biomed"] = 0.2
                scores["cs"] = 0.2
        
        # Ensure we always have a non-zero total
        total = sum(scores.values())
        if total == 0:
            scores["general"] = 1.0
            total = 1.0
        
        normalized_scores = {
            domain: round(score / total, 3)
            for domain, score in scores.items()
        }
        
        # Cache result
        self.domain_cache[cache_key] = normalized_scores
        
        # Limit cache size
        if len(self.domain_cache) > 500:
            keys_to_remove = list(self.domain_cache.keys())[:50]
            for k in keys_to_remove:
                del self.domain_cache[k]
        
        logger.info(f"🌍 Domain classification for '{text[:50]}...': {normalized_scores}")
        return normalized_scores
    
    # ==================== GENERIC MODEL ACCESS ====================
    
    async def get_model(self, model_name: str, domain: str = "general"):
        """Get any model by name"""
        if model_name == "embedding":
            return await self.load_embedding_model()
        elif model_name == "biomistral":
            return await self.load_biomistral()
        elif model_name == "cs_model":
            return await self.load_cs_model()
        elif model_name == "qwen":
            return await self.load_qwen()
        elif model_name == "reward":
            return await self.load_reward_model()
        else:
            logger.warning(f"⚠️ Unknown model: {model_name}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Simple status check"""
        return {
            "embedding": {
                "loaded": self.embedding_model is not None,
                "type": "SentenceTransformer"
            },
            "biomistral": {
                "loaded": self.biomistral_model is not None and self.biomistral_model != "api_fallback",
                "type": "BioGPT"
            },
            "cs_model": {
                "loaded": self.cs_model is not None,
                "type": "Qwen-Coder"
            },
            "qwen": {
                "loaded": self.qwen_model is not None and self.qwen_model != "api_fallback",
                "type": "Qwen-0.5B"
            },
            "mistral_api": {
                "available": self.mistral_api_available
            },
            "reward": {
                "loaded": self.reward_model is not None
            }
        }

    def print_status(self):
        """Simple status display"""
        print("\n" + "="*50)
        print("📊 MODEL STATUS")
        print("="*50)
        
        status = self.get_status()
        
        models = [
            ("🔤 Embedding", status["embedding"]["loaded"]),
            ("🧬 Biomedical", status["biomistral"]["loaded"]),
            ("🤖 Qwen", status["qwen"]["loaded"]),
            ("💻 CS Model", status["cs_model"]["loaded"]),
            ("⭐ Reward", status["reward"]["loaded"]),
        ]
        
        for name, loaded in models:
            status_symbol = "✅ Loaded" if loaded else "❌ Not loaded"
            print(f"{name}: {status_symbol}")
        
        print(f"\n🌐 Mistral API: {'✅ Available' if status['mistral_api']['available'] else '❌ Unavailable'}")
        print("="*50)
        
# ==================== GLOBAL INSTANCE ====================
model_loader = OptimizedModelLoader()


# ==================== HELPER FUNCTIONS ====================

async def get_embeddings(*args, **kwargs):
    """Wrapper for get_embeddings"""
    return await model_loader.get_embeddings(*args, **kwargs)

async def classify_domain(*args, **kwargs):
    """Wrapper for classify_domain"""
    return await model_loader.classify_domain(*args, **kwargs)

async def generate_with_qwen(prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> str:
    """Wrapper for Qwen generation"""
    return await model_loader.generate_with_qwen(prompt, max_tokens, temperature)

async def generate_with_biomistral(prompt: str, max_tokens: int = 300) -> str:
    """Wrapper for BioMistral generation"""
    return await model_loader.generate_with_biomistral(prompt, max_tokens)

async def generate_with_cs_model(prompt: str, max_tokens: int = 300) -> str:
    """Wrapper for CS model generation"""
    return await model_loader.generate_with_cs_model(prompt, max_tokens)

def get_model_status() -> Dict[str, Any]:
    """Get status of all models (for health checks)"""
    return model_loader.get_status()

async def startup_models(domain: str = "biomed", warmup: bool = True):
    """
    Initialize models on startup with comprehensive logging.
    This function is called from main.py during startup.
    """
    logger.info(f"🚀 Starting model initialization for {domain} domain")
    
    try:
        # Check Mistral API first
        mistral_ok = await model_loader.check_mistral_api()
        if mistral_ok:
            logger.info("✅ Mistral API is ready")
        else:
            logger.warning("⚠️ Mistral API not available - local models will be used")
        
        # Load essential models based on domain
        if domain == "biomed":
            logger.info("🧬 Loading biomedical models...")
            # Try to load embedding model
            try:
                await model_loader.load_embedding_model()
            except Exception as e:
                logger.warning(f"⚠️ Embedding model failed: {e}")
            
            # Try to load BioMistral
            try:
                await model_loader.load_biomistral()
            except Exception as e:
                logger.warning(f"⚠️ BioMistral failed: {e}")
        
        elif domain == "cs":
            logger.info("💻 Loading computer science models...")
            # Try to load embedding model
            try:
                await model_loader.load_embedding_model()
            except Exception as e:
                logger.warning(f"⚠️ Embedding model failed: {e}")
            
            # Try to load CS model
            try:
                await model_loader.load_cs_model()
            except Exception as e:
                logger.warning(f"⚠️ CS model failed: {e}")
        
        # Always try to load Qwen for parameter extraction
        logger.info("🤖 Loading Qwen for parameter extraction...")
        try:
            await model_loader.load_qwen()
        except Exception as e:
            logger.warning(f"⚠️ Qwen failed: {e}")
        
        # Optional warmup
        if warmup:
            logger.info("🔥 Warming up models...")
            try:
                # Quick test with Mistral API
                if mistral_ok:
                    from core.mistral import call_mistral_api
                    test_prompt = "Hello"
                    test_result = await call_mistral_api(test_prompt, max_tokens=5)
                    if test_result:
                        logger.info("✅ Model warmup successful")
            except Exception as e:
                logger.warning(f"⚠️ Warmup failed: {e}")
        
        # Print final status
        model_loader.print_status()
        
        # Get final status
        status = model_loader.get_status()
        loaded_count = sum(1 for k, v in status.items() 
                          if k in ["embedding", "biomistral", "cs_model", "qwen", "reward"] 
                          and v.get("loaded", False))
        
        logger.info(f"✅ Startup complete: {loaded_count}/5 models ready")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model startup failed: {e}", exc_info=True)
        return False


# Alias for backward compatibility
initialize_models = startup_models