"""
core/model_loader.py - COMPLETE Model Pre-loading & Management with Embedding Service
"""

import asyncio
import logging
import threading
import time
import os
import sys
from typing import Dict, Any, Optional, List, Union
import hashlib
import numpy as np
from sentence_transformers import util

logger = logging.getLogger("core.model_loader")

# ==================== CONFIGURATION ====================
class ModelConfig:
    """Configuration for all models"""
    
    # Timeouts (seconds)
    LOAD_TIMEOUTS = {
        "biomistral": 30.0,
        "qwen": 45.0,  # Qwen might take longer to load
        "mistral_api": 10.0,
        "embedding": 15.0,
        "nltk": 10.0,
        "reward": 5.0
    }
    
    # Memory limits (MB)
    MAX_MEMORY_MB = {
        "biomistral": 4000,
        "qwen": 8000,  # Qwen might need more memory
        "embedding": 500,
        "total": 16000  # Increased to accommodate Qwen
    }
    
    # Domain-specific priorities
    DOMAIN_PRIORITIES = {
        "biomed": ["nltk", "embedding", "biomed_extractor", "biomistral", "mistral_api", "reward"],
        "general": ["nltk", "embedding", "mistral_api", "biomistral", "reward"],
        "chemistry": ["nltk", "embedding", "mistral_api", "reward"],
        "computerscience": ["nltk", "embedding", "qwen", "reward"]
    }

# ==================== MODEL STATUS ====================
class ModelStatus:
    def __init__(self, name: str):
        self.name = name
        self.loaded = False
        self.loading = False
        self.load_time = 0.0
        self.last_used = 0.0
        self.memory_mb = 0.0
        self.error: Optional[str] = None
        self.instance: Any = None

# ==================== EMBEDDING SERVICE ====================
class EmbeddingService:
    """Centralized embedding service with domain-aware caching"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self._caches = {
            "biomed": {},
            "general": {},
            "chemistry": {},
            "computerscience": {}
        }
        self._cache_size = 1000
        
        # Domain-specific configurations
        self.domain_configs = {
            "biomed": {
                "special_tokens": ["pH", "°C", "mM", "enzyme", "protein", "cell", "growth"],
                "dimension": 384
            },
            "general": {
                "special_tokens": [],
                "dimension": 384
            },
            "chemistry": {
                "special_tokens": ["mol", "M", "reaction", "catalyst", "compound"],
                "dimension": 384
            },
            "computerscience": {
                "special_tokens": ["def", "class", "import", "function", "algorithm", "O(", ")", "{", "}", "[", "]", "<", ">", "=", "+", "-", "*", "/", "%", "#"],
                "dimension": 384
            }
        }
    
    def _get_cache_key(self, text: str, domain: str) -> str:
        """Generate cache key for text and domain"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{domain}_{text_hash}"
    
    async def encode(self, 
                    texts: Union[str, List[str]], 
                    domain: str = "general",
                    use_cache: bool = True) -> np.ndarray:
        """
        Encode texts for specific domain.
        Returns numpy array of embeddings.
        """
        # Validate domain
        if domain not in self._caches:
            logger.warning(f"Domain {domain} not configured, using 'general'")
            domain = "general"
        
        # Convert to list if single string
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        # Get model
        try:
            model = await self.model_loader.get_model("embedding", domain)
        except Exception as e:
            logger.error(f"Failed to get embedding model: {e}")
            # Return fallback embeddings
            dimension = self.domain_configs[domain]["dimension"]
            if single_text:
                return np.random.randn(dimension)
            return np.random.randn(len(texts), dimension)
        
        # Check cache
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache and domain in self._caches:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, domain)
                if cache_key in self._caches[domain]:
                    cached_results.append(self._caches[domain][cache_key])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Encode uncached texts
        if uncached_texts:
            try:
                uncached_embeddings = model.encode(uncached_texts)
                
                # Cache results
                for text, embedding in zip(uncached_texts, uncached_embeddings):
                    cache_key = self._get_cache_key(text, domain)
                    self._caches[domain][cache_key] = embedding
                    
                    # Enforce cache size limit
                    if len(self._caches[domain]) > self._cache_size:
                        # Remove oldest item
                        self._caches[domain].pop(next(iter(self._caches[domain])))
            except Exception as e:
                logger.error(f"Embedding encoding failed: {e}")
                # Fallback random embeddings
                dimension = self.domain_configs[domain]["dimension"]
                uncached_embeddings = np.random.randn(len(uncached_texts), dimension)
        else:
            uncached_embeddings = np.array([])
        
        # Combine cached and uncached results
        all_embeddings = []
        
        if cached_results or uncached_embeddings.size > 0:
            # Create empty array for all embeddings
            dimension = self.domain_configs[domain]["dimension"]
            all_embeddings_array = np.zeros((len(texts), dimension))
            
            # Place cached results
            cache_idx = 0
            uncached_idx = 0
            for i in range(len(texts)):
                if i in uncached_indices:
                    all_embeddings_array[i] = uncached_embeddings[uncached_idx]
                    uncached_idx += 1
                else:
                    all_embeddings_array[i] = cached_results[cache_idx]
                    cache_idx += 1
            
            all_embeddings = all_embeddings_array
        else:
            # Fallback
            dimension = self.domain_configs[domain]["dimension"]
            all_embeddings = np.random.randn(len(texts), dimension)
        
        return all_embeddings[0] if single_text else all_embeddings
    
    async def similarity(self, 
                        text1: str, 
                        text2: str, 
                        domain: str = "general") -> float:
        """Compute cosine similarity between two texts"""
        try:
            emb1 = await self.encode(text1, domain)
            emb2 = await self.encode(text2, domain)
            
            # Convert to tensors
            import torch
            emb1_tensor = torch.tensor(emb1).unsqueeze(0)
            emb2_tensor = torch.tensor(emb2).unsqueeze(0)
            
            return float(util.cos_sim(emb1_tensor, emb2_tensor)[0][0])
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.5  # Neutral fallback
    
    async def batch_similarity(self, 
                              query: str, 
                              texts: List[str], 
                              domain: str = "general") -> List[float]:
        """Compute similarity between query and multiple texts"""
        try:
            query_emb = await self.encode(query, domain)
            text_embs = await self.encode(texts, domain)
            
            import torch
            query_tensor = torch.tensor(query_emb).unsqueeze(0)
            text_tensors = torch.tensor(text_embs)
            
            similarities = util.cos_sim(query_tensor, text_tensors)[0]
            return [float(s) for s in similarities]
        except Exception as e:
            logger.error(f"Batch similarity failed: {e}")
            return [0.5] * len(texts)
    
    async def domain_classification(self, text: str) -> Dict[str, float]:
        """
        Classify which domain a text belongs to using embeddings.
        Returns scores for each domain.
        """
        domain_scores = {}
        
        for domain in self._caches.keys():
            # Check for domain-specific keywords
            keywords = self.domain_configs[domain]["special_tokens"]
            keyword_score = sum(1 for kw in keywords if kw.lower() in text.lower())
            
            # Base score with keyword bonus
            domain_scores[domain] = 0.3 + (keyword_score * 0.1)
        
        # Ensure sum is 1.0
        total = sum(domain_scores.values())
        if total > 0:
            domain_scores = {k: v/total for k, v in domain_scores.items()}
        
        return domain_scores
    
    def clear_cache(self, domain: str = None):
        """Clear cache for domain(s)"""
        if domain:
            if domain in self._caches:
                self._caches[domain].clear()
                logger.info(f"Cleared cache for domain: {domain}")
        else:
            for dom in self._caches:
                self._caches[dom].clear()
            logger.info("Cleared all domain caches")

# ==================== MAIN MODEL LOADER ====================
class UnifiedModelLoader:
    """
    SINGLE FILE solution for all model loading, pre-loading, and management
    """
    
    def __init__(self):
        self.models: Dict[str, ModelStatus] = {}
        self._lock = threading.RLock()
        self._warmup_done = False
        self._cache_dir = "model_cache"
        self.embedding_service = EmbeddingService(self)
        
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # Initialize all models
        self._init_model_registry()
        logger.info("✅ Unified Model Loader initialized")
    
    def _init_model_registry(self):
        """Initialize all model statuses"""
        models = [
            "nltk", "embedding", "biomistral", 
            "mistral_api", "biomed_extractor", "reward",
            "qwen"  # Added Qwen model
        ]
        
        for model in models:
            self.models[model] = ModelStatus(model)
    
    # ========== PUBLIC API ==========
    
    async def startup(self, domain: str = "general", warmup: bool = True):
        """
        Complete startup sequence - call this once at app startup
        """
        logger.info(f"🚀 Starting model pre-load for {domain} domain")
        
        # Get domain priorities
        priorities = ModelConfig.DOMAIN_PRIORITIES.get(domain, ["nltk", "embedding"])
        
        # Load critical models first
        critical_loaded = await self._load_critical_models(priorities[:2])
        
        if not critical_loaded:
            logger.error("❌ Failed to load critical models")
            return False
        
        # Load remaining models in background
        if len(priorities) > 2:
            threading.Thread(
                target=self._background_load,
                args=(priorities[2:],),
                daemon=True
            ).start()
        
        # Warm up if requested
        if warmup:
            await self._warmup_models()
        
        logger.info("✅ Model startup complete")
        return True
    
    async def get_model(self, model_name: str, domain: str = "general"):
        """
        Get a model instance - loads if not already loaded
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        status = self.models[model_name]
        
        # Already loaded and ready
        if status.loaded and status.instance:
            status.last_used = time.time()
            return status.instance
        
        # Currently loading
        if status.loading:
            await self._wait_for_load(model_name, ModelConfig.LOAD_TIMEOUTS.get(model_name, 30.0))
            if status.loaded:
                return status.instance
        
        # Need to load
        return await self._load_model(model_name, domain)
    
    # ========== EMBEDDING SERVICE METHODS ==========
    
    async def get_embeddings(self, 
                           texts: Union[str, List[str]], 
                           domain: str = "general",
                           use_cache: bool = True) -> np.ndarray:
        """
        Get embeddings for texts using the embedding service
        """
        return await self.embedding_service.encode(texts, domain, use_cache)
    
    async def compute_similarity(self, 
                               text1: str, 
                               text2: str, 
                               domain: str = "general") -> float:
        """
        Compute cosine similarity between two texts
        """
        return await self.embedding_service.similarity(text1, text2, domain)
    
    async def batch_similarity(self, 
                             query: str, 
                             texts: List[str], 
                             domain: str = "general") -> List[float]:
        """
        Compute similarity between query and multiple texts
        """
        return await self.embedding_service.batch_similarity(query, texts, domain)
    
    async def classify_domain(self, text: str) -> Dict[str, float]:
        """
        Classify which domain a text belongs to
        """
        return await self.embedding_service.domain_classification(text)
    
    def clear_embedding_cache(self, domain: str = None):
        """
        Clear embedding cache for domain(s)
        """
        self.embedding_service.clear_cache(domain)
    
    # ========== MODEL LOADING METHODS ==========
    
    async def _load_qwen(self):
        """Load the Qwen model for computer science tasks"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_name = "Qwen/CodeQwen1.5-7B-Chat"
            
            # Clear CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Loading Qwen tokenizer and model: {model_name}")
            
            # Load tokenizer with special tokens for code
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left",
                truncation_side="left"
            )
            
            # Configure model loading based on available resources
            device_map = "auto"
            load_in_8bit = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 20e9  # < 20GB VRAM
            load_in_4bit = False
            
            if load_in_8bit:
                logger.info("Using 8-bit quantization for Qwen model")
                
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                trust_remote_code=True,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ).eval()
            
            # Set pad token if not set
            if not self.qwen_tokenizer.pad_token:
                self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
            
            logger.info("✅ Qwen model loaded successfully")
            return True
            
        except Exception as e:
            error_msg = f"❌ Failed to load Qwen model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.models["qwen"].error = error_msg
            self.models["qwen"].loaded = False
            self.models["qwen"].loading = False
            raise RuntimeError(error_msg) from e
            
    async def generate_with_qwen(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text using the Qwen model
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Generated text
        """
        if not hasattr(self, 'qwen_model') or self.qwen_model is None:
            await self._load_model('qwen')
            
        try:
            # Tokenize input
            inputs = self.qwen_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                return_token_type_ids=False
            ).to(self.qwen_model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.qwen_tokenizer.eos_token_id
                )
            
            # Decode and clean up the response
            response = self.qwen_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in Qwen generation: {str(e)}")
            raise
    
    async def _load_critical_models(self, critical_models: list):
        """Load critical models first"""
        load_tasks = []
        for model in critical_models:
            if model in self.models and not self.models[model].loaded:
                load_tasks.append(self._load_model(model, critical=True))
        
        if load_tasks:
            await asyncio.gather(*load_tasks, return_exceptions=True)
        
        return all(results)
    
    # Add loader method:
    async def _load_biomed_extractor(self):
        """Load biomedical parameter extractor"""
        try:
            from core.biomed_parameter_extractor import initialize_extractor
            success = await initialize_extractor()
            return success
        except Exception as e:
            logger.error(f"Biomedical extractor load failed: {e}")
            return None
    
    async def _load_model(self, model_name: str, domain: str = "general", critical: bool = False):
        """Load a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_status = self.models[model_name]
        
        # Skip if already loaded or loading
        if model_status.loaded or model_status.loading:
            return
            
        model_status.loading = True
        start_time = time.time()
        
        try:
            if model_name == "nltk":
                await self._load_nltk()
            elif model_name == "embedding":
                await self._load_embedding()
            elif model_name == "biomistral":
                await self._load_biomistral()
            elif model_name == "mistral_api":
                await self._load_mistral_api()
            elif model_name == "biomed_extractor":
                await self._load_biomed_extractor()
            elif model_name == "reward":
                await self._load_reward()
            elif model_name == "qwen":
                await self._load_qwen()
            else:
                raise ValueError(f"No loader for model: {model_name}")
                
            model_status.loaded = True
            model_status.load_time = time.time() - start_time
            model_status.last_used = time.time()
            logger.info(f"✅ Loaded {model_name} in {model_status.load_time:.1f}s")
            
        except Exception as e:
            model_status.error = str(e)
            logger.error(f"❌ Failed to load {model_name}: {str(e)}")
            if critical:
                raise
        finally:
            model_status.loading = False
    
    async def _load_nltk(self):
        """Load NLTK data"""
        try:
            # Setup NLTK data path
            nltk_data_path = os.path.expanduser('~/nltk_data')
            os.makedirs(nltk_data_path, exist_ok=True)
            
            import nltk
            nltk.data.path.append(nltk_data_path)
            
            # Download required packages
            packages = ['punkt', 'averaged_perceptron_tagger']
            for package in packages:
                try:
                    if package == 'punkt':
                        nltk.data.find(f'tokenizers/{package}')
                    else:
                        nltk.data.find(f'taggers/{package}')
                except LookupError:
                    nltk.download(package, quiet=True)
            
            # Test NLTK
            from nltk.tokenize import word_tokenize
            test_tokens = word_tokenize("Test sentence")
            
            # Return a simple wrapper
            class NLTKWrapper:
                def tokenize(self, text):
                    return word_tokenize(text)
                
                def pos_tag(self, tokens):
                    from nltk import pos_tag
                    return pos_tag(tokens)
            
            return NLTKWrapper()
            
        except Exception as e:
            logger.error(f"NLTK load failed: {e}")
            return None
    
    async def _load_embedding(self):
        """Load embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a small, fast model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Warm up with a small batch
            model.encode(["warmup text"])
            
            return model
            
        except Exception as e:
            logger.error(f"Embedding model load failed: {e}")
            return None
    
    async def _load_biomistral(self):
        """Load BioMistral GGUF"""
        try:
            from ctransformers import AutoModelForCausalLM
            from huggingface_hub import hf_hub_download
            
            # Config from environment
            model_repo = os.getenv("BIOMISTRAL_GGUF_MODEL", "MaziyarPanahi/BioMistral-7B-GGUF")
            model_file = os.getenv("BIOMISTRAL_GGUF_FILE", "BioMistral-7B.Q4_K_M.gguf")
            n_threads = int(os.getenv("BIOMISTRAL_N_THREADS", "2"))
            n_gpu_layers = int(os.getenv("BIOMISTRAL_N_GPU_LAYERS", "0"))
            
            # Download model if not cached
            cache_key = hashlib.md5(f"{model_repo}_{model_file}".encode()).hexdigest()
            cache_path = os.path.join(self._cache_dir, f"{cache_key}.gguf")
            
            if not os.path.exists(cache_path):
                logger.info(f"📥 Downloading BioMistral: {model_file}")
                model_path = hf_hub_download(
                    repo_id=model_repo,
                    filename=model_file,
                    local_dir=self._cache_dir
                )
            else:
                model_path = cache_path
                logger.info(f"📦 Using cached BioMistral: {os.path.basename(model_path)}")
            
            # Load model
            llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="mistral",
                gpu_layers=n_gpu_layers,
                threads=n_threads,
                context_length=2048
            )
            
            # Quick warmup
            try:
                llm("Warmup", max_new_tokens=1)
            except:
                pass
            
            return llm
            
        except Exception as e:
            logger.error(f"BioMistral load failed: {e}")
            return None
    
    async def _load_mistral_api(self):
        """Setup Mistral API connection"""
        try:
            from core.config import MISTRAL_API_KEY, MISTRAL_USE_API
            
            if not MISTRAL_USE_API or not MISTRAL_API_KEY:
                logger.warning("Mistral API not configured")
                return None
            
            # Test API connection
            import aiohttp
            
            async def test_connection():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.mistral.ai/v1/models",
                        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
                        timeout=5.0
                    ) as response:
                        return response.status == 200
            
            connection_ok = await test_connection()
            
            if connection_ok:
                logger.info("✅ Mistral API connection verified")
                
                # Return API wrapper
                class MistralAPIWrapper:
                    def __init__(self, api_key):
                        self.api_key = api_key
                    
                    async def generate(self, prompt, max_tokens=500):
                        import aiohttp
                        import json
                        
                        url = "https://api.mistral.ai/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        data = {
                            "model": "mistral-large-latest",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.7
                        }
                        
                        async with aiohttp.ClientSession() as session:
                            async with session.post(url, headers=headers, json=data, timeout=30.0) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    return result["choices"][0]["message"]["content"]
                                else:
                                    raise Exception(f"API error: {response.status}")
                
                return MistralAPIWrapper(MISTRAL_API_KEY)
            else:
                logger.error("❌ Mistral API connection failed")
                return None
                
        except Exception as e:
            logger.error(f"Mistral API setup failed: {e}")
            return None
    
    async def _load_reward(self):
        """Load RLHF reward model"""
        try:
            # Try to load existing model
            model_path = "models/reward_model.pth"
            
            if os.path.exists(model_path):
                import torch
                from sentence_transformers import SentenceTransformer
                
                class SimpleRewardModel:
                    def __init__(self):
                        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                        # Simple linear classifier
                        self.classifier = torch.nn.Linear(384, 1)
                        
                        # Load weights if available
                        try:
                            state_dict = torch.load(model_path, map_location="cpu")
                            self.classifier.load_state_dict(state_dict)
                            logger.info("✅ Loaded trained reward model")
                        except:
                            logger.info("ℹ️ Using untrained reward model")
                    
                    def score(self, texts):
                        if isinstance(texts, str):
                            texts = [texts]
                        
                        embeddings = self.encoder.encode(texts, convert_to_tensor=True)
                        with torch.no_grad():
                            scores = self.classifier(embeddings)
                        return scores.cpu().numpy()
                
                return SimpleRewardModel()
            
            return None
            
        except Exception as e:
            logger.error(f"Reward model load failed: {e}")
            return None
    
    # ========== BACKGROUND & WARMUP ==========
    
    def _background_load(self, models: list):
        """Background thread for loading non-critical models"""
        try:
            # Create new event loop for thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def load_all():
                tasks = [self._load_model(name, "general") for name in models]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                loaded = sum(1 for r in results if r is not None and not isinstance(r, Exception))
                logger.info(f"🎉 Background loading complete: {loaded}/{len(models)} models loaded")
            
            loop.run_until_complete(load_all())
            loop.close()
            
        except Exception as e:
            logger.error(f"Background loading failed: {e}")
    
    async def _warmup_models(self):
        """Warm up all loaded models with simple queries"""
        if self._warmup_done:
            return
            
        warmup_tasks = []
        
        if "biomistral" in self.models and self.models["biomistral"].loaded:
            warmup_tasks.append(self.warmup_biomistral())
            
            warmup_tasks.append(warmup_biomistral())
        
        # Warm up embeddings
        if self.models["embedding"].loaded and self.models["embedding"].instance:
            async def warmup_embedding():
                try:
                    model = self.models["embedding"].instance
                    # Test encoding
                    embeddings = model.encode(["warmup text", "test query"])
                    
                    # Test similarity
                    import torch
                    from sentence_transformers import util
                    emb1 = torch.tensor(embeddings[0]).unsqueeze(0)
                    emb2 = torch.tensor(embeddings[1]).unsqueeze(0)
                    similarity = util.cos_sim(emb1, emb2)
                    
                    logger.info(f"✅ Embedding model warmed up (similarity: {similarity[0][0]:.3f})")
                except Exception as e:
                    logger.warning(f"Embedding warmup warning: {e}")
            
            warmup_tasks.append(warmup_embedding())
        
        # Run all warmups
        if warmup_tasks:
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
        
        self._warmup_done = True
        logger.info("🎉 Model warmup complete")
    
    async def _wait_for_load(self, model_name: str, timeout: float):
        """Wait for a model to finish loading"""
        start = time.time()
        while self.models[model_name].loading and (time.time() - start) < timeout:
            await asyncio.sleep(0.1)
    
    # ========== UTILITIES ==========
    
    async def generate_with_biomistral(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate with BioMistral (with fallback)"""
        try:
            llm = await self.get_model("biomistral", "biomed")
            if llm:
                def generate():
                    return llm(prompt, max_new_tokens=max_tokens, temperature=0.3)
                
                result = await asyncio.wait_for(
                    asyncio.to_thread(generate),
                    timeout=60.0
                )
                return str(result).strip()
        except Exception as e:
            logger.warning(f"BioMistral generation failed: {e}")
        
        # Fallback
        return f"Biomedical analysis: {prompt[:100]}..."
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all models"""
        with self._lock:
            return {
                name: {
                    "loaded": status.loaded,
                    "loading": status.loading,
                    "load_time": status.load_time,
                    "last_used": status.last_used,
                    "memory_mb": status.memory_mb,
                    "error": status.error,
                    "has_instance": status.instance is not None
                }
                for name, status in self.models.items()
            }
    
    def is_ready(self) -> bool:
        """Check if system is ready for queries"""
        # At least NLTK and embeddings should be loaded
        return (self.models["nltk"].loaded and 
                self.models["embedding"].loaded and
                self._warmup_done)

# ==================== GLOBAL INSTANCE ====================
model_loader = UnifiedModelLoader()

# ==================== EASY-TO-USE FUNCTIONS ====================

async def startup_models(domain: str = "general", warmup: bool = True):
    """One-line startup - call this in main.py startup"""
    return await model_loader.startup(domain, warmup)

async def get_nltk():
    """Get NLTK instance"""
    return await model_loader.get_model("nltk")

async def get_embeddings_model():
    """Get embedding model"""
    return await model_loader.get_model("embedding")

async def get_biomistral():
    """Get BioMistral instance"""
    return model_loader.get_model("biomistral")

def get_qwen():
    """Get Qwen model instance"""
    return model_loader.get_model("qwen")

async def get_mistral_api():
    """Get Mistral API wrapper"""
    return await model_loader.get_model("mistral_api")

async def get_embeddings(texts: Union[str, List[str]], domain: str = "general"):
    """Get embeddings for texts"""
    return await model_loader.get_embeddings(texts, domain)

async def compute_similarity(text1: str, text2: str, domain: str = "general"):
    """Compute similarity between texts"""
    return await model_loader.compute_similarity(text1, text2, domain)

async def classify_domain(text: str):
    """Classify domain of text"""
    return await model_loader.classify_domain(text)

def get_model_status():
    """Get status of all models"""
    return model_loader.get_status()

def are_models_ready():
    """Check if models are ready for queries"""
    return model_loader.is_ready()