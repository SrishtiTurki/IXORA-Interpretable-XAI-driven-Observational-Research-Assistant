# core/rlhf/reward_model.py - UPDATED VERSION

import torch
import torch.nn as nn
import os
import logging

logger = logging.getLogger("core.rlhf.reward_model")

class RewardModel(nn.Module):
    def __init__(self, embedding_dim=384):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = None  # Will be loaded lazily
        self.classifier = nn.Linear(embedding_dim, 1)
        self._model_loaded = False
    
    def _ensure_encoder(self):
        """Lazy load the encoder when needed"""
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Loaded sentence transformer for reward model")
            except ImportError as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                self.encoder = None
    
    def forward(self, texts):
        # If encoder is not loaded yet, use fallback
        if self.encoder is None:
            self._ensure_encoder()
        
        if self.encoder is None:
            # Fallback random embeddings
            if isinstance(texts, str):
                texts = [texts]
            embeddings = torch.randn(len(texts), self.embedding_dim)
        else:
            embeddings = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        
        return self.classifier(embeddings)
    
    def is_trained(self):
        """Check if model has been trained (has non-random weights)"""
        if not self._model_loaded:
            return False
        
        # Check if weights are different from initialization
        with torch.no_grad():
            # Sum of absolute weights as a simple metric
            weight_sum = self.classifier.weight.abs().sum().item()
            # Random initialization typically around sqrt(1/384) ≈ 0.05 per weight
            expected_random = 0.05 * self.embedding_dim  # ~19.2
            return abs(weight_sum - expected_random) > 5.0  # If significantly different

# Global singleton
_reward_model = None

def get_reward_model():
    global _reward_model
    if _reward_model is None:
        _reward_model = RewardModel()
        
        # Try to load trained model
        model_path = "models/reward_model.pth"
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location="cpu")
                _reward_model.load_state_dict(state_dict)
                _reward_model._model_loaded = True
                logger.info("✅ Loaded trained RLHF reward model")
            except Exception as e:
                logger.warning(f"Failed to load reward model: {e} → using fresh model")
                _reward_model._model_loaded = False
        else:
            logger.info("ℹ️ No trained reward model found. Using untrained model (will learn from feedback)")
            _reward_model._model_loaded = False
    
    return _reward_model

def save_reward_model():
    """Save the current reward model"""
    if _reward_model and _reward_model._model_loaded:
        os.makedirs("models", exist_ok=True)
        model_path = "models/reward_model.pth"
        torch.save(_reward_model.state_dict(), model_path)
        logger.info(f"✅ Saved reward model to {model_path}")
        return True
    return False