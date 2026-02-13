# core/rlhf/reward_model.py - FIXED VERSION

import torch
import torch.nn as nn
import os
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("core.rlhf.reward_model")

class RewardModel(nn.Module):
    def __init__(self, embedding_dim=384):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Load encoder immediately
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded sentence transformer for reward model")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            # Fallback: create a dummy encoder
            self.encoder = None
        
        self.classifier = nn.Linear(embedding_dim, 1)
        self._model_loaded = False
        
        # Initialize with small weights
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, texts):
        if self.encoder is None:
            # Fallback random embeddings
            if isinstance(texts, str):
                texts = [texts]
            embeddings = torch.randn(len(texts), self.embedding_dim)
        else:
            # Get embeddings
            embeddings = self.encoder.encode(
                texts, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        # Ensure embeddings are correct shape
        if embeddings.shape[-1] != self.embedding_dim:
            # Reshape or project if needed
            if embeddings.shape[-1] > self.embedding_dim:
                embeddings = embeddings[:, :self.embedding_dim]
            else:
                # Pad if smaller
                pad_size = self.embedding_dim - embeddings.shape[-1]
                embeddings = torch.nn.functional.pad(embeddings, (0, pad_size))
        
        return self.classifier(embeddings)
    
    def is_trained(self):
        """Check if model has been trained"""
        if not self._model_loaded:
            return False
        
        # Check if weights have changed from initialization
        with torch.no_grad():
            weight_norm = self.classifier.weight.norm().item()
            # Random initialization norm ~ sqrt(384)*0.02 ≈ 0.4
            return weight_norm > 0.5  # If significantly different

# Global singleton
_reward_model = None
_model_path = "models/reward_model.pth"

def get_reward_model():
    global _reward_model
    if _reward_model is None:
        _reward_model = RewardModel()
        
        # Try to load trained model
        if os.path.exists(_model_path):
            try:
                state_dict = torch.load(_model_path, map_location="cpu")
                _reward_model.load_state_dict(state_dict)
                _reward_model._model_loaded = True
                logger.info("✅ Loaded trained RLHF reward model")
            except Exception as e:
                logger.warning(f"Failed to load reward model: {e} → using fresh model")
                _reward_model._model_loaded = False
        else:
            logger.info("ℹ️ No trained reward model found. Using fresh model (will learn from feedback)")
            _reward_model._model_loaded = False
    
    return _reward_model

def save_reward_model():
    """Save the current reward model"""
    global _reward_model, _model_path
    if _reward_model:
        os.makedirs("models", exist_ok=True)
        torch.save(_reward_model.state_dict(), _model_path)
        _reward_model._model_loaded = True
        logger.info(f"✅ Saved reward model to {_model_path}")
        return True
    return False

def initialize_reward_model():
    """Initialize and save a fresh reward model if none exists"""
    if not os.path.exists(_model_path):
        model = get_reward_model()
        save_reward_model()
        logger.info("🆕 Created and saved fresh reward model")
        return True
    return False