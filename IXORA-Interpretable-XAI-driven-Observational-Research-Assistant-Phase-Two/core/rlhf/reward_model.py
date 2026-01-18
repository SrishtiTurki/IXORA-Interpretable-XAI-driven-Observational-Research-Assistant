# core/rlhf/reward_model.py - DOMAIN-AWARE REWARD MODEL

import torch
import torch.nn as nn
import os
import logging
from typing import Dict, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger("core.rlhf.reward_model")

# Domain-specific configurations
DOMAIN_CONFIGS = {
    "biomed": {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "embedding_dim": 768,
        "hidden_dim": 256,
        "dropout": 0.1
    },
    "computerscience": {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "embedding_dim": 768,
        "hidden_dim": 512,  # Larger for code understanding
        "dropout": 0.2
    },
    "default": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "hidden_dim": 128,
        "dropout": 0.1
    }
}

@dataclass
class DomainConfig:
    """Configuration for a domain-specific reward model"""
    name: str
    embedding_model: str
    embedding_dim: int
    hidden_dim: int
    dropout: float = 0.1
    
    @classmethod
    def from_domain(cls, domain: str) -> 'DomainConfig':
        """Get config for a specific domain"""
        config = DOMAIN_CONFIGS.get(domain.lower(), DOMAIN_CONFIGS["default"])
        return cls(
            name=domain.lower(),
            embedding_model=config["embedding_model"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config.get("dropout", 0.1)
        )

class RewardModel(nn.Module):
    def __init__(self, domain: str = "default"):
        super().__init__()
        self.domain = domain.lower()
        self.config = DomainConfig.from_domain(self.domain)
        
        # Initialize model components
        self.encoder = None  # Lazy loading
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.embedding_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1)
        )
        self._model_loaded = False
        
        logger.info(f"Initialized {self.domain} reward model with config: {self.config}")
    
    def _ensure_encoder(self):
        """Lazy load the encoder when needed"""
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.config.embedding_model)
                logger.info(f"✅ Loaded {self.domain} encoder: {self.config.embedding_model}")
            except ImportError as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                self.encoder = None
    
    def forward(self, texts: Union[str, list]):
        # Ensure encoder is loaded
        if self.encoder is None:
            self._ensure_encoder()
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Get embeddings
        if self.encoder is None:
            # Fallback to random embeddings
            embeddings = torch.randn(len(texts), self.config.embedding_dim)
            if torch.cuda.is_available():
                embeddings = embeddings.cuda()
        else:
            with torch.no_grad():
                embeddings = self.encoder.encode(
                    texts, 
                    convert_to_tensor=True, 
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
        
        # Apply classifier
        logits = self.classifier(self.dropout(embeddings))
        return logits.squeeze(-1)
    
    def is_trained(self) -> bool:
        """Check if model has been trained"""
        if not self._model_loaded:
            return False
            
        # Check if weights are significantly different from initialization
        with torch.no_grad():
            # Check classifier weights
            for name, param in self.classifier.named_parameters():
                if 'weight' in name:
                    weight_sum = param.abs().sum().item()
                    # Simple check for non-random weights
                    if weight_sum < 1e-6:  # All zeros
                        return False
        return True

# Domain-specific model cache
_domain_models: Dict[str, RewardModel] = {}

def get_reward_model(domain: str = "default") -> RewardModel:
    """Get or create a reward model for the specified domain"""
    domain = domain.lower()
    
    # Get or create model for domain
    if domain not in _domain_models:
        _domain_models[domain] = RewardModel(domain)
        
        # Try to load pre-trained weights if they exist
        model_path = f"models/reward_model_{domain}.pth"
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location="cpu")
                _domain_models[domain].load_state_dict(state_dict)
                _domain_models[domain]._model_loaded = True
                logger.info(f"✅ Loaded trained {domain} reward model")
            except Exception as e:
                logger.warning(f"Failed to load {domain} model: {e} → using fresh model")
                _domain_models[domain]._model_loaded = False
    
    return _domain_models[domain]

def save_reward_model(domain: str = "default") -> bool:
    """Save the reward model for a specific domain"""
    domain = domain.lower()
    if domain in _domain_models and _domain_models[domain]._model_loaded:
        os.makedirs("models", exist_ok=True)
        model_path = f"models/reward_model_{domain}.pth"
        torch.save(_domain_models[domain].state_dict(), model_path)
        logger.info(f"✅ Saved {domain} reward model to {model_path}")
        return True
    return False

def save_all_models() -> Dict[str, bool]:
    """Save all domain models"""
    results = {}
    for domain in _domain_models:
        results[domain] = save_reward_model(domain)
    return results