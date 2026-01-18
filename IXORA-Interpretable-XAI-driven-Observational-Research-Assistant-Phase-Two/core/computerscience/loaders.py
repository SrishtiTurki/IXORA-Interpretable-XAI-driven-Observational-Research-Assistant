# core/computerscience/loaders.py - Computer Science Qwen Model Integration with RLHF

import asyncio
import logging
import torch
from typing import Optional, Dict, Any
import os
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from core.config import (
    CODEQWEN_MODEL,
    CODEQWEN_MAX_TOKENS,
    CODEQWEN_TEMPERATURE,
    CODEQWEN_TOP_P,
    CODEQWEN_DEVICE,
    CODEQWEN_LOAD_IN_8BIT,
    CODEQWEN_LOAD_IN_4BIT,
    CODEQWEN_TRUST_REMOTE_CODE,
    REWARD_MODEL_NAME,
    REWARD_MODEL_DEVICE,
    CACHE_DIR
)
from core.rlhf.reward_model import RewardModel
from .state import CSDomain, AnalysisType

logger = logging.getLogger("cs.loaders")

# Global cache
qwen_model: Optional[AutoModelForCausalLM] = None
qwen_tokenizer: Optional[AutoTokenizer] = None
reward_model: Optional[RewardModel] = None
load_lock = asyncio.Lock()

# Model configuration based on analysis type
MODEL_CONFIGS = {
    AnalysisType.THEORETICAL: {
        "temperature": 0.3,  # More focused for theoretical accuracy
        "top_p": 0.9,
        "max_length": 1024,
        "stop_sequences": ["\n\n"]
    },
    AnalysisType.PRACTICAL: {
        "temperature": 0.5,  # More creative for code generation
        "top_p": 0.95,
        "max_length": 2048,  # Longer for code
        "stop_sequences": ["\n```", "\n\n"]
    },
    AnalysisType.HYBRID: {
        "temperature": 0.4,
        "top_p": 0.92,
        "max_length": 1536,
        "stop_sequences": ["\n\n"]
    }
}

class CodeStopCriteria(StoppingCriteria):
    """Custom stopping criteria for code generation"""
    def __init__(self, stop_sequences, tokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        self.max_stop_len = max(len(seq) for seq in stop_sequences)
    
    def __call__(self, input_ids, scores, **kwargs):
        # Convert current sequence to text
        current_text = self.tokenizer.decode(input_ids[0])
        
        # Check for any stop sequence
        for seq in self.stop_sequences:
            if seq in current_text:
                return True
        return False

# Model specific configuration
MODEL_CONFIG = {
    "trust_remote_code": CODEQWEN_TRUST_REMOTE_CODE,
    "device_map": "auto" if torch.cuda.is_available() else None,
    "cache_dir": str(Path(CACHE_DIR) / "models"),
    "revision": "main"
}

if CODEQWEN_LOAD_IN_4BIT:
    MODEL_CONFIG["load_in_4bit"] = True
    MODEL_CONFIG["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
elif CODEQWEN_LOAD_IN_8BIT:
    MODEL_CONFIG["load_in_8bit"] = True

def get_model_config(analysis_type: Union[AnalysisType, str] = AnalysisType.HYBRID) -> Dict[str, Any]:
    """Get model configuration based on analysis type"""
    if isinstance(analysis_type, str):
        analysis_type = AnalysisType(analysis_type.lower())
    return MODEL_CONFIGS.get(analysis_type, MODEL_CONFIGS[AnalysisType.HYBRID])

async def _load_qwen_model(analysis_type: Union[AnalysisType, str] = AnalysisType.HYBRID) -> Dict[str, Any]:
    """
    Load Qwen model and tokenizer with configuration based on analysis type
    
    Args:
        analysis_type: Type of analysis (theoretical, practical, hybrid)
    """
    global qwen_model, qwen_tokenizer
    
    async with load_lock:
        if qwen_model is not None and qwen_tokenizer is not None:
            return {
                "model": qwen_model,
                "tokenizer": qwen_tokenizer,
                **get_model_config(analysis_type)
            }
            
        logger.info(f"Loading Qwen model for {analysis_type} computer science tasks...")
        
        try:
            # Get model config
            config = get_model_config(analysis_type)
            
            # Configure device and precision
            device = torch.device(CODEQWEN_DEVICE if torch.cuda.is_available() else "cpu")
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            # Configure quantization if needed
            quantization_config = None
            if CODEQWEN_LOAD_IN_4BIT:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif CODEQWEN_LOAD_IN_8BIT:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load tokenizer with special tokens for code and theory
            qwen_tokenizer = AutoTokenizer.from_pretrained(
                CODEQWEN_MODEL,
                trust_remote_code=CODEQWEN_TRUST_REMOTE_CODE,
                cache_dir=str(Path(CACHE_DIR) / "tokenizers"),
                padding_side="left"
            )
            
            # Configure special tokens
            if qwen_tokenizer.pad_token is None:
                qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
            
            # Add domain-specific tokens if needed
            additional_special_tokens = [
                "<theorem>", "</theorem>",
                "<proof>", "</proof>",
                "<code>", "</code>"
            ]
            qwen_tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
            
            # Load model with appropriate configuration
            qwen_model = AutoModelForCausalLM.from_pretrained(
                CODEQWEN_MODEL,
                trust_remote_code=CODEQWEN_TRUST_REMOTE_CODE,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                quantization_config=quantization_config,
                cache_dir=str(Path(CACHE_DIR) / "models")
            )
            
            # Resize token embeddings if we added new tokens
            qwen_model.resize_token_embeddings(len(qwen_tokenizer))
            
            # Set model to evaluation mode
            qwen_model.eval()
            
            # Configure stopping criteria
            stop_criteria = CodeStopCriteria(
                stop_sequences=config["stop_sequences"],
                tokenizer=qwen_tokenizer
            )
            
            logger.info(f"Qwen model loaded successfully for {analysis_type} tasks")
            
            return {
                "model": qwen_model,
                "tokenizer": qwen_tokenizer,
                "stop_criteria": stop_criteria,
                **config
            }
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {str(e)}")
            raise RuntimeError(f"Failed to load Qwen model: {str(e)}")
    
    if qwen_model is not None and qwen_tokenizer is not None:
        logger.info("Qwen model and tokenizer already loaded (cached)")
        return {"model": qwen_model, "tokenizer": qwen_tokenizer}

    async with load_lock:
        if qwen_model is not None and qwen_tokenizer is not None:
            return {"model": qwen_model, "tokenizer": qwen_tokenizer}

        try:
            logger.info(f"Loading Qwen model: {CODEQWEN_MODEL}")
            
            # Load tokenizer and model with appropriate device settings
            qwen_tokenizer = AutoTokenizer.from_pretrained(
                CODEQWEN_MODEL,
                trust_remote_code=CODEQWEN_TRUST_REMOTE_CODE
            )
            
            qwen_model = AutoModelForCausalLM.from_pretrained(
                CODEQWEN_MODEL,
                device_map="auto",
                trust_remote_code=CODEQWEN_TRUST_REMOTE_CODE,
                load_in_8bit=CODEQWEN_LOAD_IN_8BIT,
                load_in_4bit=CODEQWEN_LOAD_IN_4BIT
            ).eval()
            
            logger.info("✅ Qwen model and tokenizer loaded successfully")
            return {"model": qwen_model, "tokenizer": qwen_tokenizer}
            
        except Exception as e:
            logger.error(f"❌ Qwen model load failed: {e}")
            raise RuntimeError("Qwen model failed to load")

async def _load_reward_model() -> RewardModel:
    """Load the RLHF reward model for computer science domain"""
    global reward_model
    
    if reward_model is not None:
        logger.info("Reward model already loaded (cached)")
        return reward_model

    try:
        logger.info(f"Loading RLHF reward model: {REWARD_MODEL_NAME}")
        reward_model = RewardModel(
            model_name=REWARD_MODEL_NAME,
            device=REWARD_MODEL_DEVICE,
            domain="computerscience"  # Specify domain for domain-specific rewards
        )
        logger.info("✅ Reward model loaded successfully")
        return reward_model
        
    except Exception as e:
        logger.error(f"❌ Reward model load failed: {e}")
        raise RuntimeError("Failed to load reward model")

async def generate_codeqwen_response(user_input: str, max_tokens: int = None) -> str:
    """Generate a response using the Qwen model for computer science queries"""
    if max_tokens is None:
        max_tokens = CODEQWEN_MAX_TOKENS
        
    try:
        # Load model and tokenizer
        models = await _load_qwen_model()
        model = models["model"]
        tokenizer = models["tokenizer"]
        
        # Create computer science specific prompt
        prompt = f"""You are an expert computer science assistant. Please provide a clear, technical response to the following query.
        Focus on accuracy, efficiency, and best practices in computer science.
        
        Query: {user_input}
        
        Response:"""
        
        # Tokenize and generate response
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            return_token_type_ids=False
        ).to(CODEQWEN_DEVICE)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=CODEQWEN_TEMPERATURE,
                top_p=CODEQWEN_TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and clean up the response
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        logger.info(f"Generated response with {len(response)} characters")
        return response
        
    except Exception as e:
        logger.error(f"Error in Qwen generation: {e}")
        return "I encountered an error while processing your computer science query. Please try again."

def _fallback_cs_response(user_input: str) -> str:
    """Fallback response when Qwen fails"""
    return "I'm having trouble processing your request. Please try again later or rephrase your question."

# Public function to get the reward model
async def get_reward_model() -> RewardModel:
    """Get or load the RLHF reward model"""
    if reward_model is None:
        await _load_reward_model()
    return reward_model
