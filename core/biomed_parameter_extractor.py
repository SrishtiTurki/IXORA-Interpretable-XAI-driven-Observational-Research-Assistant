# core/biomed_parameter_extractor.py - SIMPLIFIED & MODERN VERSION
# just a thin wrapper around the unified domain-aware LLM extractor

import logging
import hashlib
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger("core.biomed_parameter_extractor")

class BiomedicalParameterExtractor:
    """Lightweight wrapper for biomedical parameter extraction using unified LLM extractor"""
    
    def __init__(self):
        self.extraction_cache = {}
    
    async def extract_parameters(self, query: str) -> Dict[str, Any]:
        """
        Modern biomedical parameter extraction using domain-specific LLM (BioMistral) first.
        Handles real user input like "around 37 degrees", "ph about 7.4", "shake at 150 rpm".
        """
        logger.info(f"Extracting biomed parameters: {query[:80]}...")
        
        # Cache check (keep this excellent cache!)
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.extraction_cache:
            logger.info("Cache hit")
            return self.extraction_cache[cache_key]
        
        try:
            # Use the unified extractor with biomed domain
            from core.parameter_extractor import extract_parameters as unified_extract
            result = await unified_extract(query, domain="biomed")
            
            # Cache result if meaningful
            if result.get("parameters") or "_metadata" in result:
                self.extraction_cache[cache_key] = result
                if len(self.extraction_cache) > 100:
                    self.extraction_cache.pop(next(iter(self.extraction_cache)))
            
            param_count = len(result.get("parameters", {}))
            method = result.get("_metadata", {}).get("method", "unknown")
            logger.info(f"Biomed extraction done: {param_count} params via {method}")
            return result
            
        except Exception as e:
            logger.error(f"Unified extractor failed: {e}")
            # Final empty fallback
            fallback = {
                "parameters": {},
                "_metadata": {
                    "method": "failed",
                    "fallback_used": True,
                    "error": str(e)[:100],
                    "timestamp": datetime.now().isoformat()
                }
            }
            self.extraction_cache[cache_key] = fallback
            return fallback
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Simple stats"""
        return {
            "cache_size": len(self.extraction_cache),
            "status": "active_llm_extractor"
        }

# Global instance
biomed_extractor = BiomedicalParameterExtractor()

# Helper functions
async def extract_biomedical_parameters(query: str) -> Dict[str, Any]:
    """Main entry point"""
    return await biomed_extractor.extract_parameters(query)

async def initialize_extractor():
    """No heavy SciSpaCy loading needed anymore"""
    return True