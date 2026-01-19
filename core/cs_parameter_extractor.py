# core/cs_parameter_extractor.py - COMPREHENSIVE CS PARAMETER EXTRACTION PIPELINE

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
import json
import asyncio
from datetime import datetime
import hashlib
from core.config import CS_PARAMETER_PATTERNS, CS_ONTOLOGY

logger = logging.getLogger("core.cs_parameter_extractor")

class ComputerScienceParameterExtractor:
    """Comprehensive computer science parameter extraction pipeline"""
    
    def __init__(self):
        self._initialized = False
        self._loading = False
        
        # CS parameter patterns (from config)
        self.parameter_patterns = CS_PARAMETER_PATTERNS
        
        # CS ontology mapping (from config)
        self.cs_ontology = CS_ONTOLOGY
        
        # Extended CS parameter patterns
        self.extended_patterns = {
            'learning_rate': [
                (r'learning rate[:\s]+([\d\.]+e?-?\d*)', 'single'),
                (r'lr[:\s]+([\d\.]+e?-?\d*)', 'single'),
                (r'learning_rate[:\s]+([\d\.]+e?-?\d*)', 'single'),
                (r'(\d+\.?\d*e?-?\d*)\s*learning rate', 'single'),
            ],
            'batch_size': [
                (r'batch size[:\s]+(\d+)', 'single', 'samples'),
                (r'batch_size[:\s]+(\d+)', 'single', 'samples'),
                (r'batch[:\s]+(\d+)', 'single', 'samples'),
                (r'(\d+)\s*samples per batch', 'single', 'samples'),
            ],
            'epochs': [
                (r'(\d+)\s*epochs?', 'single'),
                (r'epochs?[:\s]+(\d+)', 'single'),
                (r'train for\s*(\d+)\s*epochs?', 'single'),
            ],
            'hidden_units': [
                (r'hidden units?[:\s]+(\d+)', 'single'),
                (r'hidden_size[:\s]+(\d+)', 'single'),
                (r'(\d+)\s*hidden units?', 'single'),
            ],
            'dropout': [
                (r'dropout[:\s]+([\d\.]+)', 'single'),
                (r'dropout rate[:\s]+([\d\.]+)', 'single'),
            ],
            'optimizer': [
                (r'optimizer[:\s]+(\w+)', 'single'),
                (r'optimization algorithm[:\s]+(\w+)', 'single'),
            ]
        }
        
        # Cache for processed queries
        self.extraction_cache = {}
    
    async def initialize(self):
        """Initialize the extractor (lightweight, no heavy models needed)"""
        if self._initialized:
            return True
        
        if self._loading:
            await asyncio.sleep(0.1)
            return self._initialized
        
        self._loading = True
        
        try:
            logger.info("🔄 Initializing CS parameter extractor...")
            # CS extractor doesn't need heavy models, just patterns
            self._initialized = True
            logger.info("✅ CS parameter extractor initialized")
            
        except Exception as e:
            logger.error(f"❌ CS extractor initialization failed: {e}")
            self._initialized = False
        
        self._loading = False
        return self._initialized
    
    async def extract_parameters(self, query: str) -> Dict[str, Any]:
        """
        Complete CS parameter extraction pipeline:
        1. Pattern matching (regex)
        2. NLTK-based extraction (if available)
        3. Rule Validation
        4. LLM (only if unmapped)
        """
        logger.info(f"💻 Starting comprehensive CS parameter extraction for: {query[:100]}...")
        
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.extraction_cache:
            logger.info("📦 Using cached CS extraction results")
            return self.extraction_cache[cache_key]
        
        # Initialize if needed
        if not await self.initialize():
            logger.warning("⚠️ CS extractor not initialized, using fallback")
            return await self._fallback_extraction(query)
        
        parameters = {}
        extraction_log = []
        
        try:
            # ========== STEP 1: Pattern-based extraction ==========
            start_time = datetime.now()
            pattern_params = await self._extract_with_patterns(query)
            pattern_time = (datetime.now() - start_time).total_seconds()
            
            extraction_log.append({
                "step": "pattern_matching",
                "parameters_found": len(pattern_params),
                "time_seconds": pattern_time
            })
            
            parameters.update(pattern_params)
            
            # ========== STEP 2: NLTK-based extraction (if available) ==========
            try:
                nltk_params = await self._extract_with_nltk(query)
                if nltk_params:
                    extraction_log.append({
                        "step": "nltk_extraction",
                        "parameters_found": len(nltk_params)
                    })
                    for key, param in nltk_params.items():
                        if key not in parameters:
                            parameters[key] = param
            except Exception as e:
                logger.debug(f"NLTK extraction skipped: {e}")
            
            # ========== STEP 3: Validate and Normalize ==========
            validated_params = await self._validate_and_normalize(parameters, query)
            
            # ========== STEP 4: LLM for unmapped parameters ==========
            if len(validated_params) < 2 and len(query.split()) > 5:
                llm_params = await self._extract_with_llm(query)
                for key, param in llm_params.items():
                    if key not in validated_params:
                        validated_params[key] = param
            
            # Add metadata
            validated_params["_metadata"] = {
                "extraction_method": "comprehensive_cs_pipeline",
                "query_length": len(query),
                "pipeline_steps": extraction_log,
                "total_parameters": len([k for k in validated_params.keys() if not k.startswith("_")]),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache results
            self.extraction_cache[cache_key] = validated_params
            if len(self.extraction_cache) > 100:
                self.extraction_cache.pop(next(iter(self.extraction_cache)))
            
            logger.info(f"✅ Comprehensive CS extraction complete: {len(validated_params)} parameters")
            
            return validated_params
            
        except Exception as e:
            logger.error(f"❌ Comprehensive CS extraction failed: {e}")
            return await self._fallback_extraction(query)
    
    async def _extract_with_patterns(self, query: str) -> Dict[str, Any]:
        """Rule-based pattern matching for CS parameters"""
        parameters = {}
        
        # Combine config patterns and extended patterns
        all_patterns = {**self.parameter_patterns, **self.extended_patterns}
        
        for param_type, patterns in all_patterns.items():
            for pattern_info in patterns:
                if isinstance(pattern_info, tuple):
                    if len(pattern_info) == 2:
                        regex, pattern_type = pattern_info
                        default_unit = None
                    elif len(pattern_info) == 3:
                        regex, pattern_type, default_unit = pattern_info
                    else:
                        continue
                else:
                    continue
                
                matches = re.finditer(regex, query, re.IGNORECASE)
                for idx, match in enumerate(matches):
                    try:
                        if pattern_type == 'complexity' and len(match.groups()) >= 1:
                            value = match.group(1)
                            unit = default_unit or 'big-O'
                        elif pattern_type == 'range' and len(match.groups()) >= 2:
                            value = [float(match.group(1)), float(match.group(2))]
                            unit = self._extract_unit_from_match(match.group(0))
                        elif pattern_type == 'single':
                            value = float(match.group(1)) if match.group(1).replace('.', '').replace('-', '').replace('e', '').isdigit() or 'e' in match.group(1).lower() else match.group(1)
                            unit = default_unit or self._extract_unit_from_match(match.group(0))
                        else:
                            continue
                        
                        key = f"{param_type}_{idx+1}" if idx > 0 else param_type
                        
                        parameters[key] = {
                            "value": value,
                            "unit": unit or "",
                            "method": "pattern_matching",
                            "confidence": 0.9,
                            "pattern_used": regex,
                            "raw_match": match.group(0)
                        }
                        
                    except Exception as e:
                        logger.debug(f"Pattern extraction failed: {e}")
        
        return parameters
    
    async def _extract_with_nltk(self, query: str) -> Dict[str, Any]:
        """NLTK-based extraction for CS parameters (optional enhancement)"""
        parameters = {}
        
        try:
            import nltk
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk import pos_tag
            
            sentences = sent_tokenize(query)
            
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                tagged = pos_tag(tokens)
                
                # Look for number + CS term patterns
                for i, (word, tag) in enumerate(tagged):
                    if tag == 'CD':  # Cardinal number
                        # Look for CS terms nearby
                        context_words = [tagged[j][0].lower() for j in range(max(0, i-3), min(len(tagged), i+3))]
                        context_str = ' '.join(context_words)
                        
                        # Check for CS parameter keywords
                        if any(term in context_str for term in ['batch', 'learning rate', 'epoch', 'complexity']):
                            try:
                                num_value = float(word)
                                # Try to identify parameter type from context
                                if 'batch' in context_str:
                                    param_key = 'batch_size'
                                elif 'learning' in context_str or 'lr' in context_str:
                                    param_key = 'learning_rate'
                                elif 'epoch' in context_str:
                                    param_key = 'epochs'
                                else:
                                    continue
                                
                                if param_key not in parameters:
                                    parameters[param_key] = {
                                        "value": num_value,
                                        "unit": "",
                                        "method": "nltk_extraction",
                                        "confidence": 0.7,
                                        "context": sentence
                                    }
                            except ValueError:
                                pass
                                
        except Exception as e:
            logger.debug(f"NLTK extraction failed: {e}")
        
        return parameters
    
    def _extract_unit_from_match(self, text: str) -> str:
        """Extract unit from matched text"""
        unit_patterns = [
            r'(big-O|O\([^)]+\))',
            r'(MB|GB|TB|KB)',
            r'(ms|milliseconds|seconds)',
            r'(fps|qps|rps|ops)',
            r'(samples|epochs|iterations)',
            r'(%)',
            r'(accuracy|precision|recall|f1)'
        ]
        
        for pattern in unit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    async def _validate_and_normalize(self, parameters: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Validate and normalize CS parameters"""
        validated = {}
        
        for key, param in parameters.items():
            if key.startswith('_'):
                continue
            
            value = param.get("value")
            unit = param.get("unit", "")
            
            # Validate ranges
            if isinstance(value, list) and len(value) == 2:
                if value[0] > value[1]:
                    value = [value[1], value[0]]
            
            # Map to CS ontology
            mapped = self._map_to_cs_ontology(key, param)
            if mapped:
                validated[mapped["key"]] = {**param, **mapped["mapping"]}
            else:
                validated[key] = param
        
        # Add inferred parameters
        inferred = self._infer_parameters_from_context(query, validated)
        for key, param in inferred.items():
            if key not in validated:
                validated[key] = param
        
        return validated
    
    def _map_to_cs_ontology(self, key: str, param: Dict[str, Any]) -> Optional[Dict]:
        """Map parameter to CS ontology"""
        key_lower = key.lower()
        
        for ontology_key, ontology_info in self.cs_ontology.items():
            # Check if key matches
            if ontology_key in key_lower:
                return {
                    "key": ontology_key,
                    "mapping": {
                        "ontology_mapped": True,
                        "description": ontology_info.get("description"),
                        "common_values": ontology_info.get("common_values"),
                        "normal_range": ontology_info.get("normal_range"),
                        "units": ontology_info.get("units", []),
                        "is_standard_parameter": True
                    }
                }
        
        # Also check extended patterns
        for pattern_key in self.extended_patterns.keys():
            if pattern_key in key_lower:
                return {
                    "key": pattern_key,
                    "mapping": {
                        "ontology_mapped": True,
                        "is_standard_parameter": True
                    }
                }
        
        return None
    
    def _infer_parameters_from_context(self, query: str, extracted_params: Dict[str, Any]) -> Dict[str, Any]:
        """Infer additional parameters from context"""
        inferred = {}
        query_lower = query.lower()
        
        # Context-based inference for CS
        if any(word in query_lower for word in ['neural network', 'deep learning', 'ml', 'machine learning']):
            inferred["model_type"] = {
                "value": "neural_network",
                "unit": "",
                "method": "context_inference",
                "confidence": 0.7,
                "context": "ML/Deep learning mentioned in query"
            }
        
        if any(word in query_lower for word in ['transformer', 'attention', 'bert', 'gpt']):
            inferred["architecture"] = {
                "value": "transformer",
                "unit": "",
                "method": "context_inference",
                "confidence": 0.8,
                "context": "Transformer architecture mentioned"
            }
        
        if any(word in query_lower for word in ['classification', 'regression', 'clustering']):
            inferred["task_type"] = {
                "value": query_lower.split()[query_lower.split().index([w for w in ['classification', 'regression', 'clustering'] if w in query_lower][0])],
                "unit": "",
                "method": "context_inference",
                "confidence": 0.7,
                "context": "Task type mentioned"
            }
        
        if any(word in query_lower for word in ['gpu', 'cuda', 'parallel', 'distributed']):
            inferred["compute_environment"] = {
                "value": "gpu_parallel",
                "unit": "",
                "method": "context_inference",
                "confidence": 0.6,
                "context": "Parallel/distributed computing mentioned"
            }
        
        return inferred
    
    async def _extract_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM for complex CS parameter extraction"""
        try:
            from core.mistral import generate_with_mistral
            
            prompt = f"""Extract computer science computational parameters from this query:

Query: "{query}"

Return ONLY a JSON object with this structure:
{{
  "parameters": {{
    "parameter_name": {{
      "value": "numeric value or range",
      "unit": "unit of measurement",
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation"
    }}
  }}
}}

Focus on: batch size, learning rate, epochs, time/space complexity, dataset size, accuracy/precision/recall, latency, throughput, hidden units, dropout, optimizer type."""
            
            response, _ = await generate_with_mistral(prompt, max_tokens=300, temperature=0.3)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    llm_params = {}
                    
                    for key, param in data.get("parameters", {}).items():
                        clean_key = key.lower().replace(" ", "_")
                        llm_params[clean_key] = {
                            "value": param.get("value", ""),
                            "unit": param.get("unit", ""),
                            "method": "llm_extraction",
                            "confidence": param.get("confidence", 0.5),
                            "reasoning": param.get("reasoning", "")
                        }
                    
                    logger.info(f"CS LLM extraction found {len(llm_params)} parameters")
                    return llm_params
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response: {e}")
            
        except Exception as e:
            logger.warning(f"CS LLM extraction failed: {e}")
        
        return {}
    
    async def _fallback_extraction(self, query: str) -> Dict[str, Any]:
        """Fallback extraction when main pipeline fails"""
        logger.warning("⚠️ Using fallback CS parameter extraction")
        
        # Simple regex extraction
        parameters = await self._extract_with_patterns(query)
        
        # Add basic metadata
        if parameters:
            parameters["_metadata"] = {
                "extraction_method": "fallback_patterns",
                "query_length": len(query),
                "fallback_used": True,
                "timestamp": datetime.now().isoformat()
            }
        
        return parameters
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the extractor"""
        return {
            "initialized": self._initialized,
            "cache_size": len(self.extraction_cache),
            "cs_ontology_terms": len(self.cs_ontology),
            "parameter_patterns": sum(len(patterns) for patterns in self.parameter_patterns.values()) + sum(len(patterns) for patterns in self.extended_patterns.values())
        }

# Global instance
cs_extractor = ComputerScienceParameterExtractor()

# Async initialization function
async def initialize_cs_extractor():
    """Initialize the CS parameter extractor"""
    return await cs_extractor.initialize()

# Main extraction function
async def extract_cs_parameters(query: str) -> Dict[str, Any]:
    """Main function to extract CS parameters"""
    return await cs_extractor.extract_parameters(query)

# Quick test function
async def test_cs_extraction():
    """Test the CS parameter extraction"""
    test_queries = [
        "Train a neural network with batch size 32, learning rate 0.001, and 10 epochs",
        "Analyze algorithm with time complexity O(n log n) and space complexity O(n)",
        "Model achieves 95% accuracy with latency of 50ms on dataset of 1GB",
        "Optimize hyperparameters: batch_size=64, learning_rate=0.0001, dropout=0.5"
    ]
    
    for query in test_queries:
        print(f"\n💻 Testing: {query}")
        params = await extract_cs_parameters(query)
        
        print(f"📊 Found {len([k for k in params.keys() if not k.startswith('_')])} parameters:")
        for key, param in params.items():
            if not key.startswith('_'):
                print(f"  - {key}: {param.get('value')} {param.get('unit', '')} (confidence: {param.get('confidence', 0):.2f})")
