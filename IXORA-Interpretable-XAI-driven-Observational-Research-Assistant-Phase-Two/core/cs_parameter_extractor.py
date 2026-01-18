# core/cs_parameter_extractor.py - COMPREHENSIVE CS PARAMETER EXTRACTION PIPELINE

import spacy
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
import json
import asyncio
from datetime import datetime
import hashlib
import numpy as np

logger = logging.getLogger("core.cs_parameter_extractor")

class ComputerScienceParameterExtractor:
    """Comprehensive computer science parameter extraction pipeline"""
    
    def __init__(self):
        self.nlp = None
        self._initialized = False
        self._loading = False
        
        # CS parameter patterns with improved regex
        self.parameter_patterns = {
            'learning_rate': [
                (r'learning[-\s]rate\s*[=:]?\s*([\d\.e-]+)', 'single'),
                (r'lr\s*[=:]?\s*([\d\.e-]+)', 'single'),
                (r'([\d\.e-]+)\s*e[-]?([\d]+)', 'scientific'),  # Matches 1e-4, 2.5e-3, etc.
            ],
            'batch_size': [
                (r'batch[-\s]size\s*[=:]?\s*(\d+)', 'single'),
                (r'batch\s*(\d+)', 'single'),
                (r'bs\s*[=:]?\s*(\d+)', 'single'),
            ],
            'epochs': [
                (r'epochs?\s*[=:]?\s*(\d+)', 'single'),
                (r'train(?:ing)?\s*(?:for\s*)?(\d+)\s*epochs?', 'single'),
            ],
            'optimizer': [
                (r'optimizer\s*[=:]?\s*([a-zA-Z0-9_+\-]+)', 'single'),
                (r'(adam|sgd|rmsprop|adagrad|adadelta|adamw|nadam)', 'single'),
            ],
            'dropout': [
                (r'dropout\s*[=:]?\s*([\d\.]+)', 'single'),
                (r'drop[-\s]out\s*[=:]?\s*([\d\.]+)', 'single'),
            ],
            'hidden_units': [
                (r'hidden[-\s]units?\s*[=:]?\s*(\d+)', 'single'),
                (r'hidden[-\s]size\s*[=:]?\s*(\d+)', 'single'),
                (r'num[-\s]units?\s*[=:]?\s*(\d+)', 'single'),
            ],
            'layers': [
                (r'(\d+)\s*layers?', 'single'),
                (r'num[-\s]layers?\s*[=:]?\s*(\d+)', 'single'),
            ],
            'weight_decay': [
                (r'weight[-\s]decay\s*[=:]?\s*([\d\.e-]+)', 'single'),
                (r'L2\s*[=:]?\s*([\d\.e-]+)', 'single'),
            ],
            'warmup_steps': [
                (r'warmup[-\s]steps?\s*[=:]?\s*(\d+)', 'single'),
                (r'warm[-\s]up\s*(?:steps?)?\s*[=:]?\s*(\d+)', 'single'),
            ],
            'gradient_clip': [
                (r'grad(?:ient)?[-\s]clip(?:ping)?\s*[=:]?\s*([\d\.]+)', 'single'),
                (r'clip[-\s]grad(?:ient)?\s*[=:]?\s*([\d\.]+)', 'single'),
            ]
        }
        
        # CS parameter ontology
        self.cs_ontology = {
            'learning_rate': {
                'description': 'Learning rate for optimization',
                'default_range': [1e-5, 1e-1],
                'typical_values': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
                'units': ['', 'per step'],
                'type': 'float'
            },
            'batch_size': {
                'description': 'Number of samples per batch',
                'default_range': [8, 1024],
                'typical_values': [16, 32, 64, 128, 256, 512],
                'units': ['samples'],
                'type': 'int'
            },
            'epochs': {
                'description': 'Number of training epochs',
                'default_range': [1, 1000],
                'typical_values': [10, 20, 50, 100, 200],
                'units': ['epochs'],
                'type': 'int'
            },
            'optimizer': {
                'description': 'Optimization algorithm',
                'possible_values': ['adam', 'sgd', 'rmsprop', 'adamw', 'adagrad', 'adadelta', 'nadam'],
                'default': 'adam',
                'type': 'categorical'
            },
            'dropout': {
                'description': 'Dropout rate for regularization',
                'default_range': [0.0, 0.9],
                'typical_values': [0.1, 0.2, 0.3, 0.4, 0.5],
                'units': ['probability'],
                'type': 'float'
            },
            'hidden_units': {
                'description': 'Number of hidden units in a layer',
                'default_range': [32, 4096],
                'typical_values': [64, 128, 256, 512, 1024, 2048],
                'units': ['units'],
                'type': 'int'
            },
            'layers': {
                'description': 'Number of layers in the model',
                'default_range': [1, 100],
                'typical_values': [1, 2, 3, 4, 6, 8, 12, 24, 48],
                'units': ['layers'],
                'type': 'int'
            },
            'weight_decay': {
                'description': 'L2 regularization coefficient',
                'default_range': [0.0, 0.1],
                'typical_values': [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                'units': ['weight decay'],
                'type': 'float'
            },
            'warmup_steps': {
                'description': 'Number of warmup steps for learning rate',
                'default_range': [0, 10000],
                'typical_values': [0, 100, 1000, 4000, 8000],
                'units': ['steps'],
                'type': 'int'
            },
            'gradient_clip': {
                'description': 'Maximum gradient norm for clipping',
                'default_range': [0.1, 10.0],
                'typical_values': [0.5, 1.0, 5.0],
                'units': ['norm'],
                'type': 'float'
            }
        }
        
        # Cache for processed queries
        self.extraction_cache = {}
    
    async def initialize(self):
        """Initialize the NLP pipeline"""
        if self._initialized:
            return True
        
        if self._loading:
            await asyncio.sleep(0.1)
            return self._initialized
        
        self._loading = True
        
        try:
            logger.info("🔄 Loading spaCy model for CS parameter extraction...")
            # Using en_core_web_md as it's good for general technical text
            self.nlp = spacy.load("en_core_web_md")
            logger.info("✅ spaCy model loaded successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to load spaCy model: {e}")
            self._initialized = False
        
        self._loading = False
        return self._initialized
    
    def _parse_scientific_notation(self, match) -> float:
        """Convert scientific notation string to float"""
        try:
            base = float(match.group(1))
            exponent = int(match.group(2))
            return base * (10 ** -exponent)
        except (IndexError, ValueError):
            return None
    
    def _extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """Extract parameters using regex patterns"""
        parameters = {}
        
        for param_name, patterns in self.parameter_patterns.items():
            for pattern, pattern_type in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if pattern_type == 'range':
                        try:
                            start = float(match.group(1))
                            end = float(match.group(2))
                            value = {
                                'value': (start + end) / 2,  # Use midpoint
                                'range': [start, end],
                                'source': 'regex_range',
                                'confidence': 0.9
                            }
                            parameters[param_name] = value
                        except (IndexError, ValueError):
                            continue
                    
                    elif pattern_type == 'single':
                        try:
                            value = {
                                'value': float(match.group(1)),
                                'source': 'regex_single',
                                'confidence': 0.8
                            }
                            parameters[param_name] = value
                        except (IndexError, ValueError):
                            continue
                    
                    elif pattern_type == 'scientific':
                        try:
                            value = self._parse_scientific_notation(match)
                            if value is not None:
                                parameters[param_name] = {
                                    'value': value,
                                    'source': 'regex_scientific',
                                    'confidence': 0.85
                                }
                        except Exception:
                            continue
        
        return parameters
    
    def _validate_parameter(self, param_name: str, value: Any) -> Dict[str, Any]:
        """Validate parameter against ontology"""
        if param_name not in self.cs_ontology:
            return None
        
        param_info = self.cs_ontology[param_name]
        result = {
            'name': param_name,
            'description': param_info.get('description', ''),
            'value': value,
            'type': param_info.get('type', 'float'),
            'valid': True,
            'source': 'extracted',
            'confidence': 0.8
        }
        
        # Type conversion
        try:
            if param_info.get('type') == 'int':
                result['value'] = int(float(value))
            elif param_info.get('type') == 'float':
                result['value'] = float(value)
        except (ValueError, TypeError):
            result['valid'] = False
            result['error'] = f"Invalid value type for {param_name}"
            return result
        
        # Range validation for numeric parameters
        if 'default_range' in param_info and isinstance(value, (int, float)):
            min_val, max_val = param_info['default_range']
            if value < min_val or value > max_val:
                result['warning'] = f"Value {value} is outside typical range [{min_val}, {max_val}]"
                result['suggested_range'] = param_info['default_range']
                result['confidence'] *= 0.7  # Reduce confidence for out-of-range values
        
        # Check against typical values
        if 'typical_values' in param_info and isinstance(value, (int, float)):
            typical_values = param_info['typical_values']
            if value not in typical_values:
                # Find closest typical value
                closest = min(typical_values, key=lambda x: abs(x - value))
                result['suggested_value'] = closest
                result['confidence'] *= 0.9  # Slight reduction for non-typical values
        
        # For categorical parameters, check if value is in allowed set
        if 'possible_values' in param_info:
            if str(value).lower() not in [v.lower() for v in param_info['possible_values']]:
                result['warning'] = f"Value '{value}' not in allowed values {param_info['possible_values']}"
                result['suggested_values'] = param_info['possible_values']
                result['confidence'] *= 0.6  # Larger reduction for invalid categorical values
        
        return result
    
    async def extract_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract CS parameters from text
        
        Args:
            query: Input text containing CS parameters
            
        Returns:
            Dictionary with extracted parameters and metadata
        """
        logger.info(f"🔍 Extracting CS parameters from: {query[:100]}...")
        
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.extraction_cache:
            logger.debug("📦 Using cached extraction results")
            return self.extraction_cache[cache_key]
        
        # Initialize if needed
        if not await self.initialize():
            logger.warning("⚠️ NLP pipeline not initialized, using basic extraction")
            return self._basic_extraction(query)
        
        result = {
            'parameters': {},
            'extraction_metadata': {
                'method': 'cs_parameter_extractor',
                'version': '1.0',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        try:
            # Step 1: Extract using regex patterns
            extracted = self._extract_with_patterns(query)
            
            # Step 2: Validate and process extracted parameters
            for param_name, param_data in extracted.items():
                validated = self._validate_parameter(param_name, param_data['value'])
                if validated and validated.get('valid', False):
                    result['parameters'][param_name] = {
                        'value': validated['value'],
                        'confidence': validated.get('confidence', 0.7),
                        'source': param_data.get('source', 'unknown'),
                        'type': validated.get('type', 'unknown'),
                        'description': validated.get('description', '')
                    }
                    
                    # Add warnings if any
                    if 'warning' in validated:
                        result['parameters'][param_name]['warning'] = validated['warning']
                    
                    # Add suggestions if any
                    if 'suggested_value' in validated:
                        result['parameters'][param_name]['suggested_value'] = validated['suggested_value']
                    if 'suggested_range' in validated:
                        result['parameters'][param_name]['suggested_range'] = validated['suggested_range']
                    if 'suggested_values' in validated:
                        result['parameters'][param_name]['suggested_values'] = validated['suggested_values']
            
            # Step 3: Add any missing parameters with default values
            for param_name, param_info in self.cs_ontology.items():
                if param_name not in result['parameters'] and 'default' in param_info:
                    result['parameters'][param_name] = {
                        'value': param_info['default'],
                        'confidence': 0.5,
                        'source': 'default',
                        'type': param_info.get('type', 'unknown'),
                        'description': param_info.get('description', '')
                    }
            
            # Cache the result
            self.extraction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in parameter extraction: {e}", exc_info=True)
            return {
                'parameters': {},
                'error': str(e),
                'extraction_metadata': {
                    'method': 'cs_parameter_extractor',
                    'version': '1.0',
                    'error': True,
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def _basic_extraction(self, query: str) -> Dict[str, Any]:
        """Fallback extraction without NLP"""
        return {
            'parameters': self._extract_with_patterns(query),
            'extraction_metadata': {
                'method': 'basic_cs_parameter_extractor',
                'version': '1.0',
                'basic_mode': True,
                'timestamp': datetime.now().isoformat()
            }
        }

# Global instance
cs_extractor = ComputerScienceParameterExtractor()

# Async initialization function
async def initialize_extractor():
    """Initialize the CS parameter extractor"""
    return await cs_extractor.initialize()

# Main extraction function
async def extract_cs_parameters(query: str) -> Dict[str, Any]:
    """Main function to extract CS parameters"""
    return await cs_extractor.extract_parameters(query)

# Quick test function
async def test_extraction():
    """Test the parameter extraction"""
    test_queries = [
        "Train a model with learning rate 1e-4, batch size 32, and 100 epochs",
        "Use Adam optimizer with weight decay 1e-5 and learning rate 2.5e-4",
        "Run for 50 epochs with batch size 64 and dropout 0.3",
        "Train a transformer with 12 layers, 768 hidden units, and 12 attention heads",
        "Use learning rate 0.001 with cosine decay and 10000 warmup steps"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: {query}")
        result = await extract_cs_parameters(query)
        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_extraction())
