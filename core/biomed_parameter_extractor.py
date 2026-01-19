# core/biomed_parameter_extractor.py - COMPREHENSIVE PARAMETER EXTRACTION PIPELINE

import spacy
import scispacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
import json
import asyncio
from datetime import datetime
import hashlib

logger = logging.getLogger("core.biomed_parameter_extractor")

class BiomedicalParameterExtractor:
    """Comprehensive biomedical parameter extraction pipeline"""
    
    def __init__(self):
        self.nlp = None
        self.linker = None
        self._initialized = False
        self._loading = False
        
        # Biomedical parameter patterns
        self.parameter_patterns = {
            'ph': [
                (r'pH\s*([\d\.]+)\s*[–\-]\s*([\d\.]+)', 'range'),
                (r'pH\s*([\d\.]+)', 'single'),
                (r'[Pp][Hh]\s*([\d\.]+)', 'single'),
            ],
            'temperature': [
                (r'(\d+\.?\d*)\s*[–\-]\s*(\d+\.?\d*)\s*°?\s*[Cc]', 'range'),
                (r'(\d+\.?\d*)\s*°?\s*[Cc]', 'single'),
                (r'temp(?:erature)?\s*([\d\.]+)', 'single'),
            ],
            'concentration': [
                (r'(\d+\.?\d*)\s*(mM|µM|nM|M|mg/ml|mg/mL|mg/L|g/L|%)', 'single'),
                (r'conc(?:entration)?\s*([\d\.]+)\s*(mM|µM|nM|M)', 'single'),
            ],
            'time': [
                (r'(\d+\.?\d*)\s*[–\-]\s*(\d+\.?\d*)\s*(hours?|hrs?|minutes?|mins?|days?)', 'range'),
                (r'(\d+\.?\d*)\s*(hours?|hrs?|minutes?|mins?|days?)', 'single'),
            ],
            'pressure': [
                (r'(\d+\.?\d*)\s*(kPa|MPa|atm|bar|psi)', 'single'),
            ],
            'volume': [
                (r'(\d+\.?\d*)\s*(ml|mL|L|µl|μL)', 'single'),
            ],
            'rpm': [
                (r'(\d+\.?\d*)\s*rpm', 'single'),
                (r'(\d+\.?\d*)\s*RPM', 'single'),
            ]
        }
        
        # UMLS Semantic Types for biomedical parameters
        self.relevant_semantic_types = {
            'T046',  # Pathologic Function
            'T047',  # Disease or Syndrome
            'T048',  # Mental or Behavioral Dysfunction
            'T121',  # Pharmacologic Substance
            'T122',  # Biomedical or Dental Material
            'T123',  # Biologically Active Substance
            'T184',  # Sign or Symptom
            'T201',  # Clinical Attribute
        }
        
        # Biomedical ontology mapping
        self.biomedical_ontology = {
            'ph': {
                'umls_cui': 'C0031640',
                'preferred_term': 'Hydrogen-Ion Concentration',
                'synonyms': ['pH', 'hydrogen ion concentration', 'acidity', 'alkalinity'],
                'normal_range': [7.35, 7.45],
                'units': ['pH']
            },
            'temperature': {
                'umls_cui': 'C0039478',
                'preferred_term': 'Body Temperature',
                'synonyms': ['temperature', 'temp', 'body temp', 'fever'],
                'normal_range': [36.5, 37.5],
                'units': ['°C', 'C', 'degrees celsius']
            },
            'concentration': {
                'umls_cui': 'C0009676',
                'preferred_term': 'Concentration',
                'synonyms': ['concentration', 'conc', 'dose', 'dosage'],
                'units': ['mM', 'µM', 'nM', 'M', 'mg/mL', 'g/L', '%']
            },
            'incubation_time': {
                'umls_cui': 'C0020956',
                'preferred_term': 'Incubation Period',
                'synonyms': ['incubation time', 'incubation period', 'culture time'],
                'units': ['hours', 'hrs', 'days']
            },
            'agitation': {
                'umls_cui': 'C0001457',
                'preferred_term': 'Agitation',
                'synonyms': ['agitation', 'shaking', 'stirring', 'mixing'],
                'units': ['rpm', 'Hz']
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
            logger.info("🔄 Loading SciSpaCy models for biomedical parameter extraction...")
            
            # Load SciSpaCy model
            self.nlp = spacy.load("en_core_sci_md")
            
            # Add abbreviation detector
            self.nlp.add_pipe("abbreviation_detector")
            
            # Add UMLS linker
            self.linker = EntityLinker(
                resolve_abbreviations=True,
                name="umls",
                threshold=0.7
            )
            self.nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})
            
            logger.info("✅ SciSpaCy models loaded successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to load SciSpaCy models: {e}")
            # Fallback to regular spaCy
            try:
                self.nlp = spacy.load("en_core_web_md")
                logger.info("✅ Loaded fallback spaCy model")
                self._initialized = True
            except Exception as e2:
                logger.error(f"❌ Fallback model also failed: {e2}")
                self._initialized = False
        
        self._loading = False
        return self._initialized
    
    async def extract_parameters(self, query: str) -> Dict[str, Any]:
        """
        Complete parameter extraction pipeline:
        1. NER (SciSpaCy)
        2. Ontology Linking (UMLS)
        3. Dependency Parsing
        4. Rule Validation
        5. LLM (only if unmapped)
        """
        logger.info(f"🔬 Starting comprehensive parameter extraction for: {query[:100]}...")
        
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.extraction_cache:
            logger.info("📦 Using cached extraction results")
            return self.extraction_cache[cache_key]
        
        # Initialize if needed
        if not await self.initialize():
            logger.warning("⚠️ NLP pipeline not initialized, using fallback")
            return await self._fallback_extraction(query)
        
        parameters = {}
        extraction_log = []
        
        try:
            # ========== STEP 1: NER with SciSpaCy ==========
            start_time = datetime.now()
            doc = self.nlp(query)
            ner_time = (datetime.now() - start_time).total_seconds()
            
            extraction_log.append({
                "step": "ner",
                "entities_found": len(doc.ents),
                "time_seconds": ner_time
            })
            
            # Extract entities
            for ent in doc.ents:
                logger.debug(f"NER Entity: {ent.text} ({ent.label_})")
            
            # ========== STEP 2: Ontology Linking (UMLS) ==========
            if hasattr(doc, '_.umls_ents') and doc._.umls_ents:
                for entity in doc._.umls_ents:
                    cui = entity[0]
                    score = entity[1]
                    
                    # Get entity details from linker
                    concept = self.linker.kb.cui_to_entity[cui]
                    
                    logger.debug(f"UMLS Concept: {concept.canonical_name} (CUI: {cui}, Score: {score:.2f})")
                    
                    # Check if it's a relevant biomedical parameter
                    if any(st in self.relevant_semantic_types for st in concept.types):
                        param_name = concept.canonical_name.lower()
                        parameters[f"umls_{cui}"] = {
                            "value": param_name,
                            "unit": "",
                            "method": "umls_linking",
                            "confidence": float(score),
                            "umls_cui": cui,
                            "canonical_name": concept.canonical_name,
                            "semantic_types": list(concept.types)
                        }
            
            # ========== STEP 3: Dependency Parsing ==========
            # Find numeric values and their dependencies
            numeric_entities = []
            for token in doc:
                if token.like_num:
                    # Find what this number modifies
                    param_name = self._find_parameter_name(token)
                    numeric_value = self._extract_numeric_value(token.text)
                    
                    if param_name and numeric_value is not None:
                        numeric_entities.append({
                            "param_name": param_name,
                            "value": numeric_value,
                            "unit": self._infer_unit_from_context(token, doc),
                            "dependency": token.dep_,
                            "head": token.head.text
                        })
            
            extraction_log.append({
                "step": "dependency_parsing",
                "numeric_entities": len(numeric_entities)
            })
            
            # ========== STEP 4: Rule-based Pattern Matching ==========
            rule_based_params = await self._extract_with_patterns(query)
            
            # Merge rule-based parameters
            for key, param in rule_based_params.items():
                if key not in parameters:
                    parameters[key] = param
                else:
                    # Update confidence if rule-based is higher
                    if param.get("confidence", 0) > parameters[key].get("confidence", 0):
                        parameters[key] = param
            
            # ========== STEP 5: Validate and Normalize ==========
            validated_params = await self._validate_and_normalize(parameters, query)
            
            # Add numeric entities from dependency parsing
            for entity in numeric_entities:
                param_key = entity["param_name"].lower().replace(" ", "_")
                if param_key not in validated_params:
                    validated_params[param_key] = {
                        "value": entity["value"],
                        "unit": entity["unit"],
                        "method": "dependency_parsing",
                        "confidence": 0.7,
                        "dependency_info": {
                            "dep": entity["dependency"],
                            "head": entity["head"]
                        }
                    }
            
            # ========== STEP 6: LLM for unmapped parameters ==========
            if len(validated_params) < 2 and len(query.split()) > 5:
                # Try LLM extraction for complex queries with few parameters
                llm_params = await self._extract_with_llm(query)
                for key, param in llm_params.items():
                    if key not in validated_params:
                        validated_params[key] = param
            
            # Add metadata
            validated_params["_metadata"] = {
                "extraction_method": "comprehensive_pipeline",
                "query_length": len(query),
                "pipeline_steps": extraction_log,
                "total_parameters": len([k for k in validated_params.keys() if not k.startswith("_")]),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache results
            self.extraction_cache[cache_key] = validated_params
            if len(self.extraction_cache) > 100:  # Limit cache size
                self.extraction_cache.pop(next(iter(self.extraction_cache)))
            
            logger.info(f"✅ Comprehensive extraction complete: {len(validated_params)} parameters")
            
            return validated_params
            
        except Exception as e:
            logger.error(f"❌ Comprehensive extraction failed: {e}")
            return await self._fallback_extraction(query)
    
    def _find_parameter_name(self, token):
        """Find parameter name from dependency relations"""
        # Look for nouns that this number modifies
        for child in token.head.children:
            if child.dep_ in ["nsubj", "dobj", "attr", "amod"] and child.pos_ in ["NOUN", "PROPN"]:
                return child.text
        
        # Look at the head word
        if token.head.pos_ in ["NOUN", "PROPN"]:
            return token.head.text
        
        # Look for compound nouns
        for anc in token.ancestors:
            if anc.pos_ in ["NOUN", "PROPN"]:
                return anc.text
        
        return None
    
    def _extract_numeric_value(self, text: str):
        """Extract numeric value from text"""
        try:
            # Handle ranges like "5-7"
            if '-' in text or '–' in text:
                parts = re.split(r'[–\-]', text)
                if len(parts) == 2:
                    return [float(parts[0]), float(parts[1])]
            
            # Handle single numbers
            return float(re.sub(r'[^\d\.]', '', text))
        except:
            return None
    
    def _infer_unit_from_context(self, token, doc):
        """Infer unit from surrounding context"""
        # Look at next token
        if token.i + 1 < len(doc):
            next_token = doc[token.i + 1]
            next_text = next_token.text.lower()
            
            unit_map = {
                'ph': 'pH',
                '°c': '°C',
                'c': '°C',
                'mm': 'mM',
                'μm': 'µM',
                'nm': 'nM',
                'm': 'M',
                'mg/ml': 'mg/mL',
                'g/l': 'g/L',
                '%': '%',
                'hours': 'hours',
                'hrs': 'hours',
                'minutes': 'minutes',
                'days': 'days',
                'rpm': 'rpm'
            }
            
            for key, unit in unit_map.items():
                if key in next_text:
                    return unit
        
        return ""
    
    async def _extract_with_patterns(self, query: str) -> Dict[str, Any]:
        """Rule-based pattern matching"""
        parameters = {}
        
        for param_type, patterns in self.parameter_patterns.items():
            for pattern, pattern_type in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for idx, match in enumerate(matches):
                    try:
                        if pattern_type == 'range' and len(match.groups()) >= 2:
                            value = [float(match.group(1)), float(match.group(2))]
                            unit = self._extract_unit_from_match(match.group(0))
                        elif pattern_type == 'single':
                            value = float(match.group(1))
                            unit = self._extract_unit_from_match(match.group(0))
                        else:
                            continue
                        
                        key = f"{param_type}_{idx+1}" if idx > 0 else param_type
                        
                        parameters[key] = {
                            "value": value,
                            "unit": unit,
                            "method": "pattern_matching",
                            "confidence": 0.9,
                            "pattern_used": pattern,
                            "raw_match": match.group(0)
                        }
                        
                    except Exception as e:
                        logger.debug(f"Pattern extraction failed: {e}")
        
        return parameters
    
    def _extract_unit_from_match(self, text: str) -> str:
        """Extract unit from matched text"""
        unit_patterns = [
            r'(pH)',
            r'(°?[Cc])',
            r'(mM|µM|nM|M|mg/ml|mg/mL|mg/L|g/L|%)',
            r'(hours?|hrs?|minutes?|mins?|days?)',
            r'(rpm|RPM)',
            r'(kPa|MPa|atm|bar|psi)',
            r'(ml|mL|L|µl|μL)'
        ]
        
        for pattern in unit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    async def _validate_and_normalize(self, parameters: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Validate and normalize parameters"""
        validated = {}
        
        for key, param in parameters.items():
            if key.startswith('_'):
                continue
            
            value = param.get("value")
            unit = param.get("unit", "")
            
            # Validate ranges
            if isinstance(value, list) and len(value) == 2:
                if value[0] > value[1]:  # Swap if min > max
                    value = [value[1], value[0]]
                
                # Check for impossible values
                if key.startswith('ph') and (value[0] < 0 or value[1] > 14):
                    param["confidence"] *= 0.5  # Penalize
                    param["validation_note"] = "pH outside theoretical range 0-14"
                
                if key.startswith('temp') and (value[0] < -273 or value[1] > 1000):
                    param["confidence"] *= 0.5  # Penalize
                    param["validation_note"] = "Temperature outside reasonable range"
            
            # Map to biomedical ontology
            mapped = self._map_to_biomedical_ontology(key, param)
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
    
    def _map_to_biomedical_ontology(self, key: str, param: Dict[str, Any]) -> Optional[Dict]:
        """Map parameter to biomedical ontology"""
        for ontology_key, ontology_info in self.biomedical_ontology.items():
            # Check if key matches
            if ontology_key in key.lower():
                return {
                    "key": ontology_key,
                    "mapping": {
                        "ontology_mapped": True,
                        "umls_cui": ontology_info["umls_cui"],
                        "preferred_term": ontology_info["preferred_term"],
                        "normal_range": ontology_info.get("normal_range"),
                        "is_standard_parameter": True
                    }
                }
            
            # Check synonyms
            synonyms = ontology_info.get("synonyms", [])
            for synonym in synonyms:
                if synonym.lower() in key.lower():
                    return {
                        "key": ontology_key,
                        "mapping": {
                            "ontology_mapped": True,
                            "umls_cui": ontology_info["umls_cui"],
                            "preferred_term": ontology_info["preferred_term"],
                            "normal_range": ontology_info.get("normal_range"),
                            "is_standard_parameter": True
                        }
                    }
        
        return None
    
    def _infer_parameters_from_context(self, query: str, extracted_params: Dict[str, Any]) -> Dict[str, Any]:
        """Infer additional parameters from context"""
        inferred = {}
        query_lower = query.lower()
        
        # Context-based inference
        if any(word in query_lower for word in ['yeast', 'saccharomyces', 'cerevisiae']):
            inferred["organism"] = {
                "value": "Saccharomyces cerevisiae",
                "unit": "",
                "method": "context_inference",
                "confidence": 0.8,
                "context": "Yeast mentioned in query"
            }
        
        if any(word in query_lower for word in ['e.coli', 'escherichia', 'bacteria']):
            inferred["organism"] = {
                "value": "Escherichia coli",
                "unit": "",
                "method": "context_inference",
                "confidence": 0.8,
                "context": "Bacteria mentioned in query"
            }
        
        if any(word in query_lower for word in ['growth', 'biomass', 'cell density']):
            inferred["growth_measure"] = {
                "value": "OD600",
                "unit": "",
                "method": "context_inference",
                "confidence": 0.7,
                "context": "Growth measurement inferred"
            }
        
        if any(word in query_lower for word in ['enzyme', 'kinase', 'protease', 'catalyst']):
            inferred["biological_catalyst"] = {
                "value": "enzyme",
                "unit": "",
                "method": "context_inference",
                "confidence": 0.6,
                "context": "Enzymatic reaction inferred"
            }
        
        return inferred
    
    async def _extract_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM for complex parameter extraction"""
        try:
            from core.mistral import generate_with_mistral
            
            prompt = f"""Extract biomedical experimental parameters from this query:

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

Focus on: pH, temperature, concentration, time, pressure, volume, agitation speed, biological components."""
            
            response, _ = await generate_with_mistral(prompt, max_tokens=300, temperature=0.3)
            
            # Extract JSON from response
            import json
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
                    
                    logger.info(f"LLM extraction found {len(llm_params)} parameters")
                    return llm_params
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response: {e}")
            
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
        
        return {}
    
    async def _fallback_extraction(self, query: str) -> Dict[str, Any]:
        """Fallback extraction when main pipeline fails"""
        logger.warning("⚠️ Using fallback parameter extraction")
        
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
            "biomedical_ontology_terms": len(self.biomedical_ontology),
            "parameter_patterns": sum(len(patterns) for patterns in self.parameter_patterns.values())
        }

# Global instance
biomed_extractor = BiomedicalParameterExtractor()

# Async initialization function
async def initialize_extractor():
    """Initialize the biomedical parameter extractor"""
    return await biomed_extractor.initialize()

# Main extraction function
async def extract_biomedical_parameters(query: str) -> Dict[str, Any]:
    """Main function to extract biomedical parameters"""
    return await biomed_extractor.extract_parameters(query)

# Quick test function
async def test_extraction():
    """Test the parameter extraction"""
    test_queries = [
        "What's the optimal pH range 5-8 and temperature 25-35°C for yeast growth?",
        "I need to incubate E. coli at 37°C for 24 hours with 100 rpm agitation",
        "Study enzyme kinetics at pH 7.4 with 10 mM substrate concentration",
        "How does 50 µM drug concentration affect cell viability at 48 hours?"
    ]
    
    for query in test_queries:
        print(f"\n🔬 Testing: {query}")
        params = await extract_biomedical_parameters(query)
        
        print(f"📊 Found {len([k for k in params.keys() if not k.startswith('_')])} parameters:")
        for key, param in params.items():
            if not key.startswith('_'):
                print(f"  - {key}: {param.get('value')} {param.get('unit', '')} (confidence: {param.get('confidence', 0):.2f})")