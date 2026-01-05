# core/utils.py - COMPLETE NLTK-BASED PARAMETER EXTRACTION WITH FIXES
import re
import logging
import json
import hashlib
import time
import uuid
import os
import nltk
from typing import Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger("core.utils")

# ========== NLTK SETUP (AUTOMATIC DOWNLOAD) ==========

def _setup_nltk():
    """Ensure NLTK data is downloaded"""
    try:
        # Check and download required NLTK data
        nltk_data_path = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_path, exist_ok=True)
        
        required_packages = [
            'punkt',           # Tokenizer
            'averaged_perceptron_tagger',  # POS tagger
            'maxent_ne_chunker',  # Named entity chunker
            'words',           # Word corpus
            'punkt_tab'        # Improved tokenizer
        ]
        
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'taggers/{package}' if 'perceptron' in package else f'chunkers/{package}' if 'chunker' in package else f'corpora/{package}')
                logger.debug(f"NLTK {package} already installed")
            except LookupError:
                logger.info(f"Downloading NLTK {package}...")
                nltk.download(package, quiet=True, raise_on_error=True)
        
        return True
    except Exception as e:
        logger.error(f"NLTK setup failed: {e}")
        return False

# Setup NLTK on import
_nltk_ready = _setup_nltk()
logger.info(f"NLTK ready: {_nltk_ready}")

# ========== INTENT DETECTION WITH NLTK ==========

async def detect_intent(query: str, domain: str = "biomed") -> str:
    """Enhanced intent detection using NLTK POS tagging"""
    query_lower = query.lower().strip()
    
    # Always try NLTK first if available
    if _nltk_ready:
        try:
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk import pos_tag
            
            # Tokenize and POS tag
            sentences = sent_tokenize(query)
            all_tokens = []
            all_tags = []
            
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                tagged = pos_tag(tokens)
                all_tokens.extend(tokens)
                all_tags.extend([tag for _, tag in tagged])
            
            # Analyze POS patterns
            pos_string = ' '.join(all_tags[:15])  # Look at first 15 tags
            
            # Rule 1: Questions (starts with WH-word or contains ?)
            if query.strip().endswith('?') or any(word in query_lower.split()[0] for word in ['what', 'how', 'why', 'when', 'where', 'which', 'who']):
                # Check if it's a research question
                research_indicators = ['study', 'experiment', 'research', 'effect', 'impact', 'correlation']
                if any(indicator in query_lower for indicator in research_indicators):
                    return "research"
                return "conversational"
            
            # Rule 2: Contains imperative verbs (commands)
            if 'VB' in pos_string:  # Base form verb
                analyze_verbs = ['analyze', 'calculate', 'compute', 'optimize', 'compare', 'evaluate']
                research_verbs = ['study', 'investigate', 'examine', 'explore', 'determine']
                
                tokens_lower = [t.lower() for t in all_tokens]
                if any(verb in tokens_lower for verb in analyze_verbs):
                    return "analyze"
                if any(verb in tokens_lower for verb in research_verbs):
                    return "research"
            
            # Rule 3: Contains numbers and units (parameter analysis)
            number_pattern = r'\d+\.?\d*'
            unit_pattern = r'(ph|°?c|m[m]?|µg|mg|g|l|ml|rpm|hr|min|sec|day)'
            
            if re.search(number_pattern, query_lower) and re.search(unit_pattern, query_lower):
                return "analyze"
            
            # Rule 4: Contains scientific methodology terms
            methodology_terms = ['method', 'protocol', 'procedure', 'design', 'setup', 'trial', 'replicate']
            if any(term in query_lower for term in methodology_terms):
                return "research"
            
            # Rule 5: Short queries (< 5 words) are conversational
            if len(all_tokens) < 5:
                return "conversational"
                
        except Exception as e:
            logger.warning(f"NLTK intent detection failed, using fallback: {e}")
    
    # FALLBACK: Keyword-based detection
    # Check for conversation keywords
    conversational_keywords = [
        "hello", "hi", "hey", "how are you", "thank you", "thanks",
        "what is", "what are", "tell me about", "explain", "define",
        "who is", "who are", "can you", "could you", "would you",
        "please", "help me", "i need", "i want", "i'm looking for",
        "good morning", "good afternoon", "good evening"
    ]
    
    if any(keyword in query_lower for keyword in conversational_keywords):
        return "conversational"
    
    # Check for RESEARCH intent
    research_keywords = [
        "study", "studies", "experiment", "experiments", "research",
        "investigate", "investigation", "paper", "publication", "journal",
        "method", "methodology", "protocol", "procedure", "trial",
        "clinical trial", "randomized", "cohort", "case study", "literature",
        "hypothesis", "objective", "aim", "goal", "purpose"
    ]
    
    if any(keyword in query_lower for keyword in research_keywords):
        return "research"
    
    # Check for ANALYZE intent
    analyze_keywords = [
        "analyze", "analysis", "statistical", "statistics", "correlation",
        "regression", "anova", "t-test", "p-value", "significant", "significance",
        "optimize", "optimization", "maximize", "minimize", "improve", "enhance",
        "compare", "comparison", "contrast", "versus", "vs", "difference",
        "effect of", "impact of", "influence of", "relationship between",
        "predict", "forecast", "estimate", "calculate", "compute",
        "parameter", "variable", "factor", "condition"
    ]
    
    if any(keyword in query_lower for keyword in analyze_keywords):
        return "analyze"
    
    # Default based on domain
    if domain == "biomed":
        # Biomedical queries often imply research
        biomedical_terms = ["ph", "temperature", "enzyme", "protein", "cell", "growth", 
                           "biomass", "yeast", "bacteria", "culture", "incubation",
                           "dose", "concentration", "treatment", "therapy", "drug"]
        
        if any(term in query_lower for term in biomedical_terms):
            return "research"
    
    return "conversational"

# ========== NLTK-BASED PARAMETER EXTRACTION ==========

async def extract_parameters_nltk(query: str, domain: str = "biomed") -> Dict[str, Any]:
    """
    Robust NLTK parameter extraction with error handling
    """
    parameters = {}
    
    if not _nltk_ready:
        logger.warning("NLTK not ready, using regex extraction")
        return await extract_parameters_regex(query, domain)
    
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk import pos_tag, ne_chunk
        from nltk.tree import Tree
        from nltk.chunk import RegexpParser
        
        logger.info(f"Extracting parameters with NLTK from: {query[:100]}...")
        
        # Tokenize sentences
        sentences = sent_tokenize(query)
        
        for sentence in sentences:
            # Tokenize words
            tokens = word_tokenize(sentence)
            
            # POS tagging
            tagged = pos_tag(tokens)
            
            # Define grammar for biomedical parameters
            grammar = r"""
                NP: {<DT|JJ|NN.*>+}          # Noun Phrase
                PARAM: {<NN.*><CD>}          # Noun followed by number (e.g., "pH 7")
                PARAM2: {<JJ><NN.*><CD>}     # Adjective + Noun + Number (e.g., "optimal temperature 30")
                VALUE: {<CD><NN.*>}          # Number followed by noun (e.g., "7 pH")
                RANGE: {<CD><IN|TO|CC><CD>}  # Number to/and number (e.g., "5 to 10")
                UNIT: {<CD><NN>}             # Number + unit noun
            """
            
            # Create chunk parser
            cp = RegexpParser(grammar)
            tree = cp.parse(tagged)
            
            # Helper function to extract from subtree
            def extract_from_subtree(subtree, label):
                if isinstance(subtree, Tree) and subtree.label() == label:
                    words = [word for word, tag in subtree.leaves()]
                    return words
                return None
            
            # Walk through tree and extract parameters
            for subtree in tree:
                # PARAM pattern: Noun + Number (e.g., "pH 7")
                if words := extract_from_subtree(subtree, "PARAM"):
                    if len(words) >= 2:
                        param_name = words[0]
                        try:
                            param_value = float(words[1])
                            unit = _infer_unit_from_word(param_name)
                            parameters[param_name] = {
                                "value": param_value,
                                "unit": unit,
                                "method": "nltk_param_pattern",
                                "confidence": 0.8,
                                "context": sentence
                            }
                        except ValueError:
                            pass
                
                # PARAM2 pattern: Adjective + Noun + Number (e.g., "optimal temperature 30")
                elif words := extract_from_subtree(subtree, "PARAM2"):
                    if len(words) >= 3:
                        param_name = f"{words[0]}_{words[1]}"
                        try:
                            param_value = float(words[2])
                            unit = _infer_unit_from_word(words[1])
                            parameters[param_name] = {
                                "value": param_value,
                                "unit": unit,
                                "method": "nltk_param2_pattern",
                                "confidence": 0.7,
                                "context": sentence
                            }
                        except ValueError:
                            pass
                
                # RANGE pattern: Number to/and Number (e.g., "5 to 10")
                elif words := extract_from_subtree(subtree, "RANGE"):
                    if len(words) >= 3:
                        try:
                            low = float(words[0])
                            high = float(words[2])
                            # Find nearby noun for parameter name
                            param_name = _find_nearby_noun(sentence, words[0])
                            if param_name:
                                key = f"{param_name}_range"
                                parameters[key] = {
                                    "value": [low, high],
                                    "unit": _infer_unit_from_word(param_name),
                                    "method": "nltk_range_pattern",
                                    "confidence": 0.9,
                                    "context": sentence
                                }
                        except ValueError:
                            pass
                
                # VALUE pattern: Number + Noun (e.g., "30 °C")
                elif words := extract_from_subtree(subtree, "VALUE"):
                    if len(words) >= 2:
                        try:
                            param_value = float(words[0])
                            param_name = words[1]
                            unit = _infer_unit_from_word(param_name)
                            parameters[param_name] = {
                                "value": param_value,
                                "unit": unit,
                                "method": "nltk_value_pattern",
                                "confidence": 0.75,
                                "context": sentence
                            }
                        except ValueError:
                            pass
        
        # Also run regex extraction as backup
        regex_params = _extract_with_regex(query)
        for key, value in regex_params.items():
            if key not in parameters:
                parameters[key] = value
        
        # Post-process parameters
        parameters = _post_process_parameters(parameters, query, domain)
        
        logger.info(f"✅ NLTK extraction found {len(parameters)} parameters")
        return parameters
        
    except Exception as e:
        logger.error(f"NLTK parameter extraction failed: {e}")
        # Fallback to regex extraction
        return await extract_parameters_regex(query, domain)

def _find_nearby_noun(sentence: str, number_word: str) -> str:
    """Find noun near a number in sentence using NLTK"""
    try:
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        
        # Find position of number
        for i, (word, tag) in enumerate(tagged):
            if word == number_word:
                # Look backward for noun
                for j in range(i-1, max(-1, i-4), -1):
                    if tagged[j][1].startswith('NN'):  # Noun
                        return tagged[j][0]
                # Look forward for noun
                for j in range(i+1, min(len(tagged), i+4)):
                    if tagged[j][1].startswith('NN'):  # Noun
                        return tagged[j][0]
    except Exception as e:
        logger.debug(f"_find_nearby_noun failed: {e}")
    return ""

def _extract_with_regex(sentence: str) -> Dict[str, Any]:
    """Regex-based parameter extraction as fallback"""
    params = {}
    
    # Enhanced regex patterns for biomedical parameters
    patterns = {
        'ph': [
            (r'pH\s*([\d\.]+)\s*[–\-]\s*([\d\.]+)', 'range', 'pH'),
            (r'pH\s*([\d\.]+)', 'single', 'pH'),
            (r'pH\s*(?:of|at|around)?\s*([\d\.]+)', 'single', 'pH'),
            (r'(\d+\.?\d*)\s*[-]?\s*pH', 'single', 'pH')
        ],
        'temperature': [
            (r'(\d+\.?\d*)\s*[–\-]\s*(\d+\.?\d*)\s*°?C', 'range', '°C'),
            (r'(\d+\.?\d*)\s*°?C', 'single', '°C'),
            (r'temperature\s*(?:of|at|around)?\s*(\d+\.?\d*)\s*°?C?', 'single', '°C'),
            (r'(\d+)\s*degrees?\s*(?:celsius|C)', 'single', '°C')
        ],
        'concentration': [
            (r'(\d+\.?\d*)\s*(mM|µM|nM|M|mg/mL|mg/L|g/L|%)', 'single'),
            (r'concentration\s*(?:of|at)?\s*(\d+\.?\d*)\s*(mM|µM|nM|M)', 'single'),
            (r'(\d+)\s*(?:mg|g|ml)\s*per', 'single', 'mg/mL')
        ],
        'time': [
            (r'(\d+\.?\d*)\s*[–\-]\s*(\d+\.?\d*)\s*(?:hours?|hrs?|minutes?|mins?)', 'range', 'hours'),
            (r'(\d+\.?\d*)\s*(?:hours?|hrs?|minutes?|mins?|days?)', 'single'),
            (r'incubation\s*(?:time|period)\s*(?:of|for)?\s*(\d+\.?\d*)\s*(?:hours?|hrs?)', 'single', 'hours')
        ],
        'agitation': [
            (r'(\d+\.?\d*)\s*[–\-]\s*(\d+\.?\d*)\s*rpm', 'range', 'rpm'),
            (r'(\d+)\s*rpm', 'single', 'rpm'),
            (r'agitation\s*(?:at|of)?\s*(\d+)\s*rpm', 'single', 'rpm')
        ],
        'volume': [
            (r'(\d+\.?\d*)\s*[–\-]\s*(\d+\.?\d*)\s*(?:ml|mL|µl|μl)', 'range'),
            (r'(\d+\.?\d*)\s*(?:ml|mL|µl|μl|L)', 'single'),
            (r'volume\s*(?:of)?\s*(\d+)\s*(?:ml|mL)', 'single', 'mL')
        ]
    }
    
    for param_name, param_patterns in patterns.items():
        for pattern in param_patterns:
            regex, pattern_type = pattern[0], pattern[1]
            default_unit = pattern[2] if len(pattern) > 2 else None
            
            matches = re.finditer(regex, sentence, re.IGNORECASE)
            for match in matches:
                try:
                    if pattern_type == 'range' and len(match.groups()) >= 2:
                        value = [float(match.group(1)), float(match.group(2))]
                    elif pattern_type == 'single':
                        value = float(match.group(1))
                    else:
                        continue
                    
                    # Extract unit from match or use default
                    unit_match = re.search(r'(mM|µM|nM|M|mg/mL|mg/L|g/L|%|°C|C|hours?|hrs?|minutes?|mins?|days?|rpm|ml|mL|µl|μl|L)',
                                         match.group(0), re.IGNORECASE)
                    unit = unit_match.group(1) if unit_match else default_unit
                    
                    # Create unique key
                    base_key = param_name
                    counter = 1
                    while f"{base_key}_{counter}" in params:
                        counter += 1
                    key = f"{base_key}_{counter}" if counter > 1 else base_key
                    
                    params[key] = {
                        "value": value,
                        "unit": unit or "",
                        "method": "regex",
                        "confidence": 0.85,
                        "context": sentence
                    }
                    
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Regex extraction failed for {pattern}: {e}")
    
    return params

def _infer_unit_from_word(word: str) -> str:
    """Infer unit from word"""
    word_lower = word.lower()
    
    unit_map = {
        'ph': 'pH',
        'acidity': 'pH',
        'alkalinity': 'pH',
        'temp': '°C',
        'temperature': '°C',
        'thermal': '°C',
        'conc': 'mM',
        'concentration': 'mM',
        'dose': 'mg/kg',
        'dosage': 'mg/kg',
        'time': 'hours',
        'duration': 'hours',
        'incubation': 'hours',
        'period': 'hours',
        'rpm': 'rpm',
        'speed': 'rpm',
        'agitation': 'rpm',
        'rotation': 'rpm',
        'volume': 'mL',
        'amount': 'mL',
        'weight': 'g',
        'mass': 'g',
        'pressure': 'kPa',
        'humidity': '%',
        'light': 'lux',
        'intensity': 'units',
        'frequency': 'Hz'
    }
    
    for key, unit in unit_map.items():
        if key in word_lower:
            return unit
    
    return ""

def _post_process_parameters(params: Dict[str, Any], query: str, domain: str) -> Dict[str, Any]:
    """Post-process extracted parameters"""
    if not params and domain == "biomed":
        # Add inferred parameters from query context
        query_lower = query.lower()
        
        inferred_params = {}
        
        # Check for common biomedical contexts
        if any(term in query_lower for term in ['yeast', 'fungi', 'microbial', 'bacterial']):
            inferred_params["organism"] = {
                "value": "yeast" if 'yeast' in query_lower else "microorganism",
                "unit": "",
                "method": "context_inferred",
                "confidence": 0.6,
                "context": query
            }
        
        if 'growth' in query_lower or 'biomass' in query_lower:
            inferred_params["growth_measurement"] = {
                "value": "OD600 or dry weight",
                "unit": "",
                "method": "context_inferred",
                "confidence": 0.5,
                "context": query
            }
        
        if 'enzyme' in query_lower or 'protein' in query_lower:
            inferred_params["biomolecule"] = {
                "value": "enzyme" if 'enzyme' in query_lower else "protein",
                "unit": "",
                "method": "context_inferred",
                "confidence": 0.7,
                "context": query
            }
        
        params.update(inferred_params)
    
    # Clean up parameter names
    cleaned_params = {}
    for key, value in params.items():
        # Clean key (remove special chars, spaces)
        clean_key = re.sub(r'[^\w]', '_', key.lower()).strip('_')
        if clean_key:
            # Ensure confidence is a float
            if 'confidence' in value:
                value['confidence'] = float(value['confidence'])
            cleaned_params[clean_key] = value
    
    return cleaned_params

# ========== MAIN EXTRACTION FUNCTION ==========

# In utils.py, replace the extract_parameters function:

async def extract_parameters(query: str, domain: str = "biomed") -> Dict[str, Any]:
    """
    Main parameter extraction function - uses comprehensive biomedical pipeline
    """
    logger.info(f"🔄 Starting parameter extraction for domain: {domain}")
    
    if domain == "biomed":
        try:
            # Use comprehensive biomedical extractor
            from core.biomed_parameter_extractor import extract_biomedical_parameters
            parameters = await extract_biomedical_parameters(query)
            
            # Remove metadata for cleaner output
            params_clean = {k: v for k, v in parameters.items() if not k.startswith('_')}
            
            logger.info(f"✅ Biomedical extraction: {len(params_clean)} parameters")
            return params_clean
            
        except Exception as e:
            logger.error(f"Biomedical extraction failed: {e}")
            # Fallback to NLTK
            return await extract_parameters_nltk(query, domain)
    
    else:
        # For non-biomedical domains, use NLTK
        return await extract_parameters_nltk(query, domain)

async def extract_parameters_regex(query: str, domain: str = "biomed") -> Dict[str, Any]:
    """Regex-based parameter extraction (fallback)"""
    params = _extract_with_regex(query)
    return _post_process_parameters(params, query, domain)

# ========== EXPLAINABILITY METHOD SELECTION ==========

def select_explainability_method(user_input: str, parameters: Dict[str, Any]) -> str:
    """Select explainability method based on query and parameters"""
    input_lower = user_input.lower()
    
    # Analyze query intent
    is_local = any(word in input_lower for word in ["specific", "this", "instance", "particular", "why this"])
    is_global = any(word in input_lower for word in ["overall", "general", "all", "compare", "ranking"])
    wants_both = any(word in input_lower for word in ["both", "all methods", "comprehensive"])
    
    if wants_both:
        return "both"
    
    # Few parameters -> LIME is better for local explanations
    if len(parameters) <= 3:
        return "lime"
    
    # Many parameters -> SHAP for global importance
    if len(parameters) > 5:
        return "shap"
    
    # Default based on query type
    if is_local:
        return "lime"
    elif is_global:
        return "shap"
    
    # Default: SHAP for biomedical (global patterns matter)
    return "shap"

# ========== CACHE FUNCTIONS ==========

def cache_set(key: str, value: Any, ttl: int = 3600) -> bool:
    """Simple file-based cache with numpy type conversion"""
    try:
        os.makedirs("cache", exist_ok=True)
        cache_file = f"cache/{hashlib.md5(key.encode()).hexdigest()}.json"
        
        # Convert numpy types before saving
        from decimal import Decimal
        import numpy as np
        
        def convert_for_cache(obj):
            if isinstance(obj, dict):
                return {k: convert_for_cache(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_cache(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        value_converted = convert_for_cache(value)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                "value": value_converted,
                "expires": time.time() + ttl,
                "created": time.time()
            }, f, ensure_ascii=False)
        return True
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")
        return False

def cache_get(key: str) -> Any:
    """Simple file-based cache get"""
    try:
        cache_file = f"cache/{hashlib.md5(key.encode()).hexdigest()}.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if time.time() < data.get("expires", 0):
                    return data.get("value")
                else:
                    os.remove(cache_file)
        return None
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
        return None

# ========== TEXT PROCESSING HELPERS ==========

def clean_text(text: str) -> str:
    """Clean text for processing"""
    if not text:
        return ""
    # Remove extra whitespace, normalize
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_parameters_for_display(parameters: Dict[str, Any]) -> str:
    """Format parameters for display"""
    if not parameters:
        return "No parameters extracted"
    
    lines = []
    for key, param in parameters.items():
        value = param.get("value", "")
        unit = param.get("unit", "")
        method = param.get("method", "unknown")
        confidence = param.get("confidence", 0)
        
        if isinstance(value, list):
            value_str = f"{value[0]} – {value[1]}"
        else:
            value_str = str(value)
        
        lines.append(f"- {key}: {value_str} {unit} ({method}, confidence: {confidence:.0%})")
    
    return "\n".join(lines)