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
    """
    Modern intent detection with three clear paths:
    - meta           → obvious chit-chat, greetings, meta questions
    - research_full  → optimization, parameters, effects, experiments, comparisons
    - explanatory    → definitions, explanations, background, "what is", "how does it work"
    """
    if not query:
        return "meta"

    query_lower = query.lower().strip()

    # ──────────────────────────────────────────────
    # 1. Very clear meta / chit-chat (small group)
    # ──────────────────────────────────────────────
    meta_keywords = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "how r u", "who are you", "what are you", "what can you do",
        "thanks", "thank you", "thx", "ty", "ok", "okay", "alright", "got it",
        "bye", "goodbye", "see you", "tell me a joke", "joke", "funny story",
        "what time is it", "weather", "date today"
    ]

    if len(query.split()) < 8 and any(kw in query_lower for kw in meta_keywords):
        return "meta"

    # ──────────────────────────────────────────────
    # 2. Strong signals → full research / optimization / experimental pipeline
    # ──────────────────────────────────────────────
    strong_research_markers = [
        "effect of", "impact of", "influence of", "optimize", "maximi", "minimi",
        "best", "optimal", "better", "improve", "increase", "decrease",
        "investigat", "study", "experiment", "trial", "compare", "versus", "vs",
        "relationship", "correlation", "caus", "significant", "p-value", "anova",
        "parameter", "condition", "factor", "variable", "batch size", "learning rate",
        "lr", "epochs", "dropout", "optimizer", "adam", "sgd", "accuracy", "loss",
        "ph", "temperature", "concentration", "rpm", "incubat", "agitat", "dose",
        "dosage", "enzyme activity", "growth rate", "yield", "biomass", "fermentation"
    ]

    if any(marker in query_lower for marker in strong_research_markers):
        return "research_full"

    # Number + scientific unit → almost always full research
    number_pattern = r'\d+\.?\d*'
    unit_pattern = r'(ph|°?c|°?f|m[m]?|µ|g|mg|g|l|ml|rpm|hr|min|sec|day|h|k|µg|µM|mM|nM|xg|%)'
    if re.search(number_pattern, query_lower) and re.search(unit_pattern, query_lower):
        return "research_full"

    # Imperative + analysis/optimization verbs (using NLTK if available)
    if _nltk_ready:
        try:
            from nltk.tokenize import word_tokenize
            from nltk import pos_tag

            tokens = word_tokenize(query_lower)
            tagged = pos_tag(tokens)
            pos_string = ' '.join(tag for _, tag in tagged[:12])

            analyze_verbs = ['analyze', 'calculate', 'compute', 'optimize', 'compare', 'evaluate', 'predict']
            if 'VB' in pos_string and any(v in tokens for v in analyze_verbs):
                return "research_full"
        except:
            pass

    # ──────────────────────────────────────────────
    # 3. Questions that look explanatory / conceptual → lightweight path
    # ──────────────────────────────────────────────
    explanatory_starters = [
        "what is", "what are", "what does", "explain", "define", "describe",
        "how does", "how do", "how can", "how to", "tell me about", "what happens if",
        "difference between", "why is", "why does", "can you explain"
    ]

    if any(query_lower.startswith(s) for s in explanatory_starters) or "?" in query:
        return "explanatory"

    # ──────────────────────────────────────────────
    # 4. Domain-specific hints
    # ──────────────────────────────────────────────
    if domain == "biomed":
        biomed_terms = ["enzyme", "protein", "cell", "growth", "culture", "biomass", "yeast", "bacteria"]
        if any(t in query_lower for t in biomed_terms):
            return "explanatory"  # most biomed background questions stay light unless strong optimization signal
    elif domain == "cs":
        cs_terms = ["algorithm", "complexity", "big o", "runtime", "throughput", "latency"]
        if any(t in query_lower for t in cs_terms):
            return "explanatory"

    # ──────────────────────────────────────────────
    # 5. Default → explanatory (fail open to lighter path)
    # ──────────────────────────────────────────────
    return "explanatory"

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
    Main parameter extraction function - domain-aware
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
    elif domain == "cs":
        try:
            from core.cs_parameter_extractor import extract_cs_parameters
            parameters = await extract_cs_parameters(query)
            params_clean = {k: v for k, v in parameters.items() if not k.startswith('_')}
            logger.info(f"✅ CS extraction: {len(params_clean)} parameters")
            return params_clean
        except Exception as e:
            logger.error(f"CS extraction failed: {e}")
            return await extract_parameters_nltk(query, domain)
    else:
        # For other domains, use NLTK
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

# core/utils.py 

def load_session_state(session_id: str) -> dict:
    """Load session state from cache"""
    state = cache_get(f"session:{session_id}")
    if state:
        logger.info(f"Loaded session state for {session_id}")
    else:
        logger.debug(f"No session state found for {session_id}")
    return state or {}


def save_session_state(session_id: str, state: dict):
    """Save complete state for on-demand feature access"""
    cache_set(f"session:{session_id}", state)
    logger.info(f"Saved session state for {session_id}")