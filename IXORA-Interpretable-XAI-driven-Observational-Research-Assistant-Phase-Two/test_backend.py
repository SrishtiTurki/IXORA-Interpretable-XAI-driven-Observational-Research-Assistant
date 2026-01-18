#!/usr/bin/env python3
"""
Backend Testing Framework for IXORA

Features:
- Domain-aware testing (biomed, computerscience, general)
- Comprehensive error handling and logging
- Performance and memory tracking
- Retry logic for flaky tests
- Detailed JSON reporting

Usage:
  python test_backend.py [domain]  # Test specific domain (default: all)
"""

import asyncio
import sys
import os
import json
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum, auto
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class Domain(Enum):
    BIOMED = "biomed"
    COMPUTER_SCIENCE = "computerscience"
    GENERAL = "general"

# Domain-specific test configurations
DOMAIN_CONFIGS = {
    Domain.BIOMED: {
        "test_query": "Analyze yeast biomass production at pH 5.5 and temperature 30°C",
        "params_query": "Statistically analyze the impact of pH and temperature?",
        "expected_params": ["ph", "temperature"],
        "test_params": {
            "ph": {"value": 5.5, "unit": "pH", "description": "Test pH"},
            "temperature": {"value": 30.0, "unit": "°C", "description": "Test temperature"}
        },
        "xml_tags": ["<enthusiasm>", "<explanation>", "<hypothesis>", "<followup>"]
    },
    Domain.COMPUTER_SCIENCE: {
        "test_query": "Analyze the performance impact of batch size on training a neural network",
        "params_query": "Compare training times for batch sizes 32, 64, and 128",
        "expected_params": ["batch_size"],
        "test_params": {
            "batch_size": {"value": 64, "unit": "samples", "description": "Training batch size"},
            "learning_rate": {"value": 0.001, "unit": "", "description": "Optimizer learning rate"}
        },
        "xml_tags": ["<analysis>", "<recommendation>", "<technical_details>", "<considerations>"]
    },
    Domain.GENERAL: {
        "test_query": "Explain the key factors affecting climate change",
        "params_query": "What are the main causes of climate change?",
        "expected_params": [],
        "test_params": {},
        "xml_tags": ["<overview>", "<key_points>", "<implications>", "<sources>"]
    }
}

# Test results structure
test_results = {
    "start_time": "",
    "end_time": "",
    "system_info": {},
    "domains": {},
    "summary": {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "success_rate": 0.0
    }
}

def get_memory_usage() -> Dict[str, float]:
    """Get current process memory usage in MB"""
    process = psutil.Process()
    mem = process.memory_info()
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024)
    }

async def measure_execution(func, *args, **kwargs) -> Tuple[Any, float, Dict[str, float]]:
    """Measure execution time and memory usage of a function"""
    start_time = time.time()
    start_mem = get_memory_usage()
    
    try:
        result = await func(*args, **kwargs)
        end_mem = get_memory_usage()
        end_time = time.time()
        
        return (
            result,
            end_time - start_time,
            {
                "memory_used_mb": end_mem["rss_mb"] - start_mem["rss_mb"],
                "peak_memory_mb": end_mem["rss_mb"]
            }
        )
    except Exception as e:
        end_mem = get_memory_usage()
        end_time = time.time()
        raise e

async def test_mistral_api(domain: Domain, max_retries: int = 2) -> Dict[str, Any]:
    """Test Mistral API with domain-specific prompts.
    
    Args:
        domain: The domain to test (biomed, computerscience, or general)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dict containing test results and metrics
    """
    test_name = f"Mistral API - {domain.value}"
    logger.info(f"\n{'='*80}\nTEST: {test_name}\n{'='*80}")
    
    from core.mistral import call_mistral_api
    
    domain_config = DOMAIN_CONFIGS[domain]
    prompt = f"You are a {domain.value} research assistant. {domain_config['test_query']}"
    
    result = {
        "name": test_name,
        "status": "failed",
        "attempts": 0,
        "error": None,
        "metrics": {},
        "response": None
    }
    
    for attempt in range(max_retries + 1):
        result["attempts"] += 1
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
            
            # Measure execution time and memory
            start_time = time.time()
            start_mem = get_memory_usage()
            
            response = await call_mistral_api(
                prompt,
                max_tokens=300,
                temperature=0.3
            )
            
            end_mem = get_memory_usage()
            end_time = time.time()
            
            # Update metrics
            result["metrics"].update({
                "execution_time_sec": end_time - start_time,
                "memory_used_mb": end_mem["rss_mb"] - start_mem["rss_mb"],
                "peak_memory_mb": end_mem["rss_mb"],
                "response_length": len(response)
            })
            
            # Store truncated response for logging
            result["response"] = response[:500] + "..." if len(response) > 500 else response
            
            # Validate response
            if not response or len(response) < 10:
                raise ValueError("Response too short or empty")
                
            # Check for domain-specific XML tags
            missing_tags = [
                tag for tag in domain_config["xml_tags"]
                if tag not in response
            ]
            
            if missing_tags:
                logger.warning(f"Missing expected XML tags: {', '.join(missing_tags)}")
            
            # Test passed
            result["status"] = "passed"
            result["missing_xml_tags"] = missing_tags
            logger.info(f"✅ {test_name} PASSED")
            logger.info(f"   Response length: {len(response)} chars")
            logger.info(f"   Time: {result['metrics']['execution_time_sec']:.2f}s")
            logger.info(f"   Memory: {result['metrics']['memory_used_mb']:.1f}MB")
            
            return result
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            logger.error(error_msg)
            result["error"] = str(e)
            
            if attempt < max_retries:
                retry_delay = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ {test_name} FAILED after {max_retries + 1} attempts")
                logger.error(f"Last error: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    logger.error(traceback.format_exc())
    
    return result

async def test_mistral_generate(domain: Domain, max_retries: int = 2) -> Dict[str, Any]:
    """Test generate_with_mistral with domain-specific prompts and CoT.
    
    Args:
        domain: The domain to test (biomed, computerscience, or general)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dict containing test results and metrics
    """
    test_name = f"Mistral Generate - {domain.value}"
    logger.info(f"\n{'='*80}\nTEST: {test_name}\n{'='*80}")
    
    from core.mistral import generate_with_mistral
    
    domain_config = DOMAIN_CONFIGS[domain]
    system_prompt = f"You are a {domain.value} research assistant. Think step by step."
    user_prompt = f"{domain_config['test_query']} Provide a detailed response with reasoning."
    
    result = {
        "name": test_name,
        "status": "failed",
        "attempts": 0,
        "error": None,
        "metrics": {
            "response_length": 0,
            "cot_steps": 0,
            "execution_time_sec": 0,
            "memory_used_mb": 0,
            "peak_memory_mb": 0
        },
        "response": None,
        "cot_preview": None
    }
    
    for attempt in range(max_retries + 1):
        result["attempts"] += 1
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
            
            # Measure execution time and memory
            start_time = time.time()
            start_mem = get_memory_usage()
            
            response, cot = await generate_with_mistral(
                user_prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            end_mem = get_memory_usage()
            end_time = time.time()
            
            # Update metrics
            result["metrics"].update({
                "execution_time_sec": end_time - start_time,
                "memory_used_mb": end_mem["rss_mb"] - start_mem["rss_mb"],
                "peak_memory_mb": end_mem["rss_mb"],
                "response_length": len(response),
                "cot_steps": len(cot) if cot else 0
            })
            
            # Store truncated response and CoT for logging
            result["response"] = response[:500] + "..." if len(response) > 500 else response
            result["cot_preview"] = [
                step[:200] + ("..." if len(step) > 200 else "") 
                for step in cot[:3]  # First 3 CoT steps
            ] if cot else []
            
            # Validate response
            if not response or len(response) < 50:
                raise ValueError("Response too short or empty")
                
            if not cot or len(cot) < 1:
                logger.warning("No Chain-of-Thought (CoT) steps were generated")
            
            # Check for domain-specific XML tags
            missing_tags = [
                tag for tag in domain_config["xml_tags"]
                if tag not in response
            ]
            
            if missing_tags:
                logger.warning(f"Missing expected XML tags: {', '.join(missing_tags)}")
            
            # Test passed
            result["status"] = "passed"
            result["missing_xml_tags"] = missing_tags
            
            logger.info(f"✅ {test_name} PASSED")
            logger.info(f"   Response length: {len(response)} chars")
            logger.info(f"   CoT steps: {len(cot) if cot else 0}")
            logger.info(f"   Time: {result['metrics']['execution_time_sec']:.2f}s")
            logger.info(f"   Memory: {result['metrics']['memory_used_mb']:.1f}MB")
            
            return result
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            logger.error(error_msg)
            result["error"] = str(e)
            
            if attempt < max_retries:
                retry_delay = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ {test_name} FAILED after {max_retries + 1} attempts")
                logger.error(f"Last error: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    logger.error(traceback.format_exc())
    
    return result

async def test_parameter_extraction(domain: Domain, max_retries: int = 2) -> Dict[str, Any]:
    """Test parameter extraction with domain-specific queries.
    
    Args:
        domain: The domain to test (biomed, computerscience, or general)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dict containing test results and metrics
    """
    test_name = f"Parameter Extraction - {domain.value}"
    logger.info(f"\n{'='*80}\nTEST: {test_name}\n{'='*80}")
    
    from core.utils import extract_parameters
    
    domain_config = DOMAIN_CONFIGS[domain]
    test_query = domain_config["params_query"]
    expected_params = domain_config.get("expected_params", [])
    
    result = {
        "name": test_name,
        "status": "failed",
        "attempts": 0,
        "error": None,
        "metrics": {
            "execution_time_sec": 0,
            "memory_used_mb": 0,
            "peak_memory_mb": 0,
            "parameters_found": 0,
            "expected_parameters": len(expected_params),
            "missing_parameters": []
        },
        "extracted_parameters": {}
    }
    
    for attempt in range(max_retries + 1):
        result["attempts"] += 1
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
            logger.info(f"Query: {test_query}")
            
            # Measure execution time and memory
            start_time = time.time()
            start_mem = get_memory_usage()
            
            # Extract parameters
            params = await extract_parameters(test_query, domain.value)
            
            end_mem = get_memory_usage()
            end_time = time.time()
            
            # Update metrics
            result["metrics"].update({
                "execution_time_sec": end_time - start_time,
                "memory_used_mb": end_mem["rss_mb"] - start_mem["rss_mb"],
                "peak_memory_mb": end_mem["rss_mb"],
                "parameters_found": len(params),
                "missing_parameters": [
                    param for param in expected_params 
                    if param not in params
                ]
            })
            
            # Store extracted parameters
            result["extracted_parameters"] = params
            
            # Validate results
            if not params:
                raise ValueError("No parameters extracted")
                
            # Check for missing expected parameters
            missing_expected = [
                param for param in expected_params 
                if param not in params
            ]
            
            if missing_expected:
                logger.warning(f"Missing expected parameters: {', '.join(missing_expected)}")
            
            # Test passed
            result["status"] = "passed"
            
            logger.info(f"✅ {test_name} PASSED")
            logger.info(f"   Extracted {len(params)} parameters")
            if params:
                logger.info("   Sample parameters:")
                for i, (key, val) in enumerate(params.items()):
                    if i >= 3:  # Show first 3 parameters
                        logger.info(f"      ... and {len(params) - 3} more")
                        break
                    logger.info(f"      {key}: {val}")
            logger.info(f"   Time: {result['metrics']['execution_time_sec']:.2f}s")
            logger.info(f"   Memory: {result['metrics']['memory_used_mb']:.1f}MB")
            
            return result
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            logger.error(error_msg)
            result["error"] = str(e)
            
            if attempt < max_retries:
                retry_delay = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ {test_name} FAILED after {max_retries + 1} attempts")
                logger.error(f"Last error: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    logger.error(traceback.format_exc())
    
    return result

async def test_analytics(domain: Domain, max_retries: int = 2) -> Dict[str, Any]:
    """Test analytics pipeline with domain-specific parameters.
    
    Args:
        domain: The domain to test (biomed, computerscience, or general)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dict containing test results and metrics
    """
    test_name = f"Analytics Pipeline - {domain.value}"
    logger.info(f"\n{'='*80}\nTEST: {test_name}\n{'='*80}")
    
    from core.analytics import run_comprehensive_analytics
    
    domain_config = DOMAIN_CONFIGS[domain]
    test_query = f"Analyze {domain.value} data with the following parameters"
    
    result = {
        "name": test_name,
        "status": "failed",
        "attempts": 0,
        "error": None,
        "metrics": {
            "execution_time_sec": 0,
            "memory_used_mb": 0,
            "peak_memory_mb": 0,
            "components_run": 0,
            "components_total": 3  # explainability, causal, optimization
        },
        "components": {}
    }
    
    for attempt in range(max_retries + 1):
        result["attempts"] += 1
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
            logger.info(f"Testing with parameters: {domain_config['test_params']}")
            
            # Measure execution time and memory
            start_time = time.time()
            start_mem = get_memory_usage()
            
            # Run analytics
            analytics_result = await run_comprehensive_analytics(
                test_query,
                domain_config["test_params"],
                domain.value
            )
            
            end_mem = get_memory_usage()
            end_time = time.time()
            
            # Update metrics
            execution_time = end_time - start_time
            memory_used = end_mem["rss_mb"] - start_mem["rss_mb"]
            
            # Check which components were executed successfully
            components = {}
            component_checks = {
                "explainability": {
                    "status": "explainability" in analytics_result,
                    "shap": analytics_result.get("explainability", {}).get("shap_importance") is not None
                },
                "causal": {
                    "status": "causal" in analytics_result,
                    "ate": analytics_result.get("causal", {}).get("ate") is not None
                },
                "optimization": {
                    "status": "optimized" in analytics_result,
                    "values": analytics_result.get("optimized", {}).get("optimized_values") is not None
                }
            }
            
            # Count successful components
            components_run = sum(1 for comp in component_checks.values() if comp["status"])
            
            result["metrics"].update({
                "execution_time_sec": execution_time,
                "memory_used_mb": memory_used,
                "peak_memory_mb": end_mem["rss_mb"],
                "components_run": components_run,
                "components_total": len(component_checks)
            })
            
            # Store component status
            for comp_name, checks in component_checks.items():
                components[comp_name] = {
                    "status": "passed" if checks["status"] else "failed",
                    "sub_checks": {
                        k: "passed" if v else "failed"
                        for k, v in checks.items() if k != "status"
                    }
                }
            
            result["components"] = components
            
            # Log component status
            logger.info("Component Status:")
            for comp_name, comp_data in components.items():
                status = "✅" if comp_data["status"] == "passed" else "❌"
                logger.info(f"  {status} {comp_name}")
                for sub_check, sub_status in comp_data.get("sub_checks", {}).items():
                    sub_status_icon = "✓" if sub_status == "passed" else "✗"
                    logger.info(f"    {sub_status_icon} {sub_check}")
            
            # Determine overall test status
            if components_run == 0:
                raise ValueError("No analytics components ran successfully")
            
            result["status"] = "passed"
            
            logger.info(f"✅ {test_name} COMPLETED")
            logger.info(f"   Components: {components_run}/{len(component_checks)}")
            logger.info(f"   Time: {execution_time:.2f}s")
            logger.info(f"   Memory: {memory_used:.1f}MB")
            
            return result
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            logger.error(error_msg)
            result["error"] = str(e)
            
            if attempt < max_retries:
                retry_delay = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ {test_name} FAILED after {max_retries + 1} attempts")
                logger.error(f"Last error: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    logger.error(traceback.format_exc())
    
    return result

async def test_full_pipeline(domain: Domain, max_retries: int = 2) -> Dict[str, Any]:
    """Test the full IXORA pipeline end-to-end with domain-specific queries.
    
    Args:
        domain: The domain to test (biomed, computerscience, or general)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dict containing test results and metrics
    """
    test_name = f"Full Pipeline - {domain.value}"
    logger.info(f"\n{'='*80}\nTEST: {test_name}\n{'='*80}")
    
    from core.langgraph import invoke_graph
    
    domain_config = DOMAIN_CONFIGS[domain]
    test_query = domain_config["test_query"]
    
    result = {
        "name": test_name,
        "status": "failed",
        "attempts": 0,
        "error": None,
        "metrics": {
            "execution_time_sec": 0,
            "memory_used_mb": 0,
            "peak_memory_mb": 0,
            "response_length": 0,
            "reasoning_steps": 0,
            "trace_steps": 0,
            "missing_xml_tags": []
        },
        "response_preview": None,
        "reasoning_preview": [],
        "trace_preview": []
    }
    
    for attempt in range(max_retries + 1):
        result["attempts"] += 1
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
            logger.info(f"Query: {test_query}")
            
            # Measure execution time and memory
            start_time = time.time()
            start_mem = get_memory_usage()
            
            # Run the full pipeline
            pipeline_result = await invoke_graph(
                test_query,
                domain=domain.value,
                task_type="analyze"
            )
            
            end_mem = get_memory_usage()
            end_time = time.time()
            
            # Extract components
            response = pipeline_result.get("final_response", "")
            reasoning_log = pipeline_result.get("reasoning_log", [])
            trace = pipeline_result.get("trace", [])
            
            # Check for domain-specific XML tags
            missing_tags = [
                tag for tag in domain_config["xml_tags"]
                if tag not in response
            ]
            
            # Update metrics
            execution_time = end_time - start_time
            memory_used = end_mem["rss_mb"] - start_mem["rss_mb"]
            
            result["metrics"].update({
                "execution_time_sec": execution_time,
                "memory_used_mb": memory_used,
                "peak_memory_mb": end_mem["rss_mb"],
                "response_length": len(response),
                "reasoning_steps": len(reasoning_log),
                "trace_steps": len(trace),
                "missing_xml_tags": missing_tags
            })
            
            # Store preview data
            result["response_preview"] = response[:500] + "..." if len(response) > 500 else response
            result["reasoning_preview"] = [
                step[:200] + ("..." if len(step) > 200 else "")
                for step in reasoning_log[:3]  # First 3 reasoning steps
            ]
            result["trace_preview"] = [
                f"{step.get('step', '?')}: {step.get('reasoning', '')[:100]}"
                for step in trace[:3]  # First 3 trace steps
            ]
            
            # Validate response
            if not response or len(response) < 100:
                raise ValueError(f"Response too short ({len(response)} chars)")
                
            if missing_tags:
                logger.warning(f"Missing expected XML tags: {', '.join(missing_tags)}")
            
            # Test passed
            result["status"] = "passed"
            
            logger.info(f"✅ {test_name} COMPLETED")
            logger.info(f"   Response length: {len(response)} chars")
            logger.info(f"   Reasoning steps: {len(reasoning_log)}")
            logger.info(f"   Trace steps: {len(trace)}")
            logger.info(f"   Time: {execution_time:.2f}s")
            logger.info(f"   Memory: {memory_used:.1f}MB")
            
            return result
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            logger.error(error_msg)
            result["error"] = str(e)
            
            if attempt < max_retries:
                retry_delay = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ {test_name} FAILED after {max_retries + 1} attempts")
                logger.error(f"Last error: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    logger.error(traceback.format_exc())
    
    return result

def get_system_info() -> Dict[str, Any]:
    """Collect system information for the test report."""
    import platform
    import psutil
    import sys
    
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
            "available_memory_gb": round(psutil.virtual_memory().available / (1024 ** 3), 1),
        },
        "environment": {
            "MISTRAL_API_KEY": "SET" if os.getenv("MISTRAL_API_KEY") else "NOT SET",
            "MISTRAL_USE_API": os.getenv("MISTRAL_USE_API", "not set"),
            "PYTHONPATH": os.getenv("PYTHONPATH", "not set"),
            "CONDA_DEFAULT_ENV": os.getenv("CONDA_DEFAULT_ENV", "not in conda env")
        }
    }

def save_test_report(results: Dict[str, Any], output_dir: str = "test_reports") -> str:
    """Save test results to a JSON file with timestamp.
    
    Args:
        results: Dictionary containing test results
        output_dir: Directory to save the report (will be created if it doesn't exist)
        
    Returns:
        Path to the saved report file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"ixora_test_report_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save results to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return filepath

async def run_tests_for_domain(domain: Domain) -> Dict[str, Any]:
    """Run all tests for a specific domain.
    
    Args:
        domain: The domain to test
        
    Returns:
        Dictionary containing test results for the domain
    """
    domain_results = {}
    
    # Run each test with error handling
    tests = [
        ("mistral_api", test_mistral_api),
        ("mistral_generate", test_mistral_generate),
        ("parameter_extraction", test_parameter_extraction),
        ("analytics", test_analytics),
        ("full_pipeline", test_full_pipeline)
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func(domain)
            domain_results[test_name] = result
            
            # Add a small delay between tests
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error running {test_name} for {domain.value}: {e}")
            domain_results[test_name] = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    return domain_results

async def main():
    """Main function to run all tests and generate reports."""
    # Initialize test results
    start_time = time.time()
    test_results = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "system_info": get_system_info(),
        "domains": {},
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "success_rate": 0.0
        }
    }
    
    # Log test start
    logger.info("\n" + "#" * 80)
    logger.info("# IXORA BACKEND TEST SUITE")
    logger.info("# " + time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    logger.info("#" * 80 + "\n")
    
    # Log system info
    logger.info("System Information:")
    for key, value in test_results["system_info"]["system"].items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Log environment variables
    logger.info("\nEnvironment:")
    for key, value in test_results["system_info"]["environment"].items():
        logger.info(f"  {key}: {value}")
    
    # Determine which domains to test
    domains_to_test = list(Domain)
    if len(sys.argv) > 1:
        domain_arg = sys.argv[1].lower()
        domains_to_test = [d for d in domains_to_test if d.value.lower() == domain_arg]
        if not domains_to_test:
            logger.error(f"Invalid domain: {domain_arg}. Available domains: {[d.value for d in Domain]}")
            return False
    
    # Run tests for each domain
    for domain in domains_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING DOMAIN: {domain.value.upper()}")
        logger.info("="*80)
        
        domain_results = await run_tests_for_domain(domain)
        test_results["domains"][domain.value] = domain_results
    
    # Calculate summary statistics
    total_tests = 0
    passed_tests = 0
    
    for domain, tests in test_results["domains"].items():
        for test_name, test_result in tests.items():
            total_tests += 1
            if test_result.get("status") == "passed":
                passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    test_results["summary"].update({
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": total_tests - passed_tests,
        "success_rate": round(success_rate, 2)
    })
    
    # Add end time and duration
    test_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    test_results["duration_seconds"] = round(time.time() - start_time, 2)
    
    # Save test report
    report_path = save_test_report(test_results)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for domain, tests in test_results["domains"].items():
        domain_passed = sum(1 for t in tests.values() if t.get("status") == "passed")
        domain_total = len(tests)
        logger.info(f"{domain.upper()}: {domain_passed}/{domain_total} tests passed")
        
        for test_name, test_result in tests.items():
            status = "✅ PASSED" if test_result.get("status") == "passed" else "❌ FAILED"
            logger.info(f"  {status} - {test_name}")
            
            # Log additional test details
            if "metrics" in test_result:
                metrics = test_result["metrics"]
                logger.info(f"    Time: {metrics.get('execution_time_sec', 0):.2f}s")
                logger.info(f"    Memory: {metrics.get('memory_used_mb', 0):.1f}MB")
            
            if test_result.get("status") != "passed" and "error" in test_result:
                logger.error(f"    Error: {test_result['error']}")
    
    # Print final summary
    logger.info("\n" + "="*80)
    if success_rate == 100:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.warning(f"⚠️  {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    logger.info(f"\nTest report saved to: {os.path.abspath(report_path)}")
    logger.info("="*80)
    
    return success_rate == 100

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.error("\nTest execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)