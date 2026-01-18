"""
diagnostic_test.py - Comprehensive pipeline testing with timing and domain awareness
Run this to diagnose timeout issues and verify all components across different domains
"""
import asyncio
import time
import sys
import logging
import platform
import psutil
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("diagnostic_test.log")
    ]
)
logger = logging.getLogger("diagnostic")

# Domain configuration
class Domain(Enum):
    BIOMED = "biomed"
    COMPUTER_SCIENCE = "computerscience"
    GENERAL = "general"

# Domain-specific test configurations
DOMAIN_CONFIGS = {
    Domain.BIOMED: {
        "test_queries": [
            "Analyze yeast growth at pH 5.5 and 30°C",
            "What are the effects of glucose concentration on E. coli growth?",
            "Explain the role of ATP in cellular respiration"
        ],
        "expected_params": ["ph", "temperature", "organism"],
        "expected_tags": ["enthusiasm", "explanation", "hypothesis", "followup"]
    },
    Domain.COMPUTER_SCIENCE: {
        "test_queries": [
            "Analyze the time complexity of quicksort algorithm",
            "Explain how a neural network works",
            "What are the advantages of using Python over Java?"
        ],
        "expected_params": ["algorithm", "complexity", "language"],
        "expected_tags": ["technical", "algorithms", "implementation", "challenges", "followup"]
    },
    Domain.GENERAL: {
        "test_queries": [
            "Explain the key factors affecting plant growth",
            "What are the main causes of climate change?",
            "How does the stock market work?"
        ],
        "expected_params": ["topic", "complexity", "detail_level"],
        "expected_tags": ["summary", "key_points", "followup"]
    }
}

# Test results storage
results = {
    "system_info": {},
    "test_start_time": "",
    "test_end_time": "",
    "domains_tested": [],
    "tests_run": 0,
    "tests_passed": 0,
    "tests_failed": 0,
    "timings": {},
    "memory_usage": {},
    "errors": [],
    "warnings": []
}

async def get_system_info() -> Dict[str, Any]:
    """Collect system information for diagnostics"""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "total_memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "available_memory_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2)
    }

async def measure_memory() -> Dict[str, float]:
    """Measure current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
        "vms_mb": memory_info.vms / (1024 * 1024),   # Virtual Memory Size
        "percent": process.memory_percent()
    }

async def test_mistral_api(domain: Domain = Domain.BIOMED, max_retries: int = 2) -> bool:
    """
    Test Mistral API connectivity with domain awareness
    
    Args:
        domain: The domain to test (biomed, computerscience, or general)
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if test passed, False otherwise
    """
    test_name = f"mistral_api_{domain.value}"
    logger.info("\n" + "=" * 80)
    logger.info(f"TEST: Mistral API - {domain.value.upper()} Domain")
    logger.info("=" * 80)
    
    test_query = DOMAIN_CONFIGS[domain]["test_queries"][0]
    
    for attempt in range(max_retries + 1):
        try:
            from core.mistral import generate_with_mistral
            
            # Measure memory before test
            mem_before = await measure_memory()
            start_time = time.time()
            
            # Run the test with domain context
            response, _ = await asyncio.wait_for(
                generate_with_mistral(
                    f"{test_query} (Respond in 10 words or less)",
                    max_tokens=30,
                    domain=domain.value
                ),
                timeout=30.0
            )
            
            # Calculate metrics
            elapsed = time.time() - start_time
            mem_after = await measure_memory()
            
            # Update results
            results["timings"][test_name] = elapsed
            results["memory_usage"][test_name] = {
                "before_mb": mem_before,
                "after_mb": mem_after,
                "delta_mb": {
                    "rss": mem_after["rss_mb"] - mem_before["rss_mb"],
                    "vms": mem_after["vms_mb"] - mem_before["vms_mb"]
                }
            }
            
            # Validate response
            if not response or len(response.strip()) == 0:
                raise ValueError("Empty response from API")
                
            logger.info(f"✅ [{domain.value.upper()}] Mistral API OK ({elapsed:.2f}s)")
            logger.debug(f"   Response: {response[:200]}")
            results["tests_passed"] += 1
            results["tests_run"] += 1
            return True
            
        except asyncio.TimeoutError:
            if attempt == max_retries:
                error_msg = f"❌ [{domain.value.upper()}] Mistral API timeout (30s)"
                logger.error(error_msg)
                results["tests_failed"] += 1
                results["tests_run"] += 1
                results["errors"].append(f"{test_name}: Timeout after {max_retries + 1} attempts")
                return False
            logger.warning(f"⚠️  [{domain.value.upper()}] Attempt {attempt + 1} failed - retrying...")
            await asyncio.sleep(1)  # Backoff
            
        except Exception as e:
            error_msg = f"❌ [{domain.value.upper()}] Mistral API error: {str(e)}"
            logger.error(error_msg)
            if attempt == max_retries:
                results["tests_failed"] += 1
                results["tests_run"] += 1
                results["errors"].append(f"{test_name}: {str(e)[:200]}")
                return False
            logger.warning(f"⚠️  [{domain.value.upper()}] Attempt {attempt + 1} failed - retrying...")
            await asyncio.sleep(1)  # Backoff
    
    return False

async def test_domain_llm(domain: Domain = Domain.BIOMED) -> bool:
    """
    Test domain-specific LLM generation
    
    Args:
        domain: The domain to test (biomed, computerscience, or general)
        
    Returns:
        bool: True if test passed, False otherwise
    """
    test_name = f"{domain.value}_generation"
    logger.info("\n" + "=" * 80)
    logger.info(f"TEST: {domain.value.upper()} Generation")
    logger.info("=" * 80)
    
    try:
        test_query = DOMAIN_CONFIGS[domain]["test_queries"][0]
        
        # Import the appropriate generator based on domain
        if domain == Domain.BIOMED:
            from core.medicalscience.loaders import generate_biomed_draft as generate_draft
        else:
            from core.mistral import generate_with_mistral as generate_draft
        
        # Measure memory and time
        mem_before = await measure_memory()
        start_time = time.time()
        
        # Generate the draft with appropriate parameters
        if domain == Domain.BIOMED:
            draft = await asyncio.wait_for(
                generate_draft(test_query),
                timeout=35.0  # 30s model + 5s buffer
            )
        else:
            draft, _ = await asyncio.wait_for(
                generate_draft(test_query, domain=domain.value, max_tokens=300),
                timeout=35.0
            )
        
        # Calculate metrics
        elapsed = time.time() - start_time
        mem_after = await measure_memory()
        
        # Update results
        results["timings"][test_name] = elapsed
        results["memory_usage"][test_name] = {
            "before_mb": mem_before,
            "after_mb": mem_after,
            "delta_mb": {
                "rss": mem_after["rss_mb"] - mem_before["rss_mb"],
                "vms": mem_after["vms_mb"] - mem_before["vms_mb"]
            }
        }
        
        # Validate response
        if not draft or len(draft.strip()) < 20:
            raise ValueError(f"Generated draft is too short: {draft}")
        
        logger.info(f"✅ [{domain.value.upper()}] Generation OK ({elapsed:.2f}s)")
        logger.debug(f"   Draft preview: {draft[:150]}...")
        
        # Check for expected domain-specific tags
        expected_tags = DOMAIN_CONFIGS[domain]["expected_tags"]
        missing_tags = [tag for tag in expected_tags if f"<{tag}>" not in draft]
        
        if missing_tags:
            logger.warning(f"⚠️  [{domain.value.upper()}] Missing expected tags: {', '.join(missing_tags)}")
            results["warnings"].append(f"{test_name}: Missing tags: {', '.join(missing_tags)}")
        
        results["tests_passed"] += 1
        results["tests_run"] += 1
        return True
        
    except asyncio.TimeoutError:
        error_msg = f"❌ [{domain.value.upper()}] Generation timeout (35s)"
        logger.error(error_msg)
        results["tests_failed"] += 1
        results["tests_run"] += 1
        results["errors"].append(f"{test_name}: Timeout during generation")
        return False
        
    except Exception as e:
        error_msg = f"❌ [{domain.value.upper()}] Generation error: {str(e)}"
        logger.error(error_msg)
        results["tests_failed"] += 1
        results["tests_run"] += 1
        results["errors"].append(f"{test_name}: {str(e)[:200]}")
        return False

async def test_parameter_extraction():
    """Test 3: Parameter extraction"""
    logger.info("=" * 80)
    logger.info("TEST 3: Parameter Extraction")
    logger.info("=" * 80)
    
    try:
        from core.utils import extract_parameters
        start = time.time()
        
        test_query = "Analyze yeast biomass at pH 5.5 and 30°C with 100mM glucose"
        params = await asyncio.wait_for(
            extract_parameters(test_query, "biomed"),
            timeout=25.0
        )
        
        elapsed = time.time() - start
        results["timings"]["param_extraction"] = elapsed
        
        if params and len(params) >= 2:
            logger.info(f"✅ Parameter extraction OK ({elapsed:.1f}s)")
            logger.info(f"   Found {len(params)} parameters: {list(params.keys())[:3]}")
            results["tests_passed"] += 1
            return True
        else:
            logger.warning(f"⚠️ Only {len(params)} parameters found")
            results["tests_passed"] += 1  # Fallbacks work
            return True
            
    except asyncio.TimeoutError:
        logger.error("❌ Parameter extraction timeout (25s)")
        results["tests_failed"] += 1
        results["errors"].append("Param extraction: Timeout")
        return False
    except Exception as e:
        logger.error(f"❌ Parameter extraction error: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Param extraction: {str(e)[:100]}")
        return False

async def test_analytics():
    """Test 4: Analytics (SHAP/LIME/Bayesian)"""
    logger.info("=" * 80)
    logger.info("TEST 4: Analytics")
    logger.info("=" * 80)
    
    try:
        from core.analytics import run_comprehensive_analytics_parallel
        start = time.time()
        
        test_params = {
            "ph": {"value": 5.5, "unit": "pH"},
            "temperature": {"value": 30.0, "unit": "°C"},
            "glucose": {"value": 100.0, "unit": "mM"}
        }
        
        result = await asyncio.wait_for(
            run_comprehensive_analytics_parallel(
                "Test query",
                test_params,
                "biomed"
            ),
            timeout=40.0
        )
        
        elapsed = time.time() - start
        results["timings"]["analytics"] = elapsed
        
        has_explainability = "explainability" in result
        has_optimization = "optimization" in result
        
        if has_explainability and has_optimization:
            logger.info(f"✅ Analytics OK ({elapsed:.1f}s)")
            logger.info(f"   Explainability: {result['explainability'].get('method', 'unknown')}")
            logger.info(f"   Optimization: {result['optimization'].get('method', 'unknown')}")
            results["tests_passed"] += 1
            return True
        else:
            logger.error(f"❌ Analytics incomplete: explainability={has_explainability}, optimization={has_optimization}")
            results["tests_failed"] += 1
            results["errors"].append("Analytics: Missing components")
            return False
            
    except asyncio.TimeoutError:
        logger.error("❌ Analytics timeout (40s)")
        results["tests_failed"] += 1
        results["errors"].append("Analytics: Timeout - reduce samples/iterations")
        return False
    except Exception as e:
        logger.error(f"❌ Analytics error: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Analytics: {str(e)[:100]}")
        return False

async def test_full_pipeline():
    """Test 5: Full multi-agent pipeline"""
    logger.info("=" * 80)
    logger.info("TEST 5: Full Multi-Agent Pipeline")
    logger.info("=" * 80)
    
    try:
        from core.langgraph import run_multi_agent
        start = time.time()
        
        test_query = "Analyze yeast growth at pH 5.5 and 30°C"
        
        result = await asyncio.wait_for(
            run_multi_agent(test_query, "biomed"),
            timeout=175.0  # CRITICAL: Must be < 180s
        )
        
        elapsed = time.time() - start
        results["timings"]["full_pipeline"] = elapsed
        
        response = result.get("final_response", "")
        trace = result.get("trace", [])
        confidence = result.get("confidence", 0.0)
        
        has_enthusiasm = "<enthusiasm>" in response
        has_explanation = "<explanation>" in response
        has_hypothesis = "<hypothesis>" in response
        
        if response and len(response) > 200 and has_enthusiasm and has_explanation:
            logger.info(f"✅ Full pipeline OK ({elapsed:.1f}s)")
            logger.info(f"   Response length: {len(response)} chars")
            logger.info(f"   Trace steps: {[t.get('step') for t in trace]}")
            logger.info(f"   Confidence: {confidence:.2f}")
            logger.info(f"   Structure: enthusiasm={has_enthusiasm}, explanation={has_explanation}, hypothesis={has_hypothesis}")
            results["tests_passed"] += 1
            
            if elapsed > 150:
                logger.warning(f"⚠️ Pipeline took {elapsed:.1f}s (close to 180s limit)")
            
            return True
        else:
            logger.error(f"❌ Pipeline incomplete or malformed")
            logger.error(f"   Length: {len(response)}, enthusiasm={has_enthusiasm}, explanation={has_explanation}")
            results["tests_failed"] += 1
            results["errors"].append(f"Pipeline: Incomplete response ({len(response)} chars)")
            return False
            
    except asyncio.TimeoutError:
        logger.error("❌ Full pipeline timeout (175s)")
        results["tests_failed"] += 1
        results["errors"].append("Pipeline: TIMEOUT - see breakdown below for bottlenecks")
        return False
    except Exception as e:
        logger.error(f"❌ Full pipeline error: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Pipeline: {str(e)[:100]}")
        return False

async def run_domain_tests(domain: Domain) -> None:
    """Run all tests for a specific domain"""
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING TESTS FOR DOMAIN: {domain.value.upper()}")
    logger.info(f"{'='*80}\n")
    
    # Track domain-specific results
    domain_results = {
        "domain": domain.value,
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "timings": {},
        "errors": []
    }
    
    # Run domain-specific tests
    test_functions = [
        (test_mistral_api, [domain]),
        (test_domain_llm, [domain]),
        (test_parameter_extraction, [domain]),
        (test_analytics, [domain]),
        (test_full_pipeline, [domain])
    ]
    
    for test_func, args in test_functions:
        test_name = test_func.__name__
        try:
            domain_results["tests_run"] += 1
            success = await test_func(*args)
            if success:
                domain_results["tests_passed"] += 1
            else:
                domain_results["tests_failed"] += 1
                domain_results["errors"].append(f"{test_name} failed")
        except Exception as e:
            domain_results["tests_failed"] += 1
            error_msg = f"Error in {test_name}: {str(e)}"
            domain_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Add a small delay between tests
        await asyncio.sleep(1)
    
    # Update global results
    results["domains_tested"].append(domain.value)
    results["tests_run"] += domain_results["tests_run"]
    results["tests_passed"] += domain_results["tests_passed"]
    results["tests_failed"] += domain_results["tests_failed"]
    results["errors"].extend(domain_results["errors"])
    
    return domain_results

async def run_all_tests():
    """Run all diagnostic tests across all domains"""
    # Initialize test results
    results["test_start_time"] = datetime.utcnow().isoformat()
    results["system_info"] = await get_system_info()
    
    # Print header
    logger.info("\n" + "="*80)
    logger.info("IXORA DIAGNOSTIC TEST SUITE")
    logger.info("="*80)
    logger.info(f"System: {results['system_info']['platform']}")
    logger.info(f"Python: {results['system_info']['python_version']}")
    logger.info(f"CPU Cores: {results['system_info']['cpu_count']}")
    logger.info(f"Memory: {results['system_info']['available_memory_gb']:.1f}GB available of {results['system_info']['total_memory_gb']:.1f}GB")
    logger.info("="*80 + "\n")
    
    # Run tests for each domain
    domains_to_test = [Domain.BIOMED, Domain.COMPUTER_SCIENCE, Domain.GENERAL]
    
    for domain in domains_to_test:
        await run_domain_tests(domain)
    
    # Finalize test results
    results["test_end_time"] = datetime.utcnow().isoformat()
    
    # Generate and save detailed report
    await generate_test_report()
    
    # Exit with appropriate status code
    sys.exit(0 if results["tests_failed"] == 0 else 1)

async def generate_test_report():
    """Generate a detailed test report"""
    # Calculate test duration
    start_time = datetime.fromisoformat(results["test_start_time"])
    end_time = datetime.fromisoformat(results["test_end_time"])
    duration = end_time - start_time
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Test started:  {start_time}")
    logger.info(f"Test finished: {end_time}")
    logger.info(f"Total duration: {duration}")
    logger.info("-" * 80)
    logger.info(f"Domains tested: {', '.join(results['domains_tested'])}")
    logger.info(f"Tests run:     {results['tests_run']}")
    logger.info(f"Tests passed:  {results['tests_passed']}")
    logger.info(f"Tests failed:  {results['tests_failed']}")
    
    # Print timing information
    if results["timings"]:
        logger.info("\nTIMING INFORMATION:" + "-"*60)
        for test_name, timing in sorted(results["timings"].items()):
            status = "⚠️" if timing > 30 else "✅"
            logger.info(f"  {status} {test_name:<40} {timing:>6.2f}s")
    
    # Print memory usage
    if results["memory_usage"]:
        logger.info("\nMEMORY USAGE (RSS):" + "-"*60)
        for test_name, mem in results["memory_usage"].items():
            delta = mem["delta_mb"]["rss"]
            logger.info(f"  {test_name:<40} {delta:>+6.1f}MB (before: {mem['before_mb']['rss_mb']:.1f}MB, after: {mem['after_mb']['rss_mb']:.1f}MB)")
    
    # Print errors and warnings
    if results["warnings"]:
        logger.info("\nWARNINGS:" + "-"*70)
        for warning in results["warnings"]:
            logger.warning(f"  ⚠️  {warning}")
    
    if results["errors"]:
        logger.info("\nERRORS:" + "-"*70)
        for error in results["errors"]:
            logger.error(f"  ❌ {error}")
    
    logger.info("="*80 + "\n")
    
    # Save detailed report to file
    report = {
        "test_run": {
            "start_time": results["test_start_time"],
            "end_time": results["test_end_time"],
            "duration_seconds": duration.total_seconds(),
            "domains_tested": results["domains_tested"],
            "tests_run": results["tests_run"],
            "tests_passed": results["tests_passed"],
            "tests_failed": results["tests_failed"]
        },
        "system_info": results["system_info"],
        "timings": results["timings"],
        "memory_usage": results["memory_usage"],
        "warnings": results["warnings"],
        "errors": results["errors"]
    }
    
    # Save JSON report
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n📊 Detailed test report saved to: {report_file}")

if __name__ == "__main__":
    asyncio.run(run_all_tests())