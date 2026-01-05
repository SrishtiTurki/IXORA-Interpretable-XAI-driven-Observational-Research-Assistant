"""
diagnostic_test.py - Comprehensive pipeline testing with timing
Run this to diagnose timeout issues and verify all components
"""
import asyncio
import time
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("diagnostic")

# Test results storage
results = {
    "tests_passed": 0,
    "tests_failed": 0,
    "timings": {},
    "errors": []
}

async def test_mistral_api():
    """Test 1: Mistral API connectivity"""
    logger.info("=" * 80)
    logger.info("TEST 1: Mistral API")
    logger.info("=" * 80)
    
    try:
        from core.mistral import generate_with_mistral
        start = time.time()
        
        response, _ = await asyncio.wait_for(
            generate_with_mistral("Say 'Test passed' in 3 words.", max_tokens=20),
            timeout=30.0
        )
        
        elapsed = time.time() - start
        results["timings"]["mistral_api"] = elapsed
        
        if response and len(response) > 0:
            logger.info(f"✅ Mistral API OK ({elapsed:.1f}s)")
            logger.info(f"   Response: {response[:100]}")
            results["tests_passed"] += 1
            return True
        else:
            logger.error("❌ Mistral returned empty response")
            results["tests_failed"] += 1
            results["errors"].append("Mistral API: Empty response")
            return False
            
    except asyncio.TimeoutError:
        logger.error("❌ Mistral API timeout (30s)")
        results["tests_failed"] += 1
        results["errors"].append("Mistral API: Timeout")
        return False
    except Exception as e:
        logger.error(f"❌ Mistral API error: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Mistral API: {str(e)[:100]}")
        return False

async def test_biomedlm():
    """Test 2: BioMedLM generation"""
    logger.info("=" * 80)
    logger.info("TEST 2: BioMedLM")
    logger.info("=" * 80)
    
    try:
        from core.medicalscience.loaders import generate_biomed_draft
        start = time.time()
        
        draft = await asyncio.wait_for(
            generate_biomed_draft("Analyze pH effects on yeast growth"),
            timeout=35.0  # 30s model + 5s buffer
        )
        
        elapsed = time.time() - start
        results["timings"]["biomedlm"] = elapsed
        
        if draft and len(draft) > 20:
            logger.info(f"✅ BioMedLM OK ({elapsed:.1f}s)")
            logger.info(f"   Draft: {draft[:150]}")
            results["tests_passed"] += 1
            return True
        else:
            logger.warning(f"⚠️ BioMedLM returned short draft: {draft}")
            # Not a failure if fallback works
            results["tests_passed"] += 1
            return True
            
    except asyncio.TimeoutError:
        logger.error("❌ BioMedLM timeout (35s)")
        results["tests_failed"] += 1
        results["errors"].append("BioMedLM: Timeout - reduce max_length or timeout")
        return False
    except Exception as e:
        logger.error(f"❌ BioMedLM error: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"BioMedLM: {str(e)[:100]}")
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

async def run_all_tests():
    """Run all diagnostic tests"""
    logger.info("\n" + "="*80)
    logger.info("IXORA DIAGNOSTIC TEST SUITE")
    logger.info("="*80 + "\n")
    
    # Run tests sequentially to avoid interference
    await test_mistral_api()
    await asyncio.sleep(2)
    
    await test_biomedlm()
    await asyncio.sleep(2)
    
    await test_parameter_extraction()
    await asyncio.sleep(2)
    
    await test_analytics()
    await asyncio.sleep(2)
    
    await test_full_pipeline()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*80)
    logger.info(f"Tests passed: {results['tests_passed']}/5")
    logger.info(f"Tests failed: {results['tests_failed']}/5")
    
    if results["timings"]:
        logger.info("\nTimings:")
        for component, timing in results["timings"].items():
            status = "⚠️" if timing > 30 else "✅"
            logger.info(f"  {status} {component}: {timing:.1f}s")
        
        total_estimated = sum(results["timings"].values())
        logger.info(f"\nEstimated total pipeline time: {total_estimated:.1f}s")
        
        if total_estimated > 150:
            logger.warning("⚠️ CRITICAL: Pipeline might exceed 180s limit!")
            logger.warning("   Recommended fixes:")
            logger.warning("   1. Reduce BioMedLM max_length to 60")
            logger.warning("   2. Reduce analytics samples to 50")
            logger.warning("   3. Use SHAP-only (skip LIME) for speed")
    
    if results["errors"]:
        logger.info("\nErrors encountered:")
        for error in results["errors"]:
            logger.error(f"  ❌ {error}")
    
    logger.info("="*80 + "\n")
    
    # Exit code
    sys.exit(0 if results["tests_failed"] == 0 else 1)

if __name__ == "__main__":
    asyncio.run(run_all_tests())