#!/usr/bin/env python3
"""
Test script for BioMistral model integration with domain-specific testing.
"""
import asyncio
import os
import sys
import time
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('biomistral_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Domain-specific test configurations
BIOMED_TESTS = [
    {
        "name": "Basic Biology Concept",
        "prompt": "Explain the process of photosynthesis in plants.",
        "expected_terms": ["chlorophyll", "light", "carbon dioxide", "oxygen"],
        "max_tokens": 200
    },
    {
        "name": "Medical Terminology",
        "prompt": "What are the main symptoms of diabetes?",
        "expected_terms": ["glucose", "blood sugar", "thirst", "urination"],
        "max_tokens": 150
    },
    {
        "name": "Research Analysis",
        "prompt": "Summarize the key findings about CRISPR gene editing in recent research.",
        "expected_terms": ["CRISPR", "DNA", "editing", "Cas9"],
        "max_tokens": 250
    }
]

async def test_biomistral() -> Dict[str, Any]:
    """Test BioMistral model with domain-specific prompts.
    
    Returns:
        Dictionary containing test results and metrics
    """
    results = {
        "domain": "biomedical",
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "test_cases": []
    }
    
    try:
        # Import model loader
        try:
            from core.model_loader import get_biomistral
            logger.info("✅ Model loader import successful")
        except ImportError as e:
            logger.error(f"❌ Failed to import model loader: {e}")
            results["error"] = f"Import error: {str(e)}"
            return results
        
        # Load model
        start_time = time.time()
        try:
            model = await get_biomistral()
            if not model:
                logger.error("❌ Failed to load BioMistral model")
                results["error"] = "Model loading failed"
                return results
            logger.info(f"✅ BioMistral model loaded in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            results["error"] = f"Model loading error: {str(e)}"
            return results
        
        # Run test cases
        for test_case in BIOMED_TESTS:
            test_start = time.time()
            test_result = {
                "name": test_case["name"],
                "prompt": test_case["prompt"],
                "status": "failed",
                "duration": 0,
                "response": "",
                "missing_terms": [],
                "error": None
            }
            
            results["tests_run"] += 1
            logger.info(f"\n🧪 Running test: {test_case['name']}")
            logger.info(f"   Prompt: {test_case['prompt']}")
            
            try:
                # Generate response
                def generate():
                    return model(
                        test_case["prompt"],
                        max_new_tokens=test_case["max_tokens"],
                        temperature=0.7,
                        do_sample=True
                    )
                
                response = await asyncio.wait_for(
                    asyncio.to_thread(generate),
                    timeout=30.0
                )
                
                # Check for expected terms
                response_text = str(response).lower()
                missing_terms = [
                    term for term in test_case["expected_terms"]
                    if term.lower() not in response_text
                ]
                
                # Update test result
                test_result.update({
                    "status": "passed" if not missing_terms else "partial",
                    "duration": time.time() - test_start,
                    "response": response_text[:500] + ("..." if len(response_text) > 500 else ""),
                    "missing_terms": missing_terms
                })
                
                if not missing_terms:
                    logger.info(f"✅ Test passed in {test_result['duration']:.2f}s")
                    results["tests_passed"] += 1
                else:
                    logger.warning(f"⚠️  Test partial - Missing terms: {', '.join(missing_terms)}")
                    results["tests_failed"] += 1
                
            except asyncio.TimeoutError:
                error_msg = "Test timed out after 30 seconds"
                logger.error(f"❌ {error_msg}")
                test_result.update({
                    "status": "failed",
                    "duration": time.time() - test_start,
                    "error": error_msg
                })
                results["tests_failed"] += 1
                
            except Exception as e:
                error_msg = f"Test failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                test_result.update({
                    "status": "failed",
                    "duration": time.time() - test_start,
                    "error": str(e),
                    "traceback": str(traceback.format_exc())
                })
                results["tests_failed"] += 1
            
            results["test_cases"].append(test_result)
    
    except Exception as e:
        logger.error(f"❌ Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results["error"] = f"Unhandled exception: {str(e)}"
    
    return results

async def main():
    """Run tests and display results."""
    logger.info("\n" + "="*80)
    logger.info("🧬 BioMistral Domain Testing")
    logger.info("="*80)
    
    results = await test_biomistral()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    if "error" in results:
        logger.error(f"❌ Error during testing: {results['error']}")
    
    logger.info(f"Tests Run: {results['tests_run']}")
    logger.info(f"✅ Passed: {results['tests_passed']}")
    logger.info(f"⚠️  Partial: {results.get('tests_partial', 0)}")
    logger.info(f"❌ Failed: {results['tests_failed']}")
    
    success_rate = (results['tests_passed'] / results['tests_run']) * 100 if results['tests_run'] > 0 else 0
    logger.info(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    # Detailed test cases
    if results.get('test_cases'):
        logger.info("\nTest Cases:")
        for i, case in enumerate(results['test_cases'], 1):
            status = "✅" if case['status'] == 'passed' else "⚠️ " if case['status'] == 'partial' else "❌"
            logger.info(f"\n{status} Test {i}: {case['name']} ({case['duration']:.2f}s)")
            if case['status'] != 'passed':
                if case['missing_terms']:
                    logger.warning(f"   Missing terms: {', '.join(case['missing_terms'])}")
                if case.get('error'):
                    logger.error(f"   Error: {case['error']}")
    
    logger.info("\nTest complete. Check biomistral_test.log for detailed logs.")
    return results['tests_failed'] == 0

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)