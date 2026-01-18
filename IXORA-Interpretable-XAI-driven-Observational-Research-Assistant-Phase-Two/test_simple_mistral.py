#!/usr/bin/env python3
"""
Simple test script for Mistral model integration with domain awareness.
"""
import asyncio
import os
import sys
import time
import psutil
import logging
from enum import Enum
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mistral_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class Domain(Enum):
    """Supported domains for testing."""
    BIOMED = "biomed"
    COMPUTER_SCIENCE = "computerscience"
    GENERAL = "general"

# Domain-specific test configurations
DOMAIN_CONFIGS = {
    Domain.BIOMED: {
        "test_query": "Analyze yeast biomass with pH 3-8 and temperature 20-37°C",
        "xml_tags": ["<enthusiasm>", "<explanation>", "<hypothesis>", "<followup>"]
    },
    Domain.COMPUTER_SCIENCE: {
        "test_query": "Analyze the performance impact of different sorting algorithms on large datasets",
        "xml_tags": ["<overview>", "<analysis>", "<conclusion>", "<followup>"]
    },
    Domain.GENERAL: {
        "test_query": "Provide an overview of climate change impacts",
        "xml_tags": ["<introduction>", "<key_points>", "<implications>", "<followup>"]
    }
}

def get_memory_usage() -> Dict[str, float]:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / (1024 * 1024),
        'vms_mb': mem_info.vms / (1024 * 1024)
    }

async def test_mistral(domain: Domain = Domain.BIOMED, max_tokens: int = 300) -> Dict[str, Any]:
    """Test Mistral model with the given domain and parameters.
    
    Args:
        domain: The domain to test (biomed, computerscience, or general)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary containing test results and metrics
    """
    start_time = time.time()
    start_mem = get_memory_usage()
    
    domain_config = DOMAIN_CONFIGS[domain]
    test_prompt = domain_config["test_query"]
    
    result = {
        "domain": domain.value,
        "prompt": test_prompt,
        "status": "failed",
        "metrics": {
            "execution_time_sec": 0,
            "memory_used_mb": 0,
            "peak_memory_mb": 0,
            "response_length": 0,
            "missing_xml_tags": []
        },
        "response_preview": ""
    }
    
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING: {domain.value.upper()} DOMAIN")
        logger.info(f"Prompt: {test_prompt}")
        
        # Import here to catch import errors
        try:
            from core.mistral import generate_with_mistral
        except ImportError as e:
            logger.error(f"Failed to import generate_with_mistral: {e}")
            result["error"] = f"Import error: {str(e)}"
            return result
        
        # Generate response
        content, cot = await generate_with_mistral(test_prompt, max_tokens=max_tokens)
        
        # Measure memory after generation
        end_mem = get_memory_usage()
        end_time = time.time()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_used = end_mem["rss_mb"] - start_mem["rss_mb"]
        
        # Check for required XML tags
        missing_tags = [
            tag for tag in domain_config["xml_tags"]
            if tag not in content
        ]
        
        # Update result
        result.update({
            "status": "passed" if not missing_tags else "partial",
            "metrics": {
                "execution_time_sec": round(execution_time, 2),
                "memory_used_mb": round(memory_used, 1),
                "peak_memory_mb": round(end_mem["rss_mb"], 1),
                "response_length": len(content),
                "missing_xml_tags": missing_tags
            },
            "response_preview": content[:500] + ("..." if len(content) > 500 else ""),
            "cot_steps": len(cot) if cot else 0
        })
        
        # Log results
        logger.info(f"✅ Test completed in {execution_time:.2f}s")
        logger.info(f"   Memory used: {memory_used:.1f}MB")
        logger.info(f"   Response length: {len(content)} chars")
        logger.info(f"   CoT steps: {len(cot) if cot else 0}")
        
        if missing_tags:
            logger.warning(f"   Missing XML tags: {', '.join(missing_tags)}")
        
        logger.info("\nResponse preview:")
        logger.info("-" * 80)
        logger.info(content[:300] + ("..." if len(content) > 300 else ""))
        logger.info("-" * 80)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        result.update({
            "status": "failed",
            "error": str(e),
            "traceback": str(traceback.format_exc())
        })
    
    return result

async def main():
    """Run tests for all domains and print summary."""
    # Determine which domains to test
    domains_to_test = [Domain.BIOMED]  # Default to biomed only
    
    if len(sys.argv) > 1:
        domain_arg = sys.argv[1].lower()
        domains_to_test = [d for d in Domain if d.value == domain_arg]
        if not domains_to_test:
            logger.error(f"Invalid domain: {domain_arg}. Available domains: {[d.value for d in Domain]}")
            return
    
    # Run tests
    results = []
    for domain in domains_to_test:
        result = await test_mistral(domain)
        results.append(result)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for result in results:
        status = "✅ PASSED" if result["status"] == "passed" else "⚠️  PARTIAL" if result["status"] == "partial" else "❌ FAILED"
        logger.info(f"{status} - {result['domain'].upper()}:")
        logger.info(f"  Time: {result['metrics']['execution_time_sec']:.2f}s")
        logger.info(f"  Memory: {result['metrics']['memory_used_mb']:.1f}MB")
        logger.info(f"  Response: {result['metrics']['response_length']} chars")
        
        if result["status"] != "passed":
            if "error" in result:
                logger.error(f"  Error: {result['error']}")
            if result["metrics"]["missing_xml_tags"]:
                logger.warning(f"  Missing tags: {', '.join(result['metrics']['missing_xml_tags'])}")
    
    logger.info("\nTest complete. Check mistral_test.log for detailed logs.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)