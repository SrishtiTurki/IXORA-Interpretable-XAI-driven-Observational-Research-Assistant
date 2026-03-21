'''#!/usr/bin/env python3
"""
Diagnostic script to test backend components individually
Run: python test_backend.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_mistral_api():
    """Test 1: Can we call Mistral API directly?"""
    print("\n" + "="*80)
    print("TEST 1: Mistral API Direct Call")
    print("="*80)
    
    from core.mistral import call_mistral_api
    
    simple_prompt = "Say exactly: 'Mistral API is working correctly.'"
    
    try:
        response = await call_mistral_api(simple_prompt, max_tokens=50, temperature=0.3)
        
        print(f"✅ Response received: {len(response)} chars")
        print(f"Content: {response}")
        
        if len(response) > 10:
            print("✅ TEST 1 PASSED")
            return True
        else:
            print("❌ TEST 1 FAILED - Response too short")
            return False
            
    except Exception as e:
        print(f"❌ TEST 1 FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mistral_generate():
    """Test 2: Does generate_with_mistral work?"""
    print("\n" + "="*80)
    print("TEST 2: Mistral Generate Function")
    print("="*80)
    
    from core.mistral import generate_with_mistral
    
    test_prompt = """You are a biomedical research assistant. 

User question: What is pH?

Provide a brief 2-paragraph explanation."""
    
    try:
        response, cot = await generate_with_mistral(test_prompt, max_tokens=300, temperature=0.7)
        
        print(f"Response length: {len(response)}")
        print(f"First 200 chars: {response[:200]}")
        print(f"CoT steps: {len(cot)}")
        
        if len(response) > 50:
            print("✅ TEST 2 PASSED")
            return True
        else:
            print("❌ TEST 2 FAILED - Response too short")
            print(f"Full response: {response}")
            return False
            
    except Exception as e:
        print(f"❌ TEST 2 FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_parameter_extraction():
    """Test 3: Can we extract biomedical parameters?"""
    print("\n" + "="*80)
    print("TEST 3: Parameter Extraction")
    print("="*80)
    
    from core.utils import extract_parameters
    
    #test_query = '''#Statistically analyze the impact of pH and temperature? I want to do this, and I also want to know the other factors involving in this.
#so here is what i am using: Range of pH & Temp: pH 3–8 and 20–37 °C (covering acidic to near-neutral and typical mesophilic range for yeast).
'''
    
    try:
        params = await extract_parameters(test_query, "biomed")
        
        print(f"Extracted {len(params)} parameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")
        
        if len(params) > 0:
            print("✅ TEST 3 PASSED")
            return True
        else:
            print("❌ TEST 3 FAILED - No parameters extracted")
            return False
            
    except Exception as e:
        print(f"❌ TEST 3 FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cs_parameter_extraction():
    """Test 3b: Can we extract CS parameters?"""
    print("\n" + "="*80)
    print("TEST 3b: CS Parameter Extraction")
    print("="*80)
    
    from core.utils import extract_parameters
    
    test_query = (
        "Optimize training with batch size 32, learning rate 0.001, epochs 10. "
        "Dataset size is 1.2M samples; target latency < 20ms; discuss Big-O time complexity."
    )
    
    try:
        params = await extract_parameters(test_query, "cs")
        
        print(f"Extracted {len(params)} parameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")
        
        if len(params) > 0:
            print("✅ TEST 3b PASSED")
            return True
        else:
            print("❌ TEST 3b FAILED - No parameters extracted")
            return False
            
    except Exception as e:
        print(f"❌ TEST 3b FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_analytics():
    """Test 4: Do analytics run (biomed)?"""
    print("\n" + "="*80)
    print("TEST 4: Analytics Pipeline")
    print("="*80)
    
    from core.analytics import run_comprehensive_analytics_parallel
    
    test_params = {
        "ph": {"value": 5.5, "unit": "pH", "description": "Test pH"},
        "temp": {"value": 30.0, "unit": "°C", "description": "Test temperature"}
    }
    
    try:
        result = await run_comprehensive_analytics_parallel(
            user_input="Test query for pH and temperature analysis",
            parameters=test_params,
            domain="biomed"
        )
        
        print(f"Result keys: {list(result.keys())}")
        
        checks = {
            "Has explainability": "explainability" in result,
            "Has causal": "causal" in result,
            "Has optimization": "optimized" in result,
            "SHAP available": result.get("explainability", {}).get("shap_importance") is not None,
            "ATE available": result.get("causal", {}).get("ate") is not None,
            "Optimized values": result.get("optimized", {}).get("optimized_values") is not None
        }
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        
        if all(checks.values()):
            print("✅ TEST 4 PASSED")
            return True
        else:
            print("⚠️ TEST 4 PARTIAL - Some analytics missing")
            return False
            
    except Exception as e:
        print(f"❌ TEST 4 FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_analytics_cs():
    """Test 4b: Do analytics run (cs)?"""
    print("\n" + "="*80)
    print("TEST 4b: Analytics Pipeline (CS)")
    print("="*80)
    
    from core.analytics import run_comprehensive_analytics_parallel
    
    test_params = {
        "batch_size": {"value": 32, "unit": "samples", "description": "Test batch size"},
        "learning_rate": {"value": 0.001, "unit": "", "description": "Test learning rate"},
        "epochs": {"value": 10, "unit": "epochs", "description": "Test epochs"},
    }
    
    try:
        result = await run_comprehensive_analytics_parallel(
            user_input="Test query for optimizing batch size and learning rate",
            parameters=test_params,
            domain="cs"
        )
        
        print(f"Result keys: {list(result.keys())}")
        
        checks = {
            "Has explainability": "explainability" in result,
            "Has causal": "causal" in result,
            "Has optimization": "optimization" in result or "optimized" in result,
        }
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        
        if all(checks.values()):
            print("✅ TEST 4b PASSED")
            return True
        else:
            print("⚠️ TEST 4b PARTIAL - Some analytics missing")
            return False
            
    except Exception as e:
        print(f"❌ TEST 4b FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_pipeline():
    """Test 5: Full pipeline end-to-end (biomed, langgraph)"""
    print("\n" + "="*80)
    print("TEST 5: Full Pipeline (langgraph)")
    print("="*80)
    
    from core.langgraph import run_multi_agent
    
    test_query = "Analyze yeast biomass production at pH 5.5 and temperature 30°C"
    
    try:
        result = await run_multi_agent(query=test_query, domain="biomed", session_id="test_backend_biomed")
        
        print(f"Result keys: {list(result.keys())}")
        
        response = result.get("final_response", "")
        print(f"Response length: {len(response)}")
        print(f"First 300 chars: {response[:300]}")
        
        trace = result.get("trace", [])
        print(f"Trace steps: {len(trace)}")
        for step in trace:
            if isinstance(step, dict):
                print(f"  - {step.get('step')}: {str(step)[:80]}")
        
        if len(response) > 500:
            print("✅ TEST 5 PASSED")
            return True
        else:
            print(f"❌ TEST 5 FAILED - Response too short ({len(response)} chars)")
            print(f"Full response: {response}")
            return False
            
    except Exception as e:
        print(f"❌ TEST 5 FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_pipeline_cs():
    """Test 5b: Full pipeline end-to-end (cs, langgraph)"""
    print("\n" + "="*80)
    print("TEST 5b: Full Pipeline (CS, langgraph)")
    print("="*80)
    
    from core.langgraph import run_multi_agent
    
    test_query = "Analyze time complexity of Dijkstra vs A* and propose a benchmark plan with baselines."
    
    try:
        result = await run_multi_agent(query=test_query, domain="cs", session_id="test_backend_cs")
        
        print(f"Result keys: {list(result.keys())}")
        
        response = result.get("final_response", "")
        print(f"Response length: {len(response)}")
        print(f"First 300 chars: {response[:300]}")
        
        trace = result.get("trace", [])
        print(f"Trace steps: {len(trace)}")
        
        # CS structure includes <clarify>
        has_clarify = "<clarify>" in response
        print(f"Has <clarify>: {'✅' if has_clarify else '❌'}")
        
        if len(response) > 500 and has_clarify:
            print("✅ TEST 5b PASSED")
            return True
        else:
            print(f"❌ TEST 5b FAILED - Response too short or missing <clarify>")
            return False
            
    except Exception as e:
        print(f"❌ TEST 5b FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("\n" + "#"*80)
    print("# BACKEND DIAGNOSTIC TEST SUITE")
    print("#"*80)
    
    # Check environment
    print("\nEnvironment Check:")
    print(f"  MISTRAL_API_KEY: {'SET' if os.getenv('MISTRAL_API_KEY') else 'NOT SET'}")
    print(f"  MISTRAL_USE_API: {os.getenv('MISTRAL_USE_API', 'not set')}")
    
    results = {
        "Mistral API": await test_mistral_api(),
        "Mistral Generate": await test_mistral_generate(),
        "Parameter Extraction": await test_parameter_extraction(),
        "CS Parameter Extraction": await test_cs_parameter_extraction(),
        "Analytics": await test_analytics(),
        "CS Analytics": await test_analytics_cs(),
        "Full Pipeline": await test_full_pipeline(),
        "CS Full Pipeline": await test_full_pipeline_cs(),
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Backend should be working.")
    else:
        print(f"\n⚠️ {total_count - passed_count} TEST(S) FAILED")
        print("Check the logs above for details on what failed.")
    
    print("\nTo check detailed backend logs:")
    print("  tail -f backend_debug.log")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)'''


import asyncio
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

async def debug():
    from core.model_loader import model_loader
    
    print("="*60)
    print("DEBUG CS MODEL LOADING")
    print("="*60)
    
    # Check status first
    status = model_loader.get_status()
    print(f"\nInitial status - CS model loaded: {status['cs_model']['loaded']}")
    
    # Try to load CS model
    print("\nAttempting to load CS model...")
    try:
        cs_model = await model_loader.load_cs_model()
        print(f"load_cs_model() returned: {type(cs_model)}")
        print(f"Is None? {cs_model is None}")
        
        # Check status again
        status = model_loader.get_status()
        print(f"\nAfter loading - CS model loaded: {status['cs_model']['loaded']}")
        
    except Exception as e:
        print(f"❌ Exception during loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug())