# test_biomistral.py
import asyncio
import sys
import os
sys.path.append('.')

async def test_biomistral():
    print("🔬 Testing BioMistral...")
    
    try:
        # Test 1: Check if model_loader can get BioMistral
        from core.model_loader import get_biomistral
        print("✅ Model loader import successful")
        
        biomistral = await get_biomistral()
        if biomistral:
            print("✅ BioMistral instance retrieved")
            
            # Test generation
            test_prompt = "What is pH?"
            print(f"📝 Testing generation with: '{test_prompt}'")
            
            try:
                def generate():
                    return biomistral(test_prompt, max_new_tokens=10)
                
                result = await asyncio.wait_for(
                    asyncio.to_thread(generate),
                    timeout=10.0
                )
                print(f"✅ Generation successful: {result[:50]}...")
                return True
            except asyncio.TimeoutError:
                print("❌ Generation timeout")
            except Exception as e:
                print(f"❌ Generation error: {e}")
        else:
            print("❌ BioMistral instance is None")
            
    except Exception as e:
        print(f"❌ Import/loading error: {e}")
        import traceback
        traceback.print_exc()
    
    return False

if __name__ == "__main__":
    result = asyncio.run(test_biomistral())
    print(f"\n🎯 Test result: {'✅ PASS' if result else '❌ FAIL'}")