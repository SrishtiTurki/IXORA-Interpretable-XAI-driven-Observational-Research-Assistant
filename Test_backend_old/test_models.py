import asyncio
import sys
sys.path.append('.')
from core.model_loader import model_loader

async def test():
    print("🧪 Testing new setup...")
    
    # Test Qwen (from_pretrained)
    print("\n1. Testing Qwen...")
    try:
        qwen = await model_loader.load_qwen()
        if qwen and qwen != "api_fallback":
            print("   ✅ Qwen loaded")
            result = await model_loader.generate_with_qwen("Test", max_tokens=5)
            print(f"   Output: {result}")
        else:
            print("   ⚠️ Qwen using fallback")
    except Exception as e:
        print(f"   ❌ Qwen error: {e}")
    
    # Test BioGPT (transformers)
    print("\n2. Testing BioGPT...")
    try:
        biomistral = await model_loader.load_biomistral()
        if biomistral and biomistral != "api_fallback":
            print("   ✅ BioGPT loaded")
        else:
            print("   ⚠️ BioGPT using fallback")
    except Exception as e:
        print(f"   ❌ BioGPT error: {e}")
    
    # Test CS Model
    print("\n3. Testing CS Model...")
    try:
        cs_model = await model_loader.load_cs_model()
        if cs_model:
            print("   ✅ CS Model loaded")
        else:
            print("   ⚠️ CS Model not loaded (optional)")
    except Exception as e:
        print(f"   ❌ CS Model error: {e}")

if __name__ == "__main__":
    asyncio.run(test())