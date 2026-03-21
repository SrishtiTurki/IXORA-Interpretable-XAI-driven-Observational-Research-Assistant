# debug_mistral_raw.py
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mistral import generate_with_mistral

async def test():
    print("Testing Mistral with your exact query...")
    
    test_prompt = """I want to work fungi I want to work with yeast biomass Statistically analyze the impact of pH and temperature? I want to do this, and I also want to know the other factors involving in this. so here is what i am using: Range of pH & Temp: pH 3–8 and 20–37 °C (covering acidic to near-neutral and typical mesophilic range for yeast)."""
    
    print(f"\n📝 Prompt length: {len(test_prompt)}")
    
    try:
        content, cot = await generate_with_mistral(test_prompt, max_tokens=500)
        
        print(f"\n✅ Raw Response Analysis:")
        print(f"Content length: {len(content)}")
        print(f"Content is None? {content is None}")
        print(f"Content is empty string? {content == ''}")
        print(f"Content stripped length: {len(content.strip())}")
        
        if content:
            print(f"\n📄 First 500 chars of RAW content:")
            print("="*60)
            print(content[:500])
            print("="*60)
            
            # Check for XML tags
            print(f"\n🔍 Contains <enthusiasm> tag? {'<enthusiasm>' in content}")
            print(f"Contains <explanation> tag? {'<explanation>' in content}")
            print(f"Contains <hypothesis> tag? {'<hypothesis>' in content}")
            print(f"Contains <followup> tag? {'<followup>' in content}")
            
            # Check what enforce_xml_structure does
            from core.mistral import enforce_xml_structure
            enforced = enforce_xml_structure(content, test_prompt)
            print(f"\n📊 After enforce_xml_structure length: {len(enforced)}")
            print(f"Enforced first 200 chars:\n{enforced[:200]}")
        else:
            print("\n❌ Content is empty!")
            
        print(f"\n📋 CoT steps: {len(cot)}")
        for i, step in enumerate(cot[:3]):  # Show first 3
            print(f"  {i+1}. {step[:100]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())