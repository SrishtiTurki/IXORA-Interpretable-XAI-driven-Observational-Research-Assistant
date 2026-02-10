# test_simple_mistral.py
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the enforce_xml_structure to see what's happening
def mock_enforce_xml_structure(content, user_query):
    print(f"\n🔧 Mock enforce_xml_structure:")
    print(f"Input length: {len(content)}")
    print(f"Input preview: {content[:200]}...")
    
    # Simple enforcement
    if '<enthusiasm>' not in content:
        content = f"<enthusiasm>That's a great question!</enthusiasm>\n\n{content}"
    if '<followup>' not in content:
        content = f"{content}\n\n<followup>What would you like to explore further?</followup>"
    
    print(f"Output length: {len(content)}")
    return content

# Temporarily replace the function
import core.mistral
core.mistral.enforce_xml_structure = mock_enforce_xml_structure

from core.mistral import generate_with_mistral

async def test():
    print("Testing simplified Mistral...")
    
    test_prompt = "Analyze yeast biomass with pH 3-8 and temperature 20-37°C"
    
    try:
        content, cot = await generate_with_mistral(test_prompt, max_tokens=300)
        
        print(f"\n✅ Final Result:")
        print(f"Content length: {len(content)}")
        print(f"Content preview:\n{'-'*50}")
        print(content[:300] if content else "EMPTY!")
        print(f"{'-'*50}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())