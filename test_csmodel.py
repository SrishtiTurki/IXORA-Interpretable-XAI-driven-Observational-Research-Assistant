import asyncio
import sys
import os

sys.path.append('.')


async def test_csmodel():
    print("💻 Testing CS GGUF model (CSMODEL)...")

    try:
        # Import CS draft generator
        from core.computerscience.loaders import generate_cs_draft
        print("✅ CS loaders import successful")

        test_prompt = "Compare time complexity of Dijkstra vs A* on large graphs."
        print(f"📝 Testing CS draft generation with: '{test_prompt}'")

        try:
            # Call async CS draft generator with a small max token limit
            result = await asyncio.wait_for(
                generate_cs_draft(test_prompt, max_tokens=64),
                timeout=60.0,
            )
            if result and len(result.strip()) > 20:
                print(f"✅ CS draft generation successful: {result[:100]}...")
                return True
            else:
                print("❌ CS draft generation returned empty or too short output")
        except asyncio.TimeoutError:
            print("❌ CS draft generation timeout")
        except Exception as e:
            print(f"❌ CS draft generation error: {e}")

    except Exception as e:
        print(f"❌ Import/loading error for CS model: {e}")
        import traceback

        traceback.print_exc()

    return False


if __name__ == "__main__":
    result = asyncio.run(test_csmodel())
    print(f"\n🎯 CS Model test result: {'✅ PASS' if result else '❌ FAIL'}")

