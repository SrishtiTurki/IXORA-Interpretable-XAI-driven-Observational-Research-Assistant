'''# test_medgemma_from_pretrained.py
import asyncio
import sys
sys.path.append(".")

async def test_from_pretrained():
    print("🧪 Testing MedGemma with from_pretrained...")
    
    try:
        # First check if llama-cpp-python is installed
        import llama_cpp
        print(f"✅ llama-cpp-python version: {llama_cpp.__version__}")
        
        # Test direct from_pretrained
        from llama_cpp import Llama
        
        print("Downloading MedGemma (this may take a while, ~4.1GB)...")
        
        # Use smaller model for faster test if needed
        model = await asyncio.to_thread(
            Llama.from_pretrained,
            repo_id="lmstudio-community/medgemma-4b-it-GGUF",
            filename="medgemma-4b-it-Q3_K_L.gguf",  # Smaller: 3.1GB vs 4.1GB
            n_ctx=2048,
            n_threads=4,
            verbose=True  # Show download progress
        )
        
        print("✅ Model downloaded and loaded!")
        
        # Test generation
        print("\nTesting generation...")
        
        # MedGemma uses Gemma chat format
        prompt = """<start_of_turn>user
What is the normal range for human body temperature?
<end_of_turn>
<start_of_turn>model
"""
        
        output = model(
            prompt,
            max_tokens=50,
            temperature=0.1,
            stop=["</s>", "<end_of_turn>"]
        )
        
        response = output['choices'][0]['text'].strip()
        print(f"Response: {response}")
        
        return True
        
    except ImportError:
        print("❌ Install llama-cpp-python first:")
        print("   pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    success = asyncio.run(test_from_pretrained())
    print("="*60)
    print(f"\n{'✅ SUCCESS!' if success else '❌ FAILED'}")'''


import requests
import os

url = "https://huggingface.co/google/medgemma-4b-it-gguf/resolve/main/medgemma-4b-it-Q3_K_L.gguf"
output = "models/medgemma-4b-it-Q3_K_L.gguf"

os.makedirs("models", exist_ok=True)

print(f"Downloading MedGemma 4B Q3_K_L...")
print(f"URL: {url}")
print(f"Saving to: {output}")

# Download with streaming
response = requests.get(url, stream=True)
response.raise_for_status()

total_size = int(response.headers.get('content-length', 0))
block_size = 8192
downloaded = 0

with open(output, 'wb') as f:
    for chunk in response.iter_content(chunk_size=block_size):
        f.write(chunk)
        downloaded += len(chunk)
        
        # Show progress
        if total_size > 0:
            percent = (downloaded / total_size) * 100
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

print(f"\n✅ Download complete! File size: {total_size/(1024*1024*1024):.2f} GB")