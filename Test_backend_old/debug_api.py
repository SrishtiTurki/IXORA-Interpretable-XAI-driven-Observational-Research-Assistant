# debug_api.py
import requests
import json

def test_api():
    url = "http://localhost:8000/chat"
    payload = {
        "message": "test fungi growth at pH 5.5",
        "session_id": "debug123"
    }
    
    print("Testing API endpoint...")
    print(f"URL: {url}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n=== KEYS ===")
            print(f"Keys: {list(data.keys())}")
            
            print("\n=== CONTENT ===")
            if "content" in data:
                content = data["content"]
                print(f"Content length: {len(content)}")
                print(f"First 500 chars:\n{content[:500]}")
            else:
                print("No 'content' key found!")
                
            print("\n=== FINAL_RESPONSE ===")
            if "final_response" in data:
                final = data["final_response"]
                print(f"Final response length: {len(final)}")
                print(f"First 500 chars:\n{final[:500]}")
            
            print("\n=== TRACE ===")
            if "trace" in data:
                for step in data["trace"]:
                    print(f"{step.get('step')}: {step.get('reasoning', '')[:100]}")
                    
            # Save to file for inspection
            with open("api_response.json", "w") as f:
                json.dump(data, f, indent=2)
            print("\n✅ Saved full response to api_response.json")
            
        else:
            print(f"\n❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api()