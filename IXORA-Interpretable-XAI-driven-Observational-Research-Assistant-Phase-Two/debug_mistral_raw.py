# debug_mistral_raw.py
import asyncio
import sys
import os
import time
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mistral import generate_with_mistral, enforce_xml_structure

# Domain-specific test prompts
TEST_PROMPTS = {
    "biomed": {
        "prompt": """Analyze the growth of yeast biomass with the following parameters:
- pH range: 3-8
- Temperature: 20-37°C
- Growth medium: YPD
- Incubation time: 24-48 hours

Provide a detailed analysis of the expected growth patterns and potential influencing factors.""",
        "expected_tags": ["enthusiasm", "explanation", "hypothesis", "followup"]
    },
    "computerscience": {
        "prompt": """Analyze the time complexity of the following algorithm and suggest optimizations:

def find_duplicates(arr):
    seen = set()
    duplicates = []
    for item in arr:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
    return duplicates""",
        "expected_tags": ["technical", "algorithms", "implementation", "challenges", "followup"]
    },
    "general": {
        "prompt": "Explain the key factors affecting plant growth indoors",
        "expected_tags": ["summary", "key_points", "followup"]
    }
}

class MistralTester:
    def __init__(self, domain: str = "biomed"):
        self.domain = domain
        self.test_prompt = TEST_PROMPTS[domain]["prompt"]
        self.expected_tags = TEST_PROMPTS[domain]["expected_tags"]
        self.results = {
            "domain": domain,
            "timestamp": datetime.utcnow().isoformat(),
            "test_prompt": self.test_prompt,
            "results": {}
        }

    async def run_test(self, max_tokens: int = 1000):
        """Run the Mistral generation test with the configured domain and prompt"""
        print(f"\n{'='*80}")
        print(f"🔬 Testing {self.domain.upper()} domain")
        print(f"📝 Prompt length: {len(self.test_prompt)}")
        print(f"⏳ Starting generation...")
        
        start_time = time.time()
        
        try:
            # Run the generation
            content, cot = await generate_with_mistral(
                self.test_prompt, 
                max_tokens=max_tokens,
                domain=self.domain
            )
            
            generation_time = time.time() - start_time
            
            # Store basic results
            self.results["generation_time"] = f"{generation_time:.2f}s"
            self.results["content_length"] = len(content) if content else 0
            self.results["cot_steps"] = len(cot)
            
            # Analyze the response
            await self._analyze_response(content, cot)
            
            # Validate XML structure
            self._validate_xml_structure(content)
            
            # Save results to file
            self._save_results()
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Error during generation: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.results["error"] = error_msg
            self._save_results()
            return False
    
    async def _analyze_response(self, content: str, cot: List[str]):
        """Analyze the generated response"""
        print("\n✅ Response Analysis:")
        print(f"- Content length: {len(content)} chars")
        print(f"- Chain-of-Thought steps: {len(cot)}")
        
        # Check for expected XML tags
        print("\n🔍 Checking for expected XML tags:")
        tag_analysis = {}
        for tag in self.expected_tags:
            has_tag = f"<{tag}>" in content and f"</{tag}>" in content
            tag_analysis[tag] = has_tag
            print(f"- <{tag}>: {'✅ Found' if has_tag else '❌ Missing'}")
        
        self.results["tag_analysis"] = tag_analysis
        
        # Show a preview of the content
        print("\n📄 Content Preview:")
        preview = content[:500] + ("..." if len(content) > 500 else "")
        print("="*60)
        print(preview)
        print("="*60)
        
        # Show CoT steps
        if cot:
            print(f"\n📋 First 3 CoT steps:")
            for i, step in enumerate(cot[:3]):
                print(f"  {i+1}. {step[:150]}{'...' if len(step) > 150 else ''}")
    
    def _validate_xml_structure(self, content: str):
        """Validate the XML structure of the response"""
        print("\n🔍 Validating XML structure...")
        try:
            enforced = enforce_xml_structure(content, self.test_prompt)
            is_valid = "<error>" not in enforced.lower()
            
            if is_valid:
                print("✅ XML structure is valid")
                self.results["xml_validation"] = "valid"
            else:
                print("❌ Invalid XML structure detected")
                self.results["xml_validation"] = "invalid"
                
            # Show first 200 chars of enforced content
            print(f"\n📊 Enforced XML (first 200 chars):")
            print("-"*60)
            print(enforced[:200] + ("..." if len(enforced) > 200 else ""))
            print("-"*60)
            
            self.results["enforced_xml_preview"] = enforced[:500] + ("..." if len(enforced) > 500 else "")
            
        except Exception as e:
            error_msg = f"Error during XML validation: {str(e)}"
            print(f"❌ {error_msg}")
            self.results["xml_validation"] = f"error: {error_msg}"
    
    def _save_results(self):
        """Save test results to a JSON file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{self.domain}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {filename}")


async def main():
    # Test all domains or specify one
    domains = ["biomed", "computerscience", "general"]
    # domains = ["biomed"]  # Uncomment to test specific domain
    
    for domain in domains:
        tester = MistralTester(domain=domain)
        await tester.run_test(max_tokens=1000)

if __name__ == "__main__":
    asyncio.run(main())