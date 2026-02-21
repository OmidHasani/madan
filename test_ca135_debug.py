"""
Debug test to see retrieved documents for CA135
"""
import requests
import json
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

base_url = "http://localhost:8000"

print("=" * 60)
print("CA135 Debug Test - Checking Retrieved Documents")
print("=" * 60)

try:
    # Initialize the system
    print("\n1. Initializing system...")
    init_response = requests.post(f"{base_url}/api/initialize", json={}, timeout=120)
    
    if init_response.status_code != 200:
        print(f"✗ Initialization failed: {init_response.status_code}")
        sys.exit(1)
    
    print("✓ System initialized\n")
    
    # Search for CA135
    print("2. Searching for CA135 documents...")
    data = {
        "question": "CA135 Eng Oil Press Sensor High Error",
        "language": "persian",
        "top_k": 10,
        "use_reranking": True
    }
    
    response = requests.post(f"{base_url}/api/ask", json=data, timeout=90)
    
    if response.status_code == 200:
        result = response.json()
        sources = result.get("sources", [])
        
        print(f"\n✓ Found {len(sources)} source documents\n")
        print("=" * 60)
        print("RETRIEVED DOCUMENTS:")
        print("=" * 60)
        
        for i, source in enumerate(sources[:5], 1):  # Show first 5
            print(f"\n--- Document {i} (Page {source.get('page', 'N/A')}) ---")
            print(f"Relevance: {source.get('relevance_score', 0):.4f}")
            print(f"\nContent Preview:")
            print(source.get('preview', 'N/A'))
            print("-" * 60)
            
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"✗ Request failed: {e}")
    import traceback
    traceback.print_exc()
