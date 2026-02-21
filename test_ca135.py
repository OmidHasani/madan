"""
Quick test for CA135 error code response completeness
"""
import requests
import json
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

base_url = "http://localhost:8000"

print("=" * 60)
print("Testing CA135 Error Code Response")
print("=" * 60)

try:
    # First, initialize the system
    print("\n1. Initializing system...")
    init_response = requests.post(f"{base_url}/api/initialize", json={}, timeout=120)
    
    if init_response.status_code != 200:
        print(f"✗ Initialization failed: {init_response.status_code}")
        print(init_response.text)
        sys.exit(1)
    
    print("✓ System initialized successfully\n")
    
    # Test CA135 error code
    print("2. Testing CA135 error code...")
    data = {
        "question": "ca135",
        "language": "persian",
        "top_k": 20,
        "use_reranking": True
    }
    
    response = requests.post(f"{base_url}/api/ask", json=data, timeout=90)
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("answer", "")
        
        print("\n✓ Response received successfully\n")
        print("ANSWER:")
        print("-" * 60)
        print(answer)
        print("-" * 60)
        
        # Check for completeness indicators
        checks = {
            "پین (37)": "بین ENG (37)" in answer or "پین (37)" in answer or "پین‌های (37)" in answer,
            "پین (47)": "بین ENG (47)" in answer or "پین (47)" in answer or "پین‌های (47)" in answer,
            "پین (13)": "بین ENG (13)" in answer or "پین (13)" in answer or "پین‌های (13)" in answer,
            "POIL (1)": "POIL (1)" in answer,
            "POIL (2)": "POIL (2)" in answer,
            "POIL (3)": "POIL (3)" in answer,
            "مقاومت": "مقاومت" in answer or "اهم" in answer,
            "ولتاژ": "ولتاژ" in answer or "ولت" in answer,
        }
        
        print("\n" + "=" * 60)
        print("COMPLETENESS CHECKS:")
        print("=" * 60)
        passed = 0
        for check_name, passed_check in checks.items():
            status = "✓ PASS" if passed_check else "✗ FAIL"
            print(f"{status}: {check_name}")
            if passed_check:
                passed += 1
        
        print(f"\nScore: {passed}/{len(checks)}")
        
        if passed >= 6:
            print("\n✓ Response appears complete!")
        else:
            print("\n✗ Response may be missing details")
            
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"✗ Request failed: {e}")
