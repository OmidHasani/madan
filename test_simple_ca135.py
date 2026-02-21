"""
Simple test exactly like user's example
"""
import requests
import sys

# Fix encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    # Initialize
    init_resp = requests.post("http://localhost:8000/api/initialize", json={}, timeout=120)
    
    if init_resp.status_code == 200:
        print("✓ Initialized\n")
    
    # Ask simple "ca135" like the user
    resp = requests.post(
        "http://localhost:8000/api/ask",
        json={
            "question": "ca135",
            "language": "persian",
            "top_k": 20,
            "use_reranking": True
        },
        timeout=90
    )
    
    if resp.status_code == 200:
        result = resp.json()
        print("کاربر: ca135")
        print("\nچت‌بات:")
        print(result.get("answer", ""))
    else:
        print(f"Error: {resp.status_code}")
        print(resp.text)
        
except Exception as e:
    print(f"خطا: {e}")
