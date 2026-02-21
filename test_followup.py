"""
Test follow-up conversation like user example
"""
import requests
import sys
import uuid

# Fix encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

base_url = "http://localhost:8000/api"
session_id = str(uuid.uuid4())

print("=" * 70)
print("تست مکالمه Follow-up")
print("=" * 70)

try:
    # Initialize
    print("\n[راه‌اندازی سیستم...]")
    init_resp = requests.post(f"{base_url}/initialize", json={}, timeout=120)
    if init_resp.status_code != 200:
        print(f"خطا در راه‌اندازی: {init_resp.status_code}")
        sys.exit(1)
    print("✓ سیستم آماده\n")
    
    # First question: ca135
    print("=" * 70)
    print("👤 کاربر: ca135")
    print("=" * 70)
    
    resp1 = requests.post(
        f"{base_url}/ask",
        json={
            "question": "ca135",
            "language": "persian",
            "session_id": session_id,
            "top_k": 20,
            "use_reranking": True
        },
        timeout=90
    )
    
    if resp1.status_code == 200:
        result1 = resp1.json()
        answer1 = result1.get("answer", "")
        print("\n🤖 چت‌بات:")
        print(answer1)
        print("\n")
    else:
        print(f"خطا: {resp1.status_code}")
        sys.exit(1)
    
    # Follow-up question
    print("=" * 70)
    print("👤 کاربر: میشه بیشتر راجع بهش بگی؟ کامل و دقیق‌تر")
    print("=" * 70)
    
    resp2 = requests.post(
        f"{base_url}/ask",
        json={
            "question": "میشه بیشتر راجع بهش بگی؟ کامل و دقیق‌تر",
            "language": "persian",
            "session_id": session_id,
            "top_k": 20,
            "use_reranking": True
        },
        timeout=90
    )
    
    if resp2.status_code == 200:
        result2 = resp2.json()
        answer2 = result2.get("answer", "")
        print("\n🤖 چت‌بات:")
        print(answer2)
        
        # Check if response is more detailed
        print("\n" + "=" * 70)
        print("بررسی کامل بودن پاسخ:")
        print("=" * 70)
        
        checks = {
            "تعداد کلمات پاسخ اول": len(answer1.split()),
            "تعداد کلمات پاسخ دوم": len(answer2.split()),
            "آیا پاسخ دوم بلندتر است؟": len(answer2) > len(answer1),
            "آیا همه 5 علت آمده؟": answer2.count("**") >= 10,  # حداقل 5 علت با **
        }
        
        for key, value in checks.items():
            print(f"- {key}: {value}")
            
    else:
        print(f"خطا: {resp2.status_code}")
        print(resp2.text)
        
except Exception as e:
    print(f"خطا: {e}")
    import traceback
    traceback.print_exc()
