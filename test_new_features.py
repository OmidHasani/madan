"""
اسکریپت تست قابلیت‌های جدید چت‌بات
این اسکریپت به صورت خودکار قابلیت‌های تعاملی و حافظه را تست می‌کند.
"""

import requests
import time
import json
from typing import List, Dict

# تنظیمات
BASE_URL = "http://localhost:8000"
API_ASK = f"{BASE_URL}/api/ask"
API_SESSION = f"{BASE_URL}/api/session"
API_SESSIONS = f"{BASE_URL}/api/sessions"


class ChatbotTester:
    """کلاس تست قابلیت‌های چت‌بات"""
    
    def __init__(self):
        self.session_id = None
        self.conversation_history = []
    
    def print_separator(self, title: str = ""):
        """چاپ خط جدا کننده"""
        print("\n" + "=" * 80)
        if title:
            print(f"  {title}")
            print("=" * 80)
    
    def ask_question(self, question: str, show_response: bool = True) -> Dict:
        """ارسال سوال به چت‌بات"""
        print(f"\n👤 سوال: {question}")
        
        payload = {
            "question": question,
            "top_k": 10,
            "use_reranking": True,
            "language": "persian",
            "session_id": self.session_id
        }
        
        try:
            response = requests.post(API_ASK, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # ذخیره session_id
            if data.get("session_id") and not self.session_id:
                self.session_id = data["session_id"]
                print(f"✅ Session ایجاد شد: {self.session_id[:8]}...")
            
            # ذخیره در تاریخچه
            self.conversation_history.append({
                "question": question,
                "answer": data.get("answer", ""),
                "intent": data.get("intent", "unknown"),
                "confidence": data.get("confidence", "unknown")
            })
            
            if show_response:
                print(f"\n🤖 پاسخ: {data['answer'][:200]}...")
                print(f"📊 Intent: {data.get('intent', 'N/A')}")
                print(f"🎯 Confidence: {data.get('confidence', 'N/A')}")
                print(f"📚 منابع: {data.get('num_sources', 0)} عدد")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ خطا در ارسال درخواست: {e}")
            return {}
    
    def get_session_info(self) -> Dict:
        """دریافت اطلاعات session"""
        if not self.session_id:
            print("⚠️ هیچ session فعالی وجود ندارد")
            return {}
        
        try:
            response = requests.get(f"{API_SESSION}/{self.session_id}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ خطا در دریافت session: {e}")
            return {}
    
    def clear_session(self) -> bool:
        """پاک کردن session"""
        if not self.session_id:
            print("⚠️ هیچ session فعالی وجود ندارد")
            return False
        
        try:
            response = requests.delete(f"{API_SESSION}/{self.session_id}", timeout=10)
            response.raise_for_status()
            print(f"✅ Session {self.session_id[:8]}... پاک شد")
            self.session_id = None
            self.conversation_history = []
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ خطا در پاک کردن session: {e}")
            return False
    
    def test_intent_detection(self):
        """تست تشخیص Intent"""
        self.print_separator("🔍 تست 1: تشخیص Intent")
        
        test_cases = [
            ("CA1626", "error_code"),
            ("کد خطای E15", "error_code"),
            ("چطور این مشکل رو حل کنم؟", "troubleshooting"),
            ("دستگاه کار نمیکنه", "troubleshooting"),
            ("شیر بای‌پس چیست؟", "information"),
            ("توضیح بده", "unclear"),
            ("مشکل", "unclear")
        ]
        
        print("\nتست موارد مختلف:\n")
        
        for question, expected_intent in test_cases:
            data = self.ask_question(question, show_response=False)
            detected_intent = data.get("intent", "unknown")
            
            status = "✅" if detected_intent == expected_intent else "❌"
            print(f"{status} '{question}' → انتظار: {expected_intent}, تشخیص: {detected_intent}")
            
            # پاک کردن session برای تست بعدی
            if self.session_id:
                self.clear_session()
            
            time.sleep(0.5)
    
    def test_conversation_memory(self):
        """تست حافظه مکالمه"""
        self.print_separator("💬 تست 2: حافظه مکالمه")
        
        print("\nشروع مکالمه پیوسته:\n")
        
        # مکالمه پیوسته
        questions = [
            "کد خطای CA1626 چیست؟",
            "علتش چیه؟",  # باید به CA1626 اشاره کنه
            "چطور رفعش کنم؟",  # هنوز در مورد CA1626
            "اگر جواب نداد چی؟"  # اشاره به راهکارهای قبلی
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- پیام {i} ---")
            self.ask_question(question)
            time.sleep(1)
        
        # بررسی تاریخچه در سرور
        print("\n📜 بررسی تاریخچه در سرور:")
        session_info = self.get_session_info()
        if session_info:
            print(f"✅ تعداد پیام‌ها: {session_info.get('message_count', 0)}")
            print(f"✅ Session ایجاد شده در: {session_info.get('created_at', 'N/A')}")
    
    def test_clarification(self):
        """تست سوالات روشن‌کننده"""
        self.print_separator("❓ تست 3: سوالات روشن‌کننده")
        
        print("\nتست سوالات مبهم:\n")
        
        unclear_questions = [
            "مشکل",
            "این",
            "چرا؟",
            "چی کار کنم"
        ]
        
        for question in unclear_questions:
            data = self.ask_question(question)
            needs_clarification = data.get("needs_clarification", False)
            
            if needs_clarification:
                print(f"✅ چت‌بات درست سوالات روشن‌کننده پرسید")
            else:
                print(f"⚠️ چت‌بات سوال روشن‌کننده نپرسید")
            
            # پاک کردن session
            if self.session_id:
                self.clear_session()
            
            time.sleep(0.5)
    
    def test_full_conversation(self):
        """تست یک مکالمه کامل"""
        self.print_separator("🎭 تست 4: مکالمه کامل و واقعی")
        
        print("\nشبیه‌سازی یک مکالمه واقعی با کاربر:\n")
        
        conversation = [
            "سلام",
            "دستگاهم مشکل داره",
            "کد CA135 نشون میده",
            "علت اول رو چک کردم ولی جواب نداد",
            "علت دوم چیه؟",
            "ممنون، الان تست میکنم"
        ]
        
        for message in conversation:
            self.ask_question(message)
            time.sleep(1.5)
        
        print("\n📊 خلاصه مکالمه:")
        print(f"تعداد پیام‌ها: {len(self.conversation_history)}")
        print(f"Session ID: {self.session_id[:8] if self.session_id else 'N/A'}...")
    
    def run_all_tests(self):
        """اجرای همه تست‌ها"""
        print("\n" + "🚀" * 40)
        print("   شروع تست قابلیت‌های جدید چت‌بات")
        print("🚀" * 40)
        
        try:
            # تست 1: Intent Detection
            self.test_intent_detection()
            time.sleep(2)
            
            # تست 2: Conversation Memory
            self.test_conversation_memory()
            time.sleep(2)
            
            # پاک کردن session قبلی
            if self.session_id:
                self.clear_session()
            time.sleep(1)
            
            # تست 3: Clarification Questions
            self.test_clarification()
            time.sleep(2)
            
            # پاک کردن session قبلی
            if self.session_id:
                self.clear_session()
            time.sleep(1)
            
            # تست 4: Full Conversation
            self.test_full_conversation()
            
            self.print_separator("✅ همه تست‌ها با موفقیت انجام شد!")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ تست توسط کاربر متوقف شد")
        except Exception as e:
            print(f"\n\n❌ خطای غیرمنتظره: {e}")
        finally:
            # پاک کردن session نهایی
            if self.session_id:
                print("\n🧹 پاک کردن session...")
                self.clear_session()


def check_server():
    """بررسی در دسترس بودن سرور"""
    print("🔍 بررسی اتصال به سرور...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        response.raise_for_status()
        print("✅ سرور در دسترس است")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ سرور در دسترس نیست: {e}")
        print("\n💡 لطفاً ابتدا سرور را با دستور زیر اجرا کنید:")
        print("   python run.py")
        return False


def main():
    """تابع اصلی"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🧪 اسکریپت تست قابلیت‌های جدید چت‌بات                    ║
║                                                              ║
║   این اسکریپت قابلیت‌های زیر را تست می‌کند:               ║
║   ✅ تشخیص Intent (نوع سوال)                               ║
║   ✅ حافظه مکالمه                                           ║
║   ✅ سوالات روشن‌کننده                                      ║
║   ✅ مکالمه پیوسته                                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # بررسی سرور
    if not check_server():
        return
    
    print("\n⏳ شروع تست‌ها در 3 ثانیه...")
    time.sleep(3)
    
    # اجرای تست‌ها
    tester = ChatbotTester()
    tester.run_all_tests()
    
    print("\n\n" + "=" * 80)
    print("   🎉 تست‌ها به پایان رسید!")
    print("=" * 80)
    print("\n💡 نکات:")
    print("   - برای مشاهده مستندات کامل: IMPROVEMENTS.md")
    print("   - برای راهنمای استفاده: CONVERSATION_GUIDE.md")
    print("   - برای مستندات API: http://localhost:8000/docs")
    print("\n")


if __name__ == "__main__":
    main()
