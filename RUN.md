# راهنمای اجرا و به‌روزرسانی

## الان دقیقا چیکار کنم؟

### حالت ۱: فقط کد RAG/API/فرانت عوض شده (بدون عوض کردن PDF یا TOC یا chunking)

**نیازی به دوباره اجرا کردن embedding نیست.**

1. سرور را ریستارت کن:
   ```bash
   # اگر با uvicorn مستقیم اجرا می‌کنی، همان پروسس را ببند و دوباره اجرا کن:
   python api.py
   # یا
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
2. مرورگر را رفرش کن و تست کن.

---

### حالت ۲: PDF عوض شده، یا TOC عیب‌یابی (troubleshooting_toc.py) عوض شده، یا می‌خواهی متادیتای جدید (مثل section_code) روی همهٔ چانک‌ها باشد

**باید یک بار embedding را از نو بسازی.**

1. **دیتابیس بردکتور فعلی را پاک کن** (تا چانک‌ها و embeddingها از نو ساخته شوند):
   ```bash
   # در پوشهٔ پروژه (همان جایی که vector_db قرار دارد)
   # پوشهٔ vector_db را حذف کن یا محتویاتش را خالی کن:
   # Windows (PowerShell):
   Remove-Item -Recurse -Force vector_db
   # یا فقط محتویات:
   Remove-Item -Recurse -Force vector_db\*
   ```

2. **ایندکس و embedding را دوباره بساز:**
   - **روش الف (توصیه‌شده):** اسکریپت initialize را اجرا کن:
     ```bash
     python initialize_db.py
     ```
     این کار PDF را می‌خواند، چانک می‌سازد، متادیتا (از جمله section_code / section_title از TOC) را غنی می‌کند و embedding می‌سازد و در `vector_db` ذخیره می‌کند.

   - **روش ب:** سرور را بدون پاک کردن دستی اجرا کن؛ اگر `vector_db` خالی باشد، در startup خود API با `auto_initialize` همان کار را می‌کند (PDF را پردازش و ذخیره می‌کند). پس اگر `vector_db` را پاک کرده باشی و بعد `python api.py` بزنی، خودش دوباره همه چیز را می‌سازد.

3. **بعد از اتمام initialize، سرور را اجرا کن:**
   ```bash
   python api.py
   ```

---

### خلاصهٔ تصمیم

| وضعیت | کار تو |
|--------|--------|
| فقط کد برنامه (RAG، API، UI) عوض شده | فقط سرور را ریستارت کن؛ **embedding را دوباره نزن**. |
| PDF عوض شده یا TOC (صفحات کدهای خطا) عوض شده | `vector_db` را پاک کن → `python initialize_db.py` (یا اجرای سرور با DB خالی) → بعد سرور را اجرا کن. |
| می‌خواهی متادیتای جدید (section_code و ...) روی چانک‌ها باشد | مثل بالا: یک بار DB را پاک کن و دوباره `initialize_db.py` یا startup با DB خالی. |

---

## نکات

- **Faiss:** اگر `faiss-cpu` نصب باشد، جستجو سریع‌تر است: `pip install faiss-cpu`
- **مدل embedding:** در `.env` می‌توانی `EMBEDDING_MODEL=text-embedding-3-large` بگذاری برای کیفیت بالاتر (با هزینهٔ بیشتر).
- **MMR و ریرنک:** در `config` / `.env` می‌توانی `RERANK_INITIAL_K`, `USE_MMR`, `MMR_LAMBDA` را تنظیم کنی (پیش‌فرض: MMR روشن، lambda=0.7).
