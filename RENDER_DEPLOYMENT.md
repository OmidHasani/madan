# راهنمای دپلوی روی Render.com

این راهنما به شما کمک می‌کند پروژه را روی Render.com دپلوی کنید. رابط وب به‌صورت **ریسپانسیو** است و روی موبایل، تبلت و دسکتاپ به‌درستی نمایش داده می‌شود.

## پیش‌نیازها

1. حساب کاربری در [Render.com](https://render.com)
2. ریپازیتوری Git (GitHub, GitLab, یا Bitbucket)
3. کلیدهای API (OpenAI و Tavily)

## مراحل دپلوی

### مرحله 1: آماده‌سازی ریپازیتوری

1. پروژه خود را به یک ریپازیتوری Git push کنید:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

### مرحله 2: ایجاد سرویس جدید در Render

1. وارد پنل Render شوید
2. روی "New +" کلیک کنید
3. "Web Service" را انتخاب کنید
4. ریپازیتوری خود را متصل کنید

### مرحله 3: تنظیمات سرویس

#### تنظیمات عمومی:
- **Name**: `pdf-chatbot` (یا نام دلخواه)
- **Environment**: `Python 3`
- **Region**: نزدیک‌ترین منطقه به شما
- **Branch**: `main` (یا branch اصلی شما)

#### Build & Start Commands:
- **Build Command**: 
  ```bash
  pip install -r requirements.txt
  ```
  
- **Start Command**: 
  ```bash
  uvicorn api:app --host 0.0.0.0 --port $PORT
  ```

### مرحله 4: تنظیم Environment Variables

در بخش "Environment" متغیرهای زیر را اضافه کنید:

#### متغیرهای ضروری:
```
OPENAI_API_KEY=your_openai_api_key_here
```

#### متغیرهای اختیاری:
```
TAVILY_API_KEY=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
EMBEDDING_MODEL=text-embedding-3-large
CHAT_MODEL=gpt-4o
PDF_FILE_PATH=shopmanual.pdf
VECTOR_DB_PATH=vector_db
COLLECTION_NAME=shop_manual
CHUNK_SIZE=2500
CHUNK_OVERLAP=500
```

### مرحله 5: آپلود فایل PDF

Render فایل‌های استاتیک را نگه می‌دارد، اما برای فایل PDF دو گزینه دارید:

#### گزینه 1: استفاده از Git LFS (برای فایل‌های بزرگ)
```bash
git lfs install
git lfs track "*.pdf"
git add shopmanual.pdf
git commit -m "Add PDF file"
git push
```

#### گزینه 2: استفاده از Render Disk (پیشنهادی)
1. در تنظیمات سرویس، "Persistent Disk" را فعال کنید
2. فایل PDF را از طریق SSH یا Build Script آپلود کنید

#### گزینه 3: استفاده از URL خارجی
اگر فایل PDF در جای دیگری است، می‌توانید URL آن را در `PDF_FILE_PATH` قرار دهید.

### مرحله 6: راه‌اندازی پایگاه داده

برای راه‌اندازی پایگاه داده، می‌توانید از یکی از روش‌های زیر استفاده کنید:

#### روش 1: استفاده از Build Command (پیشنهادی)
Build Command را به این صورت تغییر دهید:
```bash
pip install -r requirements.txt && python initialize_db.py || echo "Database already initialized"
```

#### روش 2: استفاده از Post-Deploy Script
یک فایل `post_deploy.sh` ایجاد کنید:
```bash
#!/bin/bash
python initialize_db.py || echo "Database already initialized"
```

سپس در Start Command:
```bash
python post_deploy.sh && uvicorn api:app --host 0.0.0.0 --port $PORT
```

### مرحله 7: دپلوی

1. روی "Create Web Service" کلیک کنید
2. Render شروع به build و deploy می‌کند
3. منتظر بمانید تا دپلوی کامل شود (معمولاً 5-10 دقیقه)

### مرحله 8: بررسی و تست

1. بعد از دپلوی، URL سرویس شما نمایش داده می‌شود
2. به URL بروید و تست کنید
3. اگر مشکلی بود، لاگ‌ها را در بخش "Logs" بررسی کنید

## استفاده از render.yaml (گزینه پیشرفته)

اگر می‌خواهید از فایل `render.yaml` استفاده کنید:

1. فایل `render.yaml` را در root پروژه قرار دهید
2. در Render، "New +" > "Blueprint" را انتخاب کنید
3. ریپازیتوری را متصل کنید
4. Render به صورت خودکار تنظیمات را از `render.yaml` می‌خواند

## نکات مهم

### 1. محدودیت‌های Render
- **Free Plan**: 
  - 750 ساعت در ماه
  - Sleep بعد از 15 دقیقه عدم استفاده
  - 512MB RAM
  - Disk محدود

- **Paid Plan**: 
  - همیشه روشن
  - RAM بیشتر
  - Disk بیشتر

### 2. مدیریت فایل PDF
- فایل PDF باید در ریپازیتوری یا Persistent Disk باشد
- برای فایل‌های بزرگ (>100MB) از Git LFS استفاده کنید
- یا از URL خارجی استفاده کنید

### 3. پایگاه داده Vector
- Vector database در Persistent Disk ذخیره می‌شود
- بعد از هر deploy، اگر Persistent Disk نداشته باشید، داده‌ها پاک می‌شوند
- برای production، Persistent Disk را فعال کنید

### 4. Environment Variables
- هرگز کلیدهای API را در کد commit نکنید
- فقط از Environment Variables استفاده کنید
- در Render، Environment Variables را به صورت Secret تنظیم کنید

### 5. لاگ‌ها
- لاگ‌ها را در بخش "Logs" بررسی کنید
- برای دیباگ، می‌توانید `print` یا `logger` استفاده کنید

## عیب‌یابی

### مشکل: Build فیل می‌شود
- بررسی کنید همه dependencies در `requirements.txt` هستند
- بررسی کنید Python version صحیح است
- لاگ‌های Build را بررسی کنید

### مشکل: سرویس شروع نمی‌شود
- بررسی کنید Start Command صحیح است
- بررسی کنید Environment Variables تنظیم شده‌اند
- بررسی کنید فایل PDF موجود است

### مشکل: خطای 500
- بررسی کنید پایگاه داده راه‌اندازی شده است
- بررسی کنید کلیدهای API معتبر هستند
- لاگ‌های Runtime را بررسی کنید

### مشکل: فایل PDF پیدا نمی‌شود
- بررسی کنید `PDF_FILE_PATH` صحیح است
- بررسی کنید فایل در ریپازیتوری commit شده است
- یا از Persistent Disk استفاده کنید

## بهینه‌سازی

### برای Production:
1. از Paid Plan استفاده کنید (همیشه روشن)
2. Persistent Disk را فعال کنید
3. Environment Variables را به صورت Secret تنظیم کنید
4. Health Check را فعال کنید
5. Auto-Deploy را تنظیم کنید

### Health Check:
در تنظیمات سرویس، Health Check Path را تنظیم کنید:
```
/api/health
```

## پشتیبانی

اگر مشکلی پیش آمد:
1. لاگ‌ها را در Render بررسی کنید
2. مستندات Render را مطالعه کنید
3. Community Render را بررسی کنید

## لینک‌های مفید

- [Render Documentation](https://render.com/docs)
- [Python on Render](https://render.com/docs/deploy-python)
- [Environment Variables](https://render.com/docs/environment-variables)
- [Persistent Disks](https://render.com/docs/disks)
