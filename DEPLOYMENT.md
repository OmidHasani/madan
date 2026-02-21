# راهنمای دپلوی پروژه

این راهنما برای دپلوی پروژه روی هاست و دامنه است.

## پیش‌نیازها

1. Python 3.8 یا بالاتر
2. pip (مدیر بسته Python)
3. دسترسی به سرور (SSH)
4. دامنه و هاست (مثلاً cPanel، VPS، یا سرویس‌های ابری)

## مراحل دپلوی

### 1. آماده‌سازی محیط

```bash
# کلون کردن یا آپلود پروژه
cd /path/to/your/project

# ایجاد محیط مجازی
python3 -m venv venv
source venv/bin/activate  # در Windows: venv\Scripts\activate

# نصب وابستگی‌ها
pip install -r requirements.txt
```

### 2. تنظیم متغیرهای محیطی

```bash
# کپی کردن فایل .env.example به .env
cp .env.example .env

# ویرایش فایل .env و وارد کردن کلیدهای API
nano .env  # یا از ویرایشگر دلخواه استفاده کنید
```

مقادیر ضروری:
- `OPENAI_API_KEY`: کلید API OpenAI
- `TAVILY_API_KEY`: کلید API Tavily (اختیاری - برای جستجوی وب)
- `PDF_FILE_PATH`: مسیر فایل PDF
- `HOST` و `PORT`: آدرس و پورت سرور

### 3. راه‌اندازی پایگاه داده

```bash
# راه‌اندازی پایگاه داده برداری
python initialize_db.py
```

### 4. اجرای سرور

#### گزینه 1: استفاده از uvicorn مستقیم

```bash
python run.py
```

#### گزینه 2: استفاده از systemd (برای Linux)

ایجاد فایل `/etc/systemd/system/pdf-chatbot.service`:

```ini
[Unit]
Description=PDF Chatbot API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/your/project
Environment="PATH=/path/to/your/project/venv/bin"
ExecStart=/path/to/your/project/venv/bin/python run.py
Restart=always

[Install]
WantedBy=multi-user.target
```

سپس:
```bash
sudo systemctl daemon-reload
sudo systemctl enable pdf-chatbot
sudo systemctl start pdf-chatbot
```

#### گزینه 3: استفاده از Gunicorn + Nginx

```bash
# نصب Gunicorn
pip install gunicorn

# اجرا با Gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

تنظیم Nginx (مثال):

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /path/to/your/project/static;
    }
}
```

### 5. استفاده از SSL (HTTPS)

برای استفاده از HTTPS، می‌توانید از Let's Encrypt استفاده کنید:

```bash
# نصب certbot
sudo apt-get install certbot python3-certbot-nginx

# دریافت گواهینامه
sudo certbot --nginx -d yourdomain.com
```

### 6. پیکربندی Firewall

```bash
# باز کردن پورت 8000 (یا پورت دلخواه)
sudo ufw allow 8000/tcp

# یا اگر از Nginx استفاده می‌کنید:
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

## دپلوی روی سرویس‌های ابری

### Heroku

1. ایجاد فایل `Procfile`:
```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

2. ایجاد فایل `runtime.txt`:
```
python-3.11.0
```

3. دپلوی:
```bash
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your_key
heroku config:set TAVILY_API_KEY=your_key
git push heroku main
```

### Railway

1. اتصال ریپازیتوری به Railway
2. تنظیم متغیرهای محیطی در پنل Railway
3. Railway به صورت خودکار دپلوی می‌کند

### DigitalOcean App Platform

1. اتصال ریپازیتوری
2. تنظیم Build Command: `pip install -r requirements.txt`
3. تنظیم Run Command: `python run.py`
4. تنظیم متغیرهای محیطی

## نکات مهم

1. **امنیت**: هرگز فایل `.env` را در Git commit نکنید
2. **پورت**: مطمئن شوید پورت در فایروال باز است
3. **لاگ‌ها**: لاگ‌ها را برای دیباگ بررسی کنید
4. **بکاپ**: از پایگاه داده و فایل‌های مهم بکاپ بگیرید
5. **مانیتورینگ**: از سرویس‌های مانیتورینگ استفاده کنید

## عیب‌یابی

### مشکل: سرور شروع نمی‌شود
- بررسی کنید Python و pip نصب شده باشند
- بررسی کنید همه وابستگی‌ها نصب شده باشند
- بررسی کنید فایل `.env` وجود دارد و متغیرها صحیح هستند

### مشکل: خطای 500
- بررسی لاگ‌های سرور
- بررسی کنید پایگاه داده راه‌اندازی شده باشد
- بررسی کنید کلیدهای API معتبر باشند

### مشکل: خطای CORS
- بررسی کنید CORS در `api.py` به درستی تنظیم شده باشد

## پشتیبانی

برای مشکلات بیشتر، لاگ‌ها را بررسی کنید:
```bash
# اگر از systemd استفاده می‌کنید:
sudo journalctl -u pdf-chatbot -f

# یا لاگ‌های مستقیم:
python run.py
```
