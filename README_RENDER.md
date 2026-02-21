# 🚀 دپلوی روی Render.com

## روش سریع (استفاده از Blueprint)

1. **ریپازیتوری را به Render متصل کنید**
   - وارد [render.com](https://render.com) شوید
   - "New +" > "Blueprint"
   - ریپازیتوری خود را انتخاب کنید

2. **Environment Variables را تنظیم کنید**
   - `OPENAI_API_KEY`: کلید OpenAI شما
   - `TAVILY_API_KEY`: کلید Tavily (اختیاری)

3. **دپلوی کنید!**
   - Render به صورت خودکار همه چیز را تنظیم می‌کند
   - منتظر بمانید تا deploy کامل شود

## روش دستی

### تنظیمات سرویس:

- **Build Command**: 
  ```bash
  pip install -r requirements.txt && python initialize_db.py || echo "Database init skipped"
  ```

- **Start Command**: 
  ```bash
  uvicorn api:app --host 0.0.0.0 --port $PORT
  ```

### Environment Variables:

```
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=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 (optional)
```

### فایل PDF:

- فایل PDF را در root پروژه قرار دهید و commit کنید
- یا از Persistent Disk استفاده کنید
- یا URL خارجی را در `PDF_FILE_PATH` قرار دهید

## نکات مهم

⚠️ **Free Plan**: سرویس بعد از 15 دقیقه sleep می‌شود

💾 **Persistent Disk**: برای نگهداری پایگاه داده، Persistent Disk را فعال کنید

🔑 **API Keys**: کلیدها را فقط در Environment Variables قرار دهید

## عیب‌یابی

- لاگ‌ها را در بخش "Logs" بررسی کنید
- اگر پایگاه داده راه‌اندازی نشد، از Web Interface استفاده کنید
- برای جزئیات بیشتر، `RENDER_DEPLOYMENT.md` را مطالعه کنید

## لینک‌های مفید

- [RENDER_QUICK_START.md](RENDER_QUICK_START.md) - راهنمای سریع
- [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) - راهنمای کامل
