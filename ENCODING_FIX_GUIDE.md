# MR BEN AI System - Encoding Issue Fix Guide

## مشکل شناسایی شده
کاراکترهای اضافی "رز" در ابتدای دستورات که باعث خطا در اجرا می‌شود.

## راه‌حل‌های ارائه شده

### 1. اسکریپت Python امن (`run_system_update.py`)
این اسکریپت به طور خودکار کاراکترهای اضافی را حذف می‌کند:

```bash
python run_system_update.py
```

### 2. اسکریپت PowerShell (`run_system_update.ps1`)
اسکریپت PowerShell با تنظیمات encoding مناسب:

```powershell
powershell -ExecutionPolicy Bypass -File run_system_update.ps1
```

### 3. فایل Batch تست (`test_encoding.bat`)
برای تست مشکل encoding:

```bash
test_encoding.bat
```

## راه‌حل‌های سیستمی

### تنظیمات Windows
1. **Control Panel > Region & Language > Keyboards**
2. همه زبان‌ها را به جز English حذف کنید
3. زبان پیش‌فرض را روی English تنظیم کنید

### تنظیمات CMD
در CMD این دستور را اجرا کنید:
```cmd
chcp 65001
```

### تنظیمات PowerShell
در PowerShell این دستورات را اجرا کنید:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

## مراحل اجرا

### گام 1: تست سیستم
```bash
python test_system_update.py
```

### گام 2: اجرای به‌روزرسانی جامع
```bash
python fixed_comprehensive_update.py
```

### گام 3: بررسی نتایج
فایل‌های گزارش در پوشه `logs/` ذخیره می‌شوند.

## فایل‌های ایجاد شده

1. **`run_system_update.py`** - اسکریپت Python امن
2. **`run_system_update.ps1`** - اسکریپت PowerShell
3. **`test_encoding.bat`** - فایل Batch تست
4. **`fixed_comprehensive_update.py`** - به‌روزرسانی جامع
5. **`test_system_update.py`** - تست کامپوننت‌ها

## نکات مهم

- اگر مشکل همچنان ادامه دارد، از اسکریپت Python استفاده کنید
- تمام فایل‌های گزارش در پوشه `logs/` ذخیره می‌شوند
- در صورت بروز خطا، لاگ‌ها را بررسی کنید

## پشتیبانی

در صورت بروز مشکل، فایل‌های لاگ را بررسی کنید:
- `logs/execution_*.log`
- `logs/fixed_comprehensive_update_*.log`
- `logs/system_test_*.log`
