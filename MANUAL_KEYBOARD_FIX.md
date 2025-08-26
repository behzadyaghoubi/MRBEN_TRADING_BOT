# 🚨 راهنمای دستی رفع مشکل کیبورد

## روش 1: تغییر زبان کیبورد (فوری)
1. Windows + Space → English (US)
2. یا Alt + Shift → English
3. مطمئن شوید زبان روی English است

## روش 2: تنظیمات ویندوز (دسترسی Administrator)
1. Windows + I → Time & Language → Language & Region
2. Add a language → English (United States)
3. Set as default
4. Remove Persian/Farsi if exists
5. Restart سیستم

## روش 3: تنظیمات کیبورد
1. Settings → Devices → Typing → Advanced keyboard settings
2. Override for default input method → English (US)
3. Use language bar → Off
4. Restart سیستم

## روش 4: متغیرهای محیطی
1. Windows + R → sysdm.cpl → Advanced → Environment Variables
2. Add new system variables:
   - LANG = en_US.UTF-8
   - LC_ALL = en_US.UTF-8
   - LC_CTYPE = en_US.UTF-8
   - INPUT_METHOD = default
3. Restart سیستم

## روش 5: Registry (دسترسی Administrator)
1. Windows + R → regedit
2. Navigate to: HKEY_CURRENT_USER\Keyboard Layout\Preload
3. Set value "1" to "00000409" (English US)
4. Restart سیستم

## تست بعد از رفع مشکل:
python --version
python execute_analysis_directly.py

## نکات مهم:
- همیشه به عنوان Administrator اجرا کنید
- بعد از هر تغییر، سیستم را Restart کنید
- از Command Prompt به جای PowerShell استفاده کنید
- زبان فارسی را کاملاً حذف کنید
