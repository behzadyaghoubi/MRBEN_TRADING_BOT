# 🚨 دستورالعمل‌های نهایی - رفع مشکل کیبورد و ادامه پروژه

## **❌ مشکل فعلی:**
کاراکترهای "ز" و "ر" به صورت ناخواسته قبل از دستورات در ترمینال ظاهر می‌شوند.

## **🔧 راه‌حل‌های قطعی (به ترتیب اولویت):**

### **روش 1: تغییر زبان کیبورد (فوری)**
```
1. Windows + Space → English (US)
2. یا Alt + Shift → English
3. مطمئن شوید زبان روی English است
```

### **روش 2: تنظیمات ویندوز (دسترسی Administrator)**
```
1. Windows + I → Time & Language → Language & Region
2. Add a language → English (United States)
3. Set as default
4. Remove Persian/Farsi if exists
5. Restart سیستم
```

### **روش 3: تنظیمات کیبورد**
```
1. Settings → Devices → Typing → Advanced keyboard settings
2. Override for default input method → English (US)
3. Use language bar → Off
4. Restart سیستم
```

### **روش 4: متغیرهای محیطی**
```
1. Windows + R → sysdm.cpl → Advanced → Environment Variables
2. Add new system variables:
   - LANG = en_US.UTF-8
   - LC_ALL = en_US.UTF-8
   - LC_CTYPE = en_US.UTF-8
   - INPUT_METHOD = default
3. Restart سیستم
```

### **روش 5: Registry (دسترسی Administrator)**
```
1. Windows + R → regedit
2. Navigate to: HKEY_CURRENT_USER\Keyboard Layout\Preload
3. Set value "1" to "00000409" (English US)
4. Restart سیستم
```

## **🧪 تست بعد از رفع مشکل:**

### **تست 1: دستور Python**
```cmd
python --version
```
**نتیجه مطلوب:** `Python 3.x.x` (بدون حروف اضافی)

### **تست 2: تحلیل دیتاست**
```cmd
python run_analysis_directly.py
```
**نتیجه مطلوب:** تحلیل کامل دیتاست مصنوعی

### **تست 3: بازآموزی LSTM**
```cmd
python train_lstm_balanced.py
```
**نتیجه مطلوب:** آموزش موفق مدل LSTM

## **📋 مراحل بعدی (بعد از رفع مشکل):**

### **مرحله 1: تحلیل دیتاست مصنوعی**
```cmd
python run_analysis_directly.py
```

### **مرحله 2: بازآموزی LSTM**
```cmd
python train_lstm_balanced.py
```

### **مرحله 3: تست مدل جدید**
```cmd
python test_balanced_model.py
```

### **مرحله 4: اجرای سیستم کامل**
```cmd
python live_trader_clean.py
```

## **⚠️ نکات مهم:**

### **قبل از هر کاری:**
- **همیشه به عنوان Administrator اجرا کنید**
- **بعد از هر تغییر، سیستم را Restart کنید**
- **از Command Prompt به جای PowerShell استفاده کنید**
- **زبان فارسی را کاملاً حذف کنید**

### **اگر مشکل ادامه داشت:**
1. **Restart کامل کامپیوتر**
2. **حذف زبان فارسی از تنظیمات**
3. **نصب مجدد Python**
4. **استفاده از Command Prompt**

## **🎯 هدف نهایی:**
- ✅ رفع کامل مشکل کیبورد
- ✅ تحلیل دیتاست مصنوعی
- ✅ بازآموزی LSTM با داده متعادل
- ✅ اجرای سیستم کامل MR BEN
- ✅ کاهش bias در سیگنال‌ها

## **📞 پشتیبانی:**
اگر هیچ راه‌حلی کار نکرد:
1. Screenshot از خطا بگیرید
2. نسخه Windows را مشخص کنید
3. نوع کیبورد را مشخص کنید
4. از Command Prompt استفاده کنید

---

**🚀 MR BEN Trading System - Ready for Launch!**
