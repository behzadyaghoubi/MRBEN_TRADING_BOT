import os
import sys
import subprocess
import winreg
import ctypes

def emergency_keyboard_fix():
    """Emergency keyboard fix without admin privileges."""
    print("🚨 راه‌حل اضطراری مشکل کیبورد")
    print("=" * 50)
    
    try:
        # Step 1: Set environment variables immediately
        print("\n1. تنظیم متغیرهای محیطی...")
        os.environ['LANG'] = 'en_US.UTF-8'
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LC_CTYPE'] = 'en_US.UTF-8'
        os.environ['INPUT_METHOD'] = 'default'
        print("   ✅ متغیرهای محیطی تنظیم شدند")
        
        # Step 2: Try to set permanent environment variables
        print("\n2. تنظیم متغیرهای محیطی دائمی...")
        try:
            subprocess.run(['setx', 'LANG', 'en_US.UTF-8'], check=True, capture_output=True)
            subprocess.run(['setx', 'LC_ALL', 'en_US.UTF-8'], check=True, capture_output=True)
            subprocess.run(['setx', 'LC_CTYPE', 'en_US.UTF-8'], check=True, capture_output=True)
            subprocess.run(['setx', 'INPUT_METHOD', 'default'], check=True, capture_output=True)
            print("   ✅ متغیرهای محیطی دائمی تنظیم شدند")
        except Exception as e:
            print(f"   ⚠️ خطا در تنظیم متغیرهای دائمی: {e}")
        
        # Step 3: Try registry fixes
        print("\n3. تنظیمات Registry...")
        try:
            # Set keyboard layout
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Keyboard Layout\Preload", 0, winreg.KEY_WRITE) as key:
                winreg.SetValueEx(key, "1", 0, winreg.REG_SZ, "00000409")
            print("   ✅ کیبورد به انگلیسی تنظیم شد")
        except Exception as e:
            print(f"   ⚠️ خطا در تنظیم کیبورد: {e}")
        
        # Step 4: Test the fix
        print("\n4. تست رفع مشکل...")
        try:
            # Test with subprocess
            result = subprocess.run(['python', '--version'], capture_output=True, text=True, timeout=5)
            output = result.stdout.strip()
            
            if 'ز' in output or 'ر' in output:
                print(f"   ❌ مشکل همچنان وجود دارد: {output}")
                return False
            else:
                print(f"   ✅ مشکل حل شد: {output}")
                return True
        except Exception as e:
            print(f"   ⚠️ خطا در تست: {e}")
            return False
        
    except Exception as e:
        print(f"❌ خطای کلی: {e}")
        return False

def create_manual_guide():
    """Create manual fix guide."""
    guide_content = '''# 🚨 راهنمای دستی رفع مشکل کیبورد

## مشکل: کاراکترهای "ز" و "ر" قبل از دستورات

## راه‌حل‌های دستی:

### 1. تغییر زبان کیبورد (فوری)
- Windows + Space → English (US)
- یا Alt + Shift → English

### 2. تنظیمات ویندوز
1. Windows + I → Time & Language → Language & Region
2. Add a language → English (United States)
3. Set as default
4. Remove Persian/Farsi if exists

### 3. تنظیمات کیبورد
1. Settings → Devices → Typing → Advanced keyboard settings
2. Override for default input method → English (US)

### 4. متغیرهای محیطی
1. Windows + R → sysdm.cpl → Advanced → Environment Variables
2. Add new system variables:
   - LANG = en_US.UTF-8
   - LC_ALL = en_US.UTF-8
   - LC_CTYPE = en_US.UTF-8
   - INPUT_METHOD = default

### 5. Restart سیستم
بعد از هر تغییر، سیستم را Restart کنید.

## تست بعد از رفع مشکل:
```cmd
python --version
python final_analysis.py
```

## اگر مشکل ادامه داشت:
1. از Command Prompt به جای PowerShell استفاده کنید
2. زبان فارسی را کاملاً حذف کنید
3. تنظیمات کیبورد را ریست کنید
'''
    
    try:
        with open('MANUAL_KEYBOARD_FIX.md', 'w', encoding='utf-8') as f:
            f.write(guide_content)
        print("   ✅ راهنمای دستی ایجاد شد: MANUAL_KEYBOARD_FIX.md")
    except Exception as e:
        print(f"   ❌ خطا در ایجاد راهنما: {e}")

def create_test_script():
    """Create a test script."""
    test_content = '''@echo off
echo ========================================
echo    Keyboard Fix Test
echo ========================================
echo.

echo Testing Python command...
python --version
echo.

echo Testing analysis script...
python final_analysis.py
echo.

echo ========================================
echo    Test completed!
echo ========================================
pause
'''
    
    try:
        with open('test_keyboard_fix.bat', 'w', encoding='utf-8') as f:
            f.write(test_content)
        print("   ✅ اسکریپت تست ایجاد شد: test_keyboard_fix.bat")
    except Exception as e:
        print(f"   ❌ خطا در ایجاد اسکریپت تست: {e}")

if __name__ == "__main__":
    success = emergency_keyboard_fix()
    
    if success:
        print("\n🎉 راه‌حل اضطراری موفق بود!")
        print("حالا می‌توانید تحلیل دیتاست را اجرا کنید:")
        print("python final_analysis.py")
    else:
        print("\n⚠️ راه‌حل اضطراری موفق نبود!")
        print("لطفاً از راهنمای دستی استفاده کنید.")
        create_manual_guide()
    
    create_test_script()
    print("\n📋 فایل‌های ایجاد شده:")
    print("   - MANUAL_KEYBOARD_FIX.md (راهنمای دستی)")
    print("   - test_keyboard_fix.bat (اسکریپت تست)")
    print("=" * 50) 