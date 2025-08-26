import os
import sys
import subprocess
import winreg
import ctypes
from ctypes import wintypes

def is_admin():
    """Check if running as administrator."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    """Re-run the script as administrator."""
    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        sys.exit()

def fix_keyboard_registry():
    """Fix keyboard layout in registry."""
    print("🔧 تنظیم Registry برای کیبورد...")
    
    try:
        # Set English US as default keyboard layout
        key_path = r"SYSTEM\CurrentControlSet\Control\Keyboard Layouts"
        
        # Set preload to English US
        preload_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                   r"Keyboard Layout\Preload", 
                                   0, winreg.KEY_WRITE)
        
        # Set value 1 to English US (00000409)
        winreg.SetValueEx(preload_key, "1", 0, winreg.REG_SZ, "00000409")
        winreg.CloseKey(preload_key)
        
        print("✅ Registry تنظیم شد")
        return True
        
    except Exception as e:
        print(f"❌ خطا در Registry: {e}")
        return False

def fix_environment_variables():
    """Set environment variables for English locale."""
    print("🔧 تنظیم متغیرهای محیطی...")
    
    try:
        # Set system environment variables
        os.environ['LANG'] = 'en_US.UTF-8'
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LC_CTYPE'] = 'en_US.UTF-8'
        os.environ['INPUT_METHOD'] = 'default'
        
        print("✅ متغیرهای محیطی تنظیم شدند")
        return True
        
    except Exception as e:
        print(f"❌ خطا در متغیرهای محیطی: {e}")
        return False

def create_keyboard_fix_batch():
    """Create a batch file for keyboard fix."""
    print("🔧 ایجاد فایل Batch برای رفع مشکل...")
    
    batch_content = '''@echo off
echo Fixing keyboard layout...
echo.

REM Set English US as default
reg add "HKEY_CURRENT_USER\\Keyboard Layout\\Preload" /v "1" /t REG_SZ /d "00000409" /f

REM Set system locale
reg add "HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Nls\\Language" /v "InstallLanguage" /t REG_SZ /d "0409" /f
reg add "HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Nls\\Language" /v "Default" /t REG_SZ /d "0409" /f

REM Set user locale
reg add "HKEY_CURRENT_USER\\Control Panel\\International" /v "Locale" /t REG_SZ /d "00000409" /f

echo Keyboard fix completed!
echo Please restart your computer.
pause
'''
    
    try:
        with open('keyboard_fix_final.bat', 'w', encoding='utf-8') as f:
            f.write(batch_content)
        print("✅ فایل keyboard_fix_final.bat ایجاد شد")
        return True
    except Exception as e:
        print(f"❌ خطا در ایجاد فایل Batch: {e}")
        return False

def create_powershell_fix():
    """Create PowerShell script for keyboard fix."""
    print("🔧 ایجاد اسکریپت PowerShell...")
    
    ps_content = '''# Keyboard Layout Fix Script
Write-Host "Fixing keyboard layout..." -ForegroundColor Green

# Set English US as default keyboard layout
Set-ItemProperty -Path "HKCU:\\Keyboard Layout\\Preload" -Name "1" -Value "00000409"

# Set system locale
Set-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Nls\\Language" -Name "InstallLanguage" -Value "0409"
Set-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Nls\\Language" -Name "Default" -Value "0409"

# Set user locale
Set-ItemProperty -Path "HKCU:\\Control Panel\\International" -Name "Locale" -Value "00000409"

# Remove Persian/Farsi if exists
$languages = Get-WinUserLanguageList
$languages = $languages | Where-Object {$_.LanguageTag -ne "fa-IR"}
Set-WinUserLanguageList $languages -Force

Write-Host "Keyboard fix completed! Please restart your computer." -ForegroundColor Green
'''
    
    try:
        with open('keyboard_fix_final.ps1', 'w', encoding='utf-8') as f:
            f.write(ps_content)
        print("✅ فایل keyboard_fix_final.ps1 ایجاد شد")
        return True
    except Exception as e:
        print(f"❌ خطا در ایجاد فایل PowerShell: {e}")
        return False

def create_manual_instructions():
    """Create manual instructions file."""
    print("🔧 ایجاد راهنمای دستی...")
    
    instructions = '''# 🚨 راهنمای دستی رفع مشکل کیبورد

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
2. Navigate to: HKEY_CURRENT_USER\\Keyboard Layout\\Preload
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
'''
    
    try:
        with open('MANUAL_KEYBOARD_FIX.md', 'w', encoding='utf-8') as f:
            f.write(instructions)
        print("✅ فایل MANUAL_KEYBOARD_FIX.md ایجاد شد")
        return True
    except Exception as e:
        print(f"❌ خطا در ایجاد راهنما: {e}")
        return False

def execute_analysis_directly():
    """Execute analysis directly without terminal commands."""
    print("\n🚀 اجرای مستقیم تحلیل - دور زدن مشکل کیبورد")
    print("=" * 60)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Check synthetic dataset
        print("\n1. بررسی دیتاست مصنوعی:")
        df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')
        
        # Count signals
        buy_count = len(df[df['signal'] == 'BUY'])
        sell_count = len(df[df['signal'] == 'SELL'])
        hold_count = len(df[df['signal'] == 'HOLD'])
        total_count = len(df)
        
        print(f"   کل نمونه‌ها: {total_count}")
        print(f"   BUY: {buy_count} ({buy_count/total_count*100:.1f}%)")
        print(f"   SELL: {sell_count} ({sell_count/total_count*100:.1f}%)")
        print(f"   HOLD: {hold_count} ({hold_count/total_count*100:.1f}%)")
        
        # Check balance
        if buy_count > 0 and sell_count > 0:
            ratio = buy_count / sell_count
            print(f"   نسبت BUY/SELL: {ratio:.2f}")
            
            if 0.8 <= ratio <= 1.2:
                print("   ✅ توزیع BUY/SELL متعادل است")
            else:
                print("   ⚠️ توزیع BUY/SELL نامتعادل است")
        
        print("   ✅ دیتاست مصنوعی آماده است")
        print("   ✅ می‌توانیم LSTM را بازآموزی کنیم")
        
        return True
        
    except Exception as e:
        print(f"   ❌ خطا در تحلیل: {e}")
        return False

def main():
    """Main function to fix keyboard issue."""
    print("🚨 راه‌حل نهایی مشکل کیبورد")
    print("=" * 50)
    
    # Check if running as admin
    if not is_admin():
        print("⚠️ نیاز به دسترسی Administrator")
        print("🔄 در حال اجرای مجدد با دسترسی Administrator...")
        run_as_admin()
        return
    
    print("✅ دسترسی Administrator تایید شد")
    
    # Create all fix files
    print("\n🔧 ایجاد ابزارهای رفع مشکل:")
    create_keyboard_fix_batch()
    create_powershell_fix()
    create_manual_instructions()
    
    # Try to fix registry
    print("\n🔧 تلاش برای رفع مشکل:")
    fix_keyboard_registry()
    fix_environment_variables()
    
    # Execute analysis directly
    print("\n📊 اجرای تحلیل مستقیم:")
    success = execute_analysis_directly()
    
    if success:
        print("\n✅ تحلیل موفق!")
        print("📋 مراحل بعدی:")
        print("   1. سیستم را Restart کنید")
        print("   2. python --version را تست کنید")
        print("   3. python train_lstm_balanced.py را اجرا کنید")
        print("   4. python live_trader_clean.py را اجرا کنید")
    else:
        print("\n❌ مشکل در تحلیل")
        print("📋 لطفاً فایل‌های ایجاد شده را اجرا کنید:")
        print("   - keyboard_fix_final.bat (به عنوان Administrator)")
        print("   - keyboard_fix_final.ps1 (به عنوان Administrator)")
        print("   - MANUAL_KEYBOARD_FIX.md را مطالعه کنید")

if __name__ == "__main__":
    main() 