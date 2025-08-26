import ctypes
import os
import subprocess
import winreg


def is_admin():
    """Check if running as administrator."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def fix_keyboard_permanently():
    """Permanent keyboard fix using Python."""
    print("🔧 راه‌حل قطعی مشکل کیبورد فارسی")
    print("=" * 50)

    if not is_admin():
        print("❌ این اسکریپت نیاز به Administrator دارد!")
        print("لطفاً PowerShell را به عنوان Administrator اجرا کنید")
        return False

    try:
        # Step 1: Set environment variables
        print("\n1. تنظیم متغیرهای محیطی...")
        os.environ['LANG'] = 'en_US.UTF-8'
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LC_CTYPE'] = 'en_US.UTF-8'
        os.environ['INPUT_METHOD'] = 'default'

        # Set permanent environment variables
        subprocess.run(['setx', 'LANG', 'en_US.UTF-8', '/M'], check=True)
        subprocess.run(['setx', 'LC_ALL', 'en_US.UTF-8', '/M'], check=True)
        subprocess.run(['setx', 'LC_CTYPE', 'en_US.UTF-8', '/M'], check=True)
        subprocess.run(['setx', 'INPUT_METHOD', 'default', '/M'], check=True)
        print("   ✅ متغیرهای محیطی تنظیم شدند")

        # Step 2: Registry fixes
        print("\n2. تنظیمات Registry...")

        # Set keyboard layout
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, r"Keyboard Layout\Preload", 0, winreg.KEY_WRITE
            ) as key:
                winreg.SetValueEx(key, "1", 0, winreg.REG_SZ, "00000409")
            print("   ✅ کیبورد به انگلیسی تنظیم شد")
        except Exception as e:
            print(f"   ⚠️ خطا در تنظیم کیبورد: {e}")

        # Set user language
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Control Panel\International\User Profile",
                0,
                winreg.KEY_WRITE,
            ) as key:
                winreg.SetValueEx(key, "Languages", 0, winreg.REG_MULTI_SZ, b"en-US\0")
            print("   ✅ زبان کاربر تنظیم شد")
        except Exception as e:
            print(f"   ⚠️ خطا در تنظیم زبان کاربر: {e}")

        # Step 3: System commands
        print("\n3. اجرای دستورات سیستم...")

        # Set system locale
        try:
            subprocess.run(
                ['powershell', 'Set-WinSystemLocale', '-SystemLocale', 'en-US'], check=True
            )
            print("   ✅ زبان سیستم تنظیم شد")
        except Exception as e:
            print(f"   ⚠️ خطا در تنظیم زبان سیستم: {e}")

        # Set user language list
        try:
            subprocess.run(
                ['powershell', 'Set-WinUserLanguageList', '-LanguageList', 'en-US', '-Force'],
                check=True,
            )
            print("   ✅ لیست زبان کاربر تنظیم شد")
        except Exception as e:
            print(f"   ⚠️ خطا در تنظیم لیست زبان: {e}")

        # Step 4: Test the fix
        print("\n4. تست رفع مشکل...")
        try:
            result = subprocess.run(
                ['python', '--version'], capture_output=True, text=True, timeout=5
            )
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


def create_restart_script():
    """Create a script to restart and test."""
    script_content = '''@echo off
echo ========================================
echo    Testing Keyboard Fix After Restart
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
        with open('test_after_restart.bat', 'w', encoding='utf-8') as f:
            f.write(script_content)
        print("   ✅ اسکریپت تست بعد از Restart ایجاد شد")
    except Exception as e:
        print(f"   ❌ خطا در ایجاد اسکریپت: {e}")


if __name__ == "__main__":
    success = fix_keyboard_permanently()

    if success:
        print("\n🎉 راه‌حل قطعی مشکل کیبورد اجرا شد!")
        print("لطفاً سیستم را Restart کنید تا تغییرات اعمال شود.")
        create_restart_script()
    else:
        print("\n⚠️ مشکل حل نشد. لطفاً از راه‌حل‌های دیگر استفاده کنید.")

    print("=" * 50)
