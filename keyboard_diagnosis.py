import os
import platform
import subprocess
import sys


def diagnose_keyboard_issue():
    """Comprehensive keyboard issue diagnosis."""
    print("🔍 تشخیص مشکل کیبورد فارسی")
    print("=" * 50)

    # Check OS
    print(f"سیستم عامل: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")

    # Test 1: Simple command execution
    print("\n📋 تست 1: اجرای دستور ساده")
    try:
        result = subprocess.run(['echo', 'test'], capture_output=True, text=True, timeout=5)
        print(f"خروجی: '{result.stdout.strip()}'")
        if 'ز' in result.stdout or 'ر' in result.stdout:
            print("⚠️ مشکل کیبورد تایید شد!")
        else:
            print("✅ دستور ساده بدون مشکل اجرا شد")
    except Exception as e:
        print(f"خطا در تست 1: {e}")

    # Test 2: Python command
    print("\n📋 تست 2: دستور Python")
    try:
        result = subprocess.run(['python', '--version'], capture_output=True, text=True, timeout=5)
        print(f"خروجی: '{result.stdout.strip()}'")
        if 'ز' in result.stdout or 'ر' in result.stdout:
            print("⚠️ مشکل کیبورد در Python تایید شد!")
        else:
            print("✅ دستور Python بدون مشکل اجرا شد")
    except Exception as e:
        print(f"خطا در تست 2: {e}")

    # Test 3: Environment variables
    print("\n📋 تست 3: متغیرهای محیطی")
    env_vars = ['LANG', 'LC_ALL', 'LC_CTYPE', 'INPUT_METHOD']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

    # Test 4: Registry check (Windows)
    if platform.system() == 'Windows':
        print("\n📋 تست 4: بررسی Registry (Windows)")
        try:
            result = subprocess.run(
                ['reg', 'query', 'HKEY_CURRENT_USER\\Keyboard Layout\\Preload'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            print("Registry keyboard layout:")
            print(result.stdout)
        except Exception as e:
            print(f"خطا در بررسی Registry: {e}")

    print("\n🎯 نتیجه‌گیری:")
    print("اگر در هر تستی 'ز' یا 'ر' ظاهر شد، مشکل کیبورد تایید است")
    print("در غیر این صورت، مشکل از جای دیگری است")


if __name__ == "__main__":
    diagnose_keyboard_issue()
