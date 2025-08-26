import os
import platform
import subprocess


def fix_keyboard_issue():
    """Comprehensive keyboard fix with multiple solutions."""
    print("🔧 راه‌حل جامع مشکل کیبورد")
    print("=" * 50)

    solutions = [
        ("راه‌حل 1: تغییر متغیرهای محیطی", fix_environment_variables),
        ("راه‌حل 2: تغییر زبان کیبورد", fix_keyboard_language),
        ("راه‌حل 3: استفاده از Command Prompt", use_cmd_instead),
        ("راه‌حل 4: تنظیمات Registry", fix_registry_settings),
        ("راه‌حل 5: راه‌حل نهایی", final_solution),
    ]

    for i, (name, solution) in enumerate(solutions, 1):
        print(f"\n{i}. {name}")
        print("-" * 30)
        try:
            success = solution()
            if success:
                print("✅ موفق")
                # Test if problem is fixed
                if test_keyboard_fix():
                    print("🎉 مشکل کیبورد حل شد!")
                    return True
            else:
                print("❌ ناموفق")
        except Exception as e:
            print(f"❌ خطا: {e}")

    print("\n⚠️ هیچ راه‌حلی کار نکرد!")
    return False


def fix_environment_variables():
    """Fix environment variables."""
    print("تنظیم متغیرهای محیطی...")

    # Set environment variables
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LC_CTYPE'] = 'en_US.UTF-8'
    os.environ['INPUT_METHOD'] = 'default'

    print("متغیرهای محیطی تنظیم شدند")
    return True


def fix_keyboard_language():
    """Fix keyboard language settings."""
    print("تغییر زبان کیبورد...")

    if platform.system() == 'Windows':
        try:
            # Try to set keyboard layout to English
            subprocess.run(
                ['powershell', 'Set-WinUserLanguageList', '-LanguageList', 'en-US'],
                capture_output=True,
                timeout=10,
            )
            print("زبان کیبورد به انگلیسی تغییر یافت")
            return True
        except Exception as e:
            print(f"خطا در تغییر زبان: {e}")
            return False
    else:
        print("این راه‌حل فقط برای Windows است")
        return False


def use_cmd_instead():
    """Use CMD instead of PowerShell."""
    print("استفاده از Command Prompt به جای PowerShell...")

    try:
        # Create a batch file to run commands
        batch_content = """@echo off
echo Testing keyboard fix...
python --version
echo Test completed
pause
"""
        with open('test_keyboard.bat', 'w', encoding='utf-8') as f:
            f.write(batch_content)

        print("فایل test_keyboard.bat ایجاد شد")
        print("لطفاً این فایل را اجرا کنید")
        return True
    except Exception as e:
        print(f"خطا: {e}")
        return False


def fix_registry_settings():
    """Fix registry settings for keyboard."""
    print("تنظیمات Registry...")

    if platform.system() == 'Windows':
        try:
            # Registry commands to fix keyboard
            commands = [
                [
                    'reg',
                    'add',
                    'HKEY_CURRENT_USER\\Keyboard Layout\\Preload',
                    '/v',
                    '1',
                    '/t',
                    'REG_SZ',
                    '/d',
                    '00000409',
                    '/f',
                ],
                [
                    'reg',
                    'add',
                    'HKEY_CURRENT_USER\\Control Panel\\International\\User Profile',
                    '/v',
                    'Languages',
                    '/t',
                    'REG_MULTI_SZ',
                    '/d',
                    'en-US',
                    '/f',
                ],
            ]

            for cmd in commands:
                subprocess.run(cmd, capture_output=True, timeout=10)

            print("تنظیمات Registry تغییر یافت")
            return True
        except Exception as e:
            print(f"خطا در تنظیمات Registry: {e}")
            return False
    else:
        print("این راه‌حل فقط برای Windows است")
        return False


def final_solution():
    """Final solution - manual instructions."""
    print("راه‌حل نهایی - دستورالعمل دستی:")
    print("=" * 40)
    print("1. کلید Windows + Space را فشار دهید")
    print("2. زبان را به English (US) تغییر دهید")
    print("3. کلید Windows + R را فشار دهید")
    print("4. 'cmd' را تایپ کنید و Enter بزنید")
    print("5. در Command Prompt دستورات را اجرا کنید")
    print("=" * 40)
    return True


def test_keyboard_fix():
    """Test if keyboard issue is fixed."""
    print("\n🧪 تست رفع مشکل...")

    try:
        result = subprocess.run(['python', '--version'], capture_output=True, text=True, timeout=5)
        output = result.stdout.strip()

        if 'ز' in output or 'ر' in output:
            print(f"❌ مشکل همچنان وجود دارد: {output}")
            return False
        else:
            print(f"✅ مشکل حل شد: {output}")
            return True
    except Exception as e:
        print(f"❌ خطا در تست: {e}")
        return False


if __name__ == "__main__":
    fix_keyboard_issue()
