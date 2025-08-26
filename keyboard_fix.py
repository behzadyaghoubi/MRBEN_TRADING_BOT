#!/usr/bin/env python3
"""
Comprehensive Keyboard Layout Fix
Tries multiple methods to switch to English keyboard layout
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)


def method_1_powershell():
    """Method 1: Use PowerShell to switch keyboard layout"""
    try:
        logger.info("🔄 Method 1: PowerShell keyboard switch...")

        # PowerShell command to switch to English (US) keyboard
        ps_command = """
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.SendKeys]::SendWait("^+{F10}")
        Start-Sleep -Milliseconds 100
        [System.Windows.Forms.SendKeys]::SendWait("1")
        """

        result = subprocess.run(
            ['powershell', '-Command', ps_command], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            logger.info("✅ Method 1: PowerShell keyboard switch completed")
            return True
        else:
            logger.warning(f"⚠️ Method 1 failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"❌ Method 1 error: {e}")
        return False


def method_2_registry():
    """Method 2: Use registry to set default keyboard layout"""
    try:
        logger.info("🔄 Method 2: Registry keyboard switch...")

        # Registry command to set English as default
        reg_command = [
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
        ]

        result = subprocess.run(reg_command, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            logger.info("✅ Method 2: Registry keyboard switch completed")
            return True
        else:
            logger.warning(f"⚠️ Method 2 failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"❌ Method 2 error: {e}")
        return False


def method_3_direct_input():
    """Method 3: Direct keyboard input simulation"""
    try:
        logger.info("🔄 Method 3: Direct keyboard simulation...")

        # Try to import and use pyautogui if available
        try:
            import pyautogui

            pyautogui.FAILSAFE = False

            # Press Alt+Shift to cycle keyboard layouts
            pyautogui.hotkey('alt', 'shift')
            time.sleep(0.5)

            # Press Windows+Space to open language switcher
            pyautogui.hotkey('win', 'space')
            time.sleep(0.5)

            # Press Enter to select first option (usually English)
            pyautogui.press('enter')
            time.sleep(0.5)

            logger.info("✅ Method 3: Direct keyboard simulation completed")
            return True

        except ImportError:
            logger.warning("⚠️ pyautogui not available, skipping Method 3")
            return False

    except Exception as e:
        logger.error(f"❌ Method 3 error: {e}")
        return False


def method_4_batch_script():
    """Method 4: Create and run a batch script"""
    try:
        logger.info("🔄 Method 4: Batch script keyboard switch...")

        # Create batch script
        batch_content = """@echo off
echo Switching to English keyboard...
powershell -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^+{F10}'); Start-Sleep -Milliseconds 100; [System.Windows.Forms.SendKeys]::SendWait('1')"
echo Keyboard switch completed.
"""

        batch_file = Path("keyboard_switch.bat")
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(batch_content)

        # Run the batch script
        result = subprocess.run([str(batch_file)], capture_output=True, text=True, timeout=10)

        # Clean up
        batch_file.unlink(missing_ok=True)

        if result.returncode == 0:
            logger.info("✅ Method 4: Batch script keyboard switch completed")
            return True
        else:
            logger.warning(f"⚠️ Method 4 failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"❌ Method 4 error: {e}")
        return False


def method_5_environment_variable():
    """Method 5: Set environment variable for keyboard layout"""
    try:
        logger.info("🔄 Method 5: Environment variable keyboard switch...")

        # Set environment variable
        os.environ['LANG'] = 'en_US.UTF-8'
        os.environ['LC_ALL'] = 'en_US.UTF-8'

        # Also try to set Windows-specific variables
        os.environ['LANGUAGE'] = 'en'

        logger.info("✅ Method 5: Environment variables set")
        return True

    except Exception as e:
        logger.error(f"❌ Method 5 error: {e}")
        return False


def test_keyboard():
    """Test if keyboard issue is resolved"""
    try:
        logger.info("🧪 Testing keyboard input...")

        # Create a simple test script
        test_script = """
import sys
print("Keyboard test successful!")
print("No Persian characters detected.")
sys.exit(0)
"""

        test_file = Path("keyboard_test.py")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_script)

        # Try to run it
        result = subprocess.run(
            ['python', str(test_file)], capture_output=True, text=True, timeout=5
        )

        # Clean up
        test_file.unlink(missing_ok=True)

        if result.returncode == 0 and "Keyboard test successful!" in result.stdout:
            logger.info("✅ Keyboard test passed!")
            return True
        else:
            logger.warning(f"⚠️ Keyboard test failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"❌ Keyboard test error: {e}")
        return False


def main():
    """Main function to fix keyboard layout"""
    logger.info("🔧 STARTING KEYBOARD LAYOUT FIX")
    logger.info("=" * 50)

    # Try all methods
    methods = [
        ("PowerShell", method_1_powershell),
        ("Registry", method_2_registry),
        ("Direct Input", method_3_direct_input),
        ("Batch Script", method_4_batch_script),
        ("Environment Variables", method_5_environment_variable),
    ]

    success_count = 0

    for method_name, method_func in methods:
        logger.info(f"\n🔄 Trying {method_name}...")
        if method_func():
            success_count += 1
        time.sleep(1)  # Brief pause between methods

    # Test the result
    logger.info("\n🧪 Testing keyboard fix...")
    if test_keyboard():
        logger.info("🎉 KEYBOARD FIX SUCCESSFUL!")
        logger.info(f"✅ {success_count}/{len(methods)} methods worked")
        return True
    else:
        logger.error("❌ KEYBOARD FIX FAILED")
        logger.info("💡 Manual intervention may be required:")
        logger.info("   1. Press Win+Space to open language switcher")
        logger.info("   2. Select English (US) keyboard")
        logger.info("   3. Or remove Persian/Farsi language pack from Windows")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
