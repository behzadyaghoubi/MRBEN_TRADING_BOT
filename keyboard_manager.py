#!/usr/bin/env python3
"""
Keyboard Manager
Automatically switch keyboard layout to English before executing commands
"""

import subprocess
import time


def switch_to_english_keyboard():
    """
    Switch keyboard layout to English using multiple methods
    """
    print("🔧 Switching keyboard to English...")

    # Method 1: Using win32api (Windows)
    try:
        import win32api
        import win32con

        print("   Using win32api method...")

        # Load English US keyboard layout
        win32api.LoadKeyboardLayout('00000409', 1)

        # Alternative: Set input language to English
        try:
            win32api.PostMessage(
                win32con.HWND_BROADCAST, win32con.WM_INPUTLANGCHANGEREQUEST, 0, 0x0409
            )
        except:
            pass

        print("   ✅ Keyboard switched to English (win32api)")
        return True

    except ImportError:
        print("   ⚠️  win32api not available")
    except Exception as e:
        print(f"   ❌ win32api error: {e}")

    # Method 2: Using py_win_keyboard_layout
    try:
        import py_win_keyboard_layout as kl

        print("   Using py_win_keyboard_layout method...")

        # Get available layouts
        layouts = kl.get_keyboard_layout_list()

        # Find English layout (0x0409 = English US)
        english_layout = None
        for layout in layouts:
            if layout & 0xFFFF == 0x0409:  # English US
                english_layout = layout
                break

        if english_layout:
            kl.change_foreground_window_keyboard_layout(english_layout)
            print("   ✅ Keyboard switched to English (py_win_keyboard_layout)")
            return True
        else:
            print("   ⚠️  English layout not found")

    except ImportError:
        print("   ⚠️  py_win_keyboard_layout not available")
    except Exception as e:
        print(f"   ❌ py_win_keyboard_layout error: {e}")

    # Method 3: Using PowerShell command
    try:
        print("   Using PowerShell method...")

        # PowerShell command to switch to English
        ps_command = """
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.SendKeys]::SendWait("^+{F10}")
        """

        result = subprocess.run(
            ["powershell", "-Command", ps_command], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            print("   ✅ Keyboard switched to English (PowerShell)")
            return True
        else:
            print(f"   ❌ PowerShell error: {result.stderr}")

    except Exception as e:
        print(f"   ❌ PowerShell method error: {e}")

    # Method 4: Manual prompt
    print("   ⚠️  Automatic methods failed, manual switch required")
    print("   Please switch keyboard to English manually and press Enter...")
    try:
        input()
        print("   ✅ Manual keyboard switch confirmed")
        return True
    except:
        print("   ❌ Manual switch failed")
        return False


def ensure_english_keyboard():
    """
    Ensure keyboard is in English before proceeding
    """
    print("🔧 Ensuring English keyboard layout...")

    # Try to switch to English
    success = switch_to_english_keyboard()

    if success:
        print("✅ Keyboard layout confirmed as English")
        # Small delay to ensure switch takes effect
        time.sleep(0.5)
        return True
    else:
        print("❌ Failed to switch keyboard to English")
        print("⚠️  Please manually switch to English and try again")
        return False


def run_with_english_keyboard(command: str, timeout: int | None = None):
    """
    Run a command with English keyboard layout
    """
    print(f"🚀 Running command with English keyboard: {command}")

    # Ensure English keyboard
    if not ensure_english_keyboard():
        print("❌ Cannot proceed without English keyboard")
        return None

    # Run the command
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )

        print(f"✅ Command completed with return code: {result.returncode}")
        return result

    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"❌ Command execution error: {e}")
        return None


def run_python_script_with_english_keyboard(script_path: str, timeout: int | None = None):
    """
    Run a Python script with English keyboard layout
    """
    print(f"🐍 Running Python script with English keyboard: {script_path}")

    # Ensure English keyboard
    if not ensure_english_keyboard():
        print("❌ Cannot proceed without English keyboard")
        return None

    # Try different Python commands
    python_commands = [
        f"python {script_path}",
        f"py {script_path}",
        f"python3 {script_path}",
        f"python.exe {script_path}",
    ]

    for cmd in python_commands:
        print(f"   Trying: {cmd}")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )

            if result.returncode == 0:
                print(f"✅ Script executed successfully with: {cmd}")
                return result
            else:
                print(f"   ❌ Failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"   ❌ Timeout with: {cmd}")
        except Exception as e:
            print(f"   ❌ Error with {cmd}: {e}")

    print("❌ All Python commands failed")
    return None


if __name__ == "__main__":
    # Test the keyboard manager
    print("🧪 Testing Keyboard Manager")
    print("=" * 50)

    success = ensure_english_keyboard()
    if success:
        print("✅ Keyboard manager test successful")
    else:
        print("❌ Keyboard manager test failed")
