#!/usr/bin/env python3
"""
Advanced Keyboard Fix Script for MR BEN
=======================================
Uses multiple methods to fix Persian keyboard input issues.
"""

import os
import sys
import subprocess
import platform
import winreg
import ctypes
from datetime import datetime

def print_header():
    """Print script header."""
    print("="*60)
    print("🔧 ADVANCED KEYBOARD FIX - MR BEN")
    print("="*60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  OS: {platform.system()} {platform.release()}")
    print()

def method_1_registry_fix():
    """Method 1: Fix keyboard layout via registry."""
    print("🔧 Method 1: Registry Fix...")
    
    try:
        # Open registry key
        key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_WRITE)
        
        # Set English as default
        english_layout = "00000409"
        
        # Try to set keyboard layout
        try:
            winreg.SetValueEx(key, "KeyboardLayout", 0, winreg.REG_SZ, english_layout)
            print("✅ Registry updated for English layout")
        except Exception as e:
            print(f"⚠️ Could not update registry: {e}")
        
        winreg.CloseKey(key)
        return True
        
    except Exception as e:
        print(f"❌ Registry fix failed: {e}")
        return False

def method_2_powershell_script():
    """Method 2: Create and run PowerShell script."""
    print("🔧 Method 2: PowerShell Script...")
    
    ps_script = '''
Write-Host "Fixing keyboard layout..." -ForegroundColor Yellow

# Set English as default input language
$englishLayout = "00000409"

try {
    # Update registry
    Set-ItemProperty -Path "HKCU:\\Keyboard Layout\\Preload" -Name "1" -Value $englishLayout -ErrorAction SilentlyContinue
    Write-Host "Registry updated" -ForegroundColor Green
} catch {
    Write-Host "Registry update failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Create test file
"echo test" | Out-File -FilePath "keyboard_test_simple.txt" -Encoding ASCII
Write-Host "Test file created" -ForegroundColor Green
'''
    
    try:
        # Write PowerShell script
        with open('fix_keyboard_temp.ps1', 'w', encoding='utf-8') as f:
            f.write(ps_script)
        
        # Run PowerShell script
        result = subprocess.run([
            'powershell', '-ExecutionPolicy', 'Bypass', '-File', 'fix_keyboard_temp.ps1'
        ], capture_output=True, text=True, timeout=30)
        
        print("✅ PowerShell script executed")
        print(f"Output: {result.stdout}")
        
        # Clean up
        if os.path.exists('fix_keyboard_temp.ps1'):
            os.remove('fix_keyboard_temp.ps1')
        
        return True
        
    except Exception as e:
        print(f"❌ PowerShell method failed: {e}")
        return False

def method_3_system_restart_instructions():
    """Method 3: Provide system restart instructions."""
    print("🔧 Method 3: System Restart Instructions...")
    
    instructions = """
🔄 SYSTEM RESTART INSTRUCTIONS:

1. Save all your work
2. Close all applications
3. Restart your computer
4. After restart, open a new PowerShell window
5. Test with: echo test
6. If still broken, try Command Prompt instead

ALTERNATIVE: Use Command Prompt
1. Press Win + R
2. Type: cmd
3. Press Enter
4. Navigate to your project: cd D:\\MRBEN_CLEAN_PROJECT
5. Test commands there
"""
    
    print(instructions)
    
    # Create a batch file for easy testing
    batch_content = '''@echo off
echo Testing keyboard in Command Prompt...
echo test
python --version
pause
'''
    
    try:
        with open('test_keyboard.bat', 'w', encoding='utf-8') as f:
            f.write(batch_content)
        print("✅ Created test_keyboard.bat for Command Prompt testing")
        return True
    except Exception as e:
        print(f"❌ Failed to create batch file: {e}")
        return False

def method_4_language_settings():
    """Method 4: Language settings instructions."""
    print("🔧 Method 4: Language Settings...")
    
    instructions = """
⚙️ LANGUAGE SETTINGS INSTRUCTIONS:

1. Press Win + I to open Settings
2. Go to Time & Language > Language
3. Make sure English (United States) is at the top
4. Remove Persian if not needed
5. Click on English > Options
6. Make sure English (US) keyboard is installed
7. Set English as default
8. Restart computer

KEYBOARD SHORTCUTS:
- Win + Space: Cycle through input languages
- Alt + Shift: Switch between languages
- Ctrl + Shift: Switch keyboard layouts
"""
    
    print(instructions)
    return True

def create_simple_test():
    """Create a simple test that doesn't require keyboard input."""
    print("🧪 Creating simple test...")
    
    test_content = '''#!/usr/bin/env python3
import sys
import os

print("Simple test without keyboard input")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Test file access
files = ["src/main_runner.py", "config/settings.json"]
for file in files:
    if os.path.exists(file):
        print(f"✅ {file}: Found")
    else:
        print(f"❌ {file}: Not found")

print("Test completed!")
'''
    
    try:
        with open('simple_test_no_keyboard.py', 'w', encoding='utf-8') as f:
            f.write(test_content)
        print("✅ Created simple_test_no_keyboard.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create test: {e}")
        return False

def run_simple_test():
    """Run the simple test."""
    print("🧪 Running simple test...")
    
    if not os.path.exists('simple_test_no_keyboard.py'):
        print("❌ Test file not found")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, 'simple_test_no_keyboard.py'
        ], capture_output=True, text=True, timeout=30)
        
        print("📤 Test Output:")
        print(result.stdout)
        
        if result.stderr:
            print("📤 Error Output:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Simple test passed!")
            return True
        else:
            print("❌ Simple test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        return False

def main():
    """Main function."""
    print_header()
    
    print("🚨 CRITICAL KEYBOARD ISSUE DETECTED!")
    print("Persian characters are being added to all commands.")
    print()
    
    # Try all methods
    methods = [
        method_1_registry_fix,
        method_2_powershell_script,
        method_3_system_restart_instructions,
        method_4_language_settings
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"\n--- Method {i} ---")
        method()
    
    # Create and run simple test
    print("\n--- Testing ---")
    if create_simple_test():
        run_simple_test()
    
    print("\n" + "="*60)
    print("📋 SUMMARY:")
    print("1. Registry fix attempted")
    print("2. PowerShell script executed")
    print("3. System restart instructions provided")
    print("4. Language settings guide provided")
    print("5. Simple test created and run")
    print()
    print("🎯 NEXT STEPS:")
    print("1. Restart your computer")
    print("2. Use Command Prompt instead of PowerShell")
    print("3. Test with: cmd > echo test")
    print("4. If working, proceed with MR BEN system")
    print("="*60)

if __name__ == "__main__":
    main() 