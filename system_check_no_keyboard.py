#!/usr/bin/env python3
"""
MR BEN - System Check (No Keyboard Required)
===========================================
Comprehensive system check that doesn't require keyboard input.
"""

import sys
import os
import platform
import subprocess
from datetime import datetime

def print_header():
    """Print script header."""
    print("="*60)
    print("🔍 MR BEN - SYSTEM CHECK (NO KEYBOARD)")
    print("="*60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  OS: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    print()

def check_python_packages():
    """Check Python packages."""
    print("📦 Checking Python packages...")
    
    packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('joblib', 'joblib'),
        ('tensorflow', 'tf'),
        ('matplotlib', 'plt')
    ]
    
    installed = []
    missing = []
    
    for package, import_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
            installed.append(package)
        except ImportError:
            print(f"❌ {package}: Not installed")
            missing.append(package)
        except Exception as e:
            print(f"⚠️ {package}: Error - {e}")
            missing.append(package)
    
    return installed, missing

def check_files():
    """Check important files."""
    print("\n📁 Checking important files...")
    
    files = [
        "src/main_runner.py",
        "config/settings.json",
        "src/settings.json",
        "models/mrben_ai_signal_filter_xgb.joblib",
        "models/mrben_lstm_model.h5",
        "data/XAUUSD_PRO_M5_live.csv",
        "requirements.txt"
    ]
    
    found = []
    missing = []
    
    for file_path in files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path}: {size} bytes")
            found.append(file_path)
        else:
            print(f"❌ {file_path}: Not found")
            missing.append(file_path)
    
    return found, missing

def check_directories():
    """Check important directories."""
    print("\n📂 Checking directories...")
    
    directories = [
        "src",
        "config",
        "models",
        "data",
        "strategies",
        "logs"
    ]
    
    for directory in directories:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✅ {directory}/: Directory exists")
        else:
            print(f"❌ {directory}/: Directory missing")

def test_python_execution():
    """Test Python execution without keyboard input."""
    print("\n🧪 Testing Python execution...")
    
    # Test basic Python functionality
    try:
        result = 2 + 2
        print(f"✅ Basic math: 2 + 2 = {result}")
    except Exception as e:
        print(f"❌ Basic math failed: {e}")
    
    # Test file operations
    try:
        test_file = "test_temp.txt"
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        with open(test_file, 'r') as f:
            content = f.read()
        
        os.remove(test_file)
        print("✅ File operations: Read/Write/Delete successful")
    except Exception as e:
        print(f"❌ File operations failed: {e}")

def check_system_resources():
    """Check system resources."""
    print("\n💻 Checking system resources...")
    
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"✅ CPU Usage: {cpu_percent}%")
        print(f"✅ Memory: {memory.percent}% used ({memory.available // (1024**3)} GB available)")
        print(f"✅ Disk: {disk.percent}% used ({disk.free // (1024**3)} GB free)")
    except ImportError:
        print("⚠️ psutil not installed - skipping resource check")
    except Exception as e:
        print(f"❌ Resource check failed: {e}")

def generate_report(installed, missing, found, missing_files):
    """Generate comprehensive report."""
    print("\n" + "="*60)
    print("📊 SYSTEM CHECK REPORT")
    print("="*60)
    
    # Python packages
    print(f"\n📦 Python Packages:")
    print(f"   Installed: {len(installed)}/{len(installed) + len(missing)}")
    print(f"   Missing: {len(missing)}")
    
    if missing:
        print(f"   Missing packages: {', '.join(missing)}")
    
    # Files
    print(f"\n📁 Files:")
    print(f"   Found: {len(found)}/{len(found) + len(missing_files)}")
    print(f"   Missing: {len(missing_files)}")
    
    if missing_files:
        print(f"   Missing files: {', '.join(missing_files)}")
    
    # Overall status
    if not missing and not missing_files:
        print(f"\n🎉 STATUS: EXCELLENT")
        print("   All packages and files are available!")
        print("   System is ready for MR BEN!")
    elif len(missing) <= 2 and len(missing_files) <= 2:
        print(f"\n✅ STATUS: GOOD")
        print("   Minor issues detected, but system should work.")
        print("   Consider installing missing packages.")
    else:
        print(f"\n⚠️ STATUS: NEEDS ATTENTION")
        print("   Several issues detected.")
        print("   Please address missing packages and files.")
    
    print("\n" + "="*60)

def main():
    """Main function."""
    print_header()
    
    # Run all checks
    installed, missing = check_python_packages()
    found, missing_files = check_files()
    check_directories()
    test_python_execution()
    check_system_resources()
    
    # Generate report
    generate_report(installed, missing, found, missing_files)
    
    # Provide next steps
    print("\n🎯 NEXT STEPS:")
    if missing:
        print("1. Install missing packages: pip install " + " ".join(missing))
    
    if not missing and not missing_files:
        print("1. ✅ System is ready!")
        print("2. 🚀 You can now run MR BEN trading system")
        print("3. 📋 Use: python start_system.py")
    else:
        print("1. 🔧 Fix missing packages and files")
        print("2. 🧪 Run tests again")
        print("3. 🚀 Then proceed with MR BEN system")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 