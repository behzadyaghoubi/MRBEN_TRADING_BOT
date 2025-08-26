#!/usr/bin/env python3
"""
MR BEN - Automatic Startup Script
=================================
Checks all requirements and runs the trading system correctly.
"""

import os
import subprocess
import sys
from datetime import datetime


def print_header():
    """Print script header."""
    print("=" * 60)
    print("🚀 MR BEN - AUTOMATIC STARTUP")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def check_python_path():
    """Check and set Python path."""
    print("🔍 Checking Python path...")

    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"✅ Added {current_dir} to Python path")

    return True


def check_required_files():
    """Check required files exist."""
    print("\n📁 Checking required files...")

    required_files = [
        "src/main_runner.py",
        "ai_filter.py",
        "risk_manager.py",
        "config/settings.json",
        "models/mrben_ai_signal_filter_xgb.joblib",
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        return False

    return True


def check_python_packages():
    """Check Python packages."""
    print("\n📦 Checking Python packages...")

    packages = ['pandas', 'numpy', 'tensorflow', 'sklearn', 'joblib']

    missing_packages = []
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        return False

    return True


def test_mt5_connection():
    """Test MT5 connection."""
    print("\n🔌 Testing MT5 connection...")

    try:
        result = subprocess.run(
            [sys.executable, 'test_mt5_connection.py'], capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            print("✅ MT5 connection successful")
            return True
        else:
            print("❌ MT5 connection failed")
            print(f"Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ MT5 test error: {e}")
        return False


def run_main_system():
    """Run the main trading system."""
    print("\n🚀 Starting MR BEN Trading System...")

    try:
        # Run the main system
        result = subprocess.run(
            [sys.executable, 'src/main_runner.py'], timeout=300
        )  # 5 minutes timeout

        if result.returncode == 0:
            print("✅ MR BEN system completed successfully")
            return True
        else:
            print("❌ MR BEN system failed")
            return False

    except subprocess.TimeoutExpired:
        print("⚠️ MR BEN system is running (timeout reached)")
        return True
    except Exception as e:
        print(f"❌ MR BEN system error: {e}")
        return False


def main():
    """Main function."""
    print_header()

    # Check all requirements
    checks = [
        ("Python Path", check_python_path),
        ("Required Files", check_required_files),
        ("Python Packages", check_python_packages),
        ("MT5 Connection", test_mt5_connection),
    ]

    all_passed = True
    for check_name, check_func in checks:
        print(f"\n--- {check_name} Check ---")
        if not check_func():
            all_passed = False
            print(f"❌ {check_name} check failed")
        else:
            print(f"✅ {check_name} check passed")

    if not all_passed:
        print("\n❌ Some checks failed. Please fix issues before running.")
        return False

    print("\n" + "=" * 60)
    print("🎉 ALL CHECKS PASSED!")
    print("🚀 Starting MR BEN Trading System...")
    print("=" * 60)

    # Run the main system
    return run_main_system()


if __name__ == "__main__":
    success = main()

    if success:
        print("\n✅ MR BEN system startup completed!")
    else:
        print("\n❌ MR BEN system startup failed!")
        sys.exit(1)
