#!/usr/bin/env python3
"""
Dependency Installation Script for MR BEN Live Trading System
Automatically installs required packages and checks compatibility
"""

import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors gracefully"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False

    print("✅ Python version is compatible")
    return True


def install_package(package, description=None):
    """Install a single package"""
    if description is None:
        description = package

    return run_command(f"{sys.executable} -m pip install {package}", f"Installing {description}")


def main():
    """Main installation function"""
    print("🚀 MR BEN Live Trading System - Dependency Installer")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        return 1

    # Upgrade pip first
    print("\n📦 Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")

    # Core dependencies
    print("\n📚 Installing core dependencies...")
    core_packages = [
        ("pandas>=1.5.0", "pandas"),
        ("numpy>=1.21.0", "numpy"),
        ("psutil>=5.9.0", "psutil"),
        ("python-dateutil>=2.8.0", "python-dateutil"),
    ]

    for package, description in core_packages:
        if not install_package(package, description):
            print(f"⚠️ Warning: {description} installation failed, continuing...")

    # AI/ML dependencies
    print("\n🤖 Installing AI/ML dependencies...")
    ai_packages = [
        ("tensorflow>=2.10.0", "TensorFlow"),
        ("scikit-learn>=1.1.0", "scikit-learn"),
        ("joblib>=1.2.0", "joblib"),
    ]

    for package, description in ai_packages:
        if not install_package(package, description):
            print(f"⚠️ Warning: {description} installation failed, continuing...")

    # MetaTrader 5 (optional)
    print("\n📊 Installing MetaTrader 5 integration...")
    if not install_package("MetaTrader5>=5.0.0", "MetaTrader5"):
        print("⚠️ Warning: MetaTrader5 installation failed - live trading will not be available")
        print("   You can still use the system in demo mode")

    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        from sklearn.preprocessing import LabelEncoder

        print("✅ All core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Some dependencies failed to import: {e}")
        print("   You may need to restart your Python environment")
        return 1

    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Restart your Python environment")
    print("   2. Run: python live_trader_clean.py --help")
    print("   3. Configure your trading parameters in config.json")
    print("   4. Start trading with: python live_trader_clean.py live")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed with error: {e}")
        sys.exit(1)
