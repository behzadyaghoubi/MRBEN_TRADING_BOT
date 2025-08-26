#!/usr/bin/env python3
"""
MRBEN LSTM Trading System - Setup Script
========================================

Quick setup script to install dependencies and prepare the system.

Usage:
    python setup.py

This will:
1. Install required dependencies
2. Create necessary directories
3. Check system compatibility
4. Provide setup instructions

Author: MRBEN Trading System
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "outputs",
        "logs",
        "models",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def check_data_file():
    """Check if required data file exists"""
    print("\n📊 Checking data files...")
    
    if os.path.exists("lstm_signals_pro.csv"):
        print("✅ Found lstm_signals_pro.csv")
        return True
    else:
        print("⚠️  Warning: lstm_signals_pro.csv not found!")
        print("   Please ensure the LSTM signals file is in the current directory.")
        return False

def test_imports():
    """Test if all imports work correctly"""
    print("\n🧪 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import tensorflow as tf
        import sklearn
        import talib
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 MRBEN LSTM Trading System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Check data file
    data_ok = check_data_file()
    
    # Test imports
    if not test_imports():
        return False
    
    # Setup complete
    print("\n✅ Setup completed successfully!")
    
    if data_ok:
        print("\n🎯 Ready to run the trading system!")
        print("   Run: python run_trading_system.py")
        print("   Optimize: python optimize_parameters.py")
        print("   Ultra Balancer: python lstm_signal_balancer_ultra.py")
    else:
        print("\n⚠️  Setup complete, but data file missing!")
        print("   Please add lstm_signals_pro.csv to continue.")
    
    print("\n📚 For more information, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 