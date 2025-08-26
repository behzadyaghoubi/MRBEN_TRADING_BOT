#!/usr/bin/env python3
"""
Setup script for MR BEN Trading Bot environment configuration.
"""

import os
import shutil
from pathlib import Path

def setup_environment():
    """Setup environment configuration file."""
    print("MR BEN Trading Bot - Environment Setup")
    print("=" * 50)
    
    # Check if .env already exists
    env_file = Path(".env")
    example_file = Path("env.example")
    
    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Copy example file to .env
    if example_file.exists():
        shutil.copy(example_file, env_file)
        print("✅ Created .env file from env.example")
        print("📝 Please edit .env file with your actual configuration values:")
        print("   - MT5_LOGIN: Your MT5 account number")
        print("   - MT5_PASSWORD: Your MT5 password")
        print("   - MT5_SERVER: Your MT5 broker server")
        print("   - MT5_ENABLE_REAL_TRADING: Set to 'true' for live trading")
        print()
        print("🔧 Other important settings to review:")
        print("   - TRADING_SYMBOL: Trading instrument (default: XAUUSD)")
        print("   - TRADING_TIMEFRAME: Chart timeframe (default: M15)")
        print("   - TRADING_BASE_RISK: Risk per trade (default: 0.01 = 1%)")
        print("   - AI_MODEL_PATH: Path to your trained AI model")
        print()
        print("📖 See env.example for detailed descriptions of all settings.")
    else:
        print("❌ env.example file not found!")
        print("Please ensure env.example exists in the project root.")

if __name__ == "__main__":
    setup_environment() 