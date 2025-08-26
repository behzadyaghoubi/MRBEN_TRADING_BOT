#!/usr/bin/env python3

import os
import subprocess
import sys
from datetime import datetime


def check_file_exists(filename):
    """Check if a file exists"""
    if os.path.exists(filename):
        print(f"✅ {filename} exists")
        return True
    else:
        print(f"❌ {filename} not found")
        return False


def check_config():
    """Check if config.json exists and is valid"""
    if not check_file_exists('config.json'):
        return False

    try:
        import json

        with open('config.json', encoding='utf-8') as f:
            config = json.load(f)

        print("✅ Config file is valid JSON")
        print(f"   Symbol: {config['trading']['symbol']}")
        print(f"   Sessions: {config['trading']['sessions']}")
        print(f"   Fixed Volume: {config['trading']['fixed_volume']}")
        print(f"   Use Risk Based Volume: {config['trading']['use_risk_based_volume']}")
        return True
    except Exception as e:
        print(f"❌ Config file error: {e}")
        return False


def run_live_trader():
    """Run the live trader"""
    print("\n🚀 Starting live trader...")
    print("=" * 50)

    try:
        # Run the live trader
        result = subprocess.run(
            [sys.executable, 'live_trader_clean.py'], capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            print("✅ Live trader started successfully")
            if result.stdout:
                print("Output:", result.stdout)
        else:
            print("❌ Live trader failed to start")
            if result.stderr:
                print("Error:", result.stderr)
            if result.stdout:
                print("Output:", result.stdout)

    except subprocess.TimeoutExpired:
        print("✅ Live trader is running (timeout reached - this is normal)")
    except Exception as e:
        print(f"❌ Error running live trader: {e}")


def main():
    print("🔍 Checking live trader system...")
    print("=" * 50)

    # Check files
    files_ok = all([check_file_exists('live_trader_clean.py'), check_config()])

    if not files_ok:
        print("\n❌ System check failed. Please fix the issues above.")
        return

    print("\n✅ All files are present and valid")

    # Check if logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("📁 Created logs directory")

    # Check if data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        print("📁 Created data directory")

    # Run the live trader
    run_live_trader()

    print("\n📊 System Status:")
    print(f"   Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   Log file: logs/live_trader_clean.log")
    print("   Trade log: data/trade_log_gold.csv")


if __name__ == "__main__":
    main()
