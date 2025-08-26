#!/usr/bin/env python3
"""
MR BEN - Trading System Selector
================================
Choose between different trading systems based on your needs.
"""

import os
import subprocess
import sys
from datetime import datetime


def print_header():
    """Print script header."""
    print("=" * 60)
    print("🎯 MR BEN - TRADING SYSTEM SELECTOR")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def show_menu():
    """Show system selection menu."""
    print("\n📋 Available Trading Systems:")
    print("1. 🔧 Simple System (src/main_runner.py)")
    print("   - Basic AI filtering")
    print("   - Simple risk management")
    print("   - Fast and stable")
    print()
    print("2. 🚀 Advanced System (live_trader_clean.py)")
    print("   - Real-time MT5 data management")
    print("   - LSTM + Technical signal generation")
    print("   - Advanced ML filtering")
    print("   - ATR-based risk management")
    print("   - Professional logging")
    print()
    print("3. 🔍 System Comparison")
    print("4. ❌ Exit")
    print()


def check_system_requirements(system_name):
    """Check if system requirements are met."""
    print(f"\n🔍 Checking requirements for {system_name}...")

    if system_name == "simple":
        required_files = [
            "src/main_runner.py",
            "ai_filter.py",
            "risk_manager.py",
            "config/settings.json",
        ]
    else:  # advanced
        required_files = [
            "live_trader_clean.py",
            "config/settings.json",
            "models/mrben_ai_signal_filter_xgb.joblib",
            "models/mrben_lstm_balanced_new.h5",
            "models/mrben_lstm_scaler_balanced.save",
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


def run_simple_system():
    """Run the simple trading system."""
    print("\n🚀 Starting Simple Trading System...")
    print("=" * 40)

    if not check_system_requirements("simple"):
        print("❌ System requirements not met!")
        return False

    try:
        result = subprocess.run([sys.executable, 'src/main_runner.py'], timeout=300)

        if result.returncode == 0:
            print("✅ Simple system completed successfully")
            return True
        else:
            print("❌ Simple system failed")
            return False

    except subprocess.TimeoutExpired:
        print("⚠️ Simple system is running (timeout reached)")
        return True
    except Exception as e:
        print(f"❌ Simple system error: {e}")
        return False


def run_advanced_system():
    """Run the advanced trading system."""
    print("\n🚀 Starting Advanced Trading System...")
    print("=" * 40)

    if not check_system_requirements("advanced"):
        print("❌ System requirements not met!")
        return False

    try:
        result = subprocess.run([sys.executable, 'live_trader_clean.py'], timeout=300)

        if result.returncode == 0:
            print("✅ Advanced system completed successfully")
            return True
        else:
            print("❌ Advanced system failed")
            return False

    except subprocess.TimeoutExpired:
        print("⚠️ Advanced system is running (timeout reached)")
        return True
    except Exception as e:
        print(f"❌ Advanced system error: {e}")
        return False


def show_comparison():
    """Show system comparison."""
    print("\n📊 SYSTEM COMPARISON")
    print("=" * 50)

    print("\n🔧 SIMPLE SYSTEM (src/main_runner.py):")
    print("✅ Pros:")
    print("   - Simple and easy to understand")
    print("   - Fast execution")
    print("   - Low resource usage")
    print("   - Stable and reliable")
    print("❌ Cons:")
    print("   - Limited features")
    print("   - Basic risk management")
    print("   - Simple logging")

    print("\n🚀 ADVANCED SYSTEM (live_trader_clean.py):")
    print("✅ Pros:")
    print("   - Real-time data management")
    print("   - Advanced signal generation (LSTM + TA)")
    print("   - Professional ML filtering")
    print("   - ATR-based risk management")
    print("   - Comprehensive logging")
    print("   - Position management")
    print("❌ Cons:")
    print("   - More complex code")
    print("   - Higher resource usage")
    print("   - More dependencies")

    print("\n🎯 RECOMMENDATION:")
    print("For beginners: Use Simple System")
    print("For professionals: Use Advanced System")


def main():
    """Main function."""
    print_header()

    while True:
        show_menu()

        try:
            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                run_simple_system()
                break
            elif choice == "2":
                run_advanced_system()
                break
            elif choice == "3":
                show_comparison()
                input("\nPress Enter to continue...")
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
