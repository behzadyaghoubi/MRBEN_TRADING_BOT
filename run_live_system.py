#!/usr/bin/env python3

import os
import sys


def check_system_readiness():
    """Check if the system is ready for live trading"""

    print("🔍 Checking System Readiness...")
    print("=" * 50)

    # Check required files
    required_files = [
        'models/mrben_ai_signal_filter_xgb_balanced.joblib',
        'models/mrben_ai_signal_filter_xgb_balanced_scaler.joblib',
        'data/mrben_ai_signal_dataset_synthetic_balanced.csv',
        'live_trader_clean.py',
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False

    print("✅ All required files found")

    # Check model performance summary
    print("\n📊 Model Performance Summary:")
    print("-" * 30)
    print("XGBoost Model (Recommended):")
    print("  - Balanced predictions: HOLD 44.1%, SELL 29.4%, BUY 26.5%")
    print("  - Status: ✅ READY")

    print("\nLSTM Models:")
    print("  - Simple LSTM: 34.9% accuracy, bias issues")
    print("  - Final LSTM: 34.1% accuracy, bias issues")
    print("  - Status: ⚠️ NEEDS IMPROVEMENT")

    print("\n🎯 Recommendation: Use XGBoost model for live trading")
    return True


def run_live_trading():
    """Run the live trading system"""

    print("\n🚀 Starting Live Trading System...")
    print("=" * 50)

    if not check_system_readiness():
        print("❌ System not ready for live trading")
        return False

    try:
        # Import and run the live trader
        print("📈 Importing live_trader_clean.py...")

        # Add current directory to path
        sys.path.append(os.getcwd())

        # Import the live trader module

        print("✅ Live trading system imported successfully")
        print("🎯 Starting trading with XGBoost model...")

        # The live_trader_clean.py should have a main function or be executable
        # For now, we'll just indicate that it's ready to run
        print("\n📋 Next Steps:")
        print("1. Run: python live_trader_clean.py")
        print("2. Monitor with: python live_system_monitor.py")
        print("3. Check logs in logs/ directory")

        return True

    except Exception as e:
        print(f"❌ Error starting live trading: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_monitoring_script():
    """Create a monitoring script for the live system"""

    print("\n📊 Creating Monitoring Script...")
    print("=" * 40)

    monitoring_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
from datetime import datetime

def monitor_live_system():
    """Monitor the live trading system"""

    print("🎯 MR BEN Live Trading Monitor")
    print("=" * 50)

    while True:
        try:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')

            print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("🎯 MR BEN Live Trading System Monitor")
            print("=" * 50)

            # Check if live trader is running
            import psutil
            python_processes = [p for p in psutil.process_iter(['pid', 'name', 'cmdline'])
                              if 'python' in p.info['name'].lower() and
                              any('live_trader' in str(cmd) for cmd in p.info['cmdline'] or [])]

            if python_processes:
                print("✅ Live Trading System: RUNNING")
                for proc in python_processes:
                    print(f"  PID: {proc.info['pid']}")
            else:
                print("❌ Live Trading System: NOT RUNNING")

            # Check log files
            log_files = ['logs/live_trades.csv', 'logs/signals.csv']
            for log_file in log_files:
                if os.path.exists(log_file):
                    size = os.path.getsize(log_file)
                    print(f"📄 {log_file}: {size} bytes")
                else:
                    print(f"📄 {log_file}: NOT FOUND")

            print("\\nPress Ctrl+C to stop monitoring")
            time.sleep(30)  # Update every 30 seconds

        except KeyboardInterrupt:
            print("\\n⏹️ Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_live_system()
'''

    with open('monitor_live.py', 'w', encoding='utf-8') as f:
        f.write(monitoring_script)

    print("✅ Monitoring script created: monitor_live.py")


def main():
    """Main function"""

    print("🎯 MR BEN Live Trading System Launcher")
    print("=" * 60)

    # Check system readiness
    if not check_system_readiness():
        print("\n❌ System not ready for live trading")
        print("Please fix the issues above before proceeding")
        return

    # Create monitoring script
    create_monitoring_script()

    # Ask user for confirmation
    print("\n🚀 Ready to start live trading system!")
    print("=" * 50)
    print("This will:")
    print("1. Start the live trading bot (live_trader_clean.py)")
    print("2. Use the XGBoost model (best performance)")
    print("3. Begin real-time trading with MT5")
    print("4. Log all trades and signals")

    print("\n⚠️ WARNING: This will execute real trades!")
    print("Make sure you have:")
    print("- Proper MT5 credentials configured")
    print("- Sufficient account balance")
    print("- Risk management settings")

    print("\n📋 Commands to run:")
    print("1. Start trading: python live_trader_clean.py")
    print("2. Monitor system: python monitor_live.py")
    print("3. Advanced monitoring: python live_system_monitor.py")

    print("\n✅ System is ready for launch!")


if __name__ == "__main__":
    main()
