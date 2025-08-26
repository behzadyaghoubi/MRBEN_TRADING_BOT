#!/usr/bin/env python3
"""
Test MT5 System
Quick test to verify MT5 integration is working
"""

import time

from live_trader_mt5 import MT5LiveTrader


def test_mt5_system():
    """Test the MT5 system for a few iterations."""

    print("🧪 Testing MT5 Live Trading System...")
    print("=" * 50)

    # Create trader
    trader = MT5LiveTrader()

    try:
        # Start the system
        trader.start()

        # Let it run for a few iterations
        print("⏳ Running system for 2 minutes...")
        time.sleep(120)  # 2 minutes

        # Stop the system
        trader.stop()

        print("✅ Test completed successfully!")

        # Check if any trades were executed
        import os

        trade_file = 'logs/mt5_trades.csv'
        if os.path.exists(trade_file):
            import pandas as pd

            df = pd.read_csv(trade_file, on_bad_lines='skip')
            print(f"📊 Total trades executed: {len(df)}")
            if len(df) > 0:
                print("📈 Sample trade:")
                print(df.iloc[-1].to_dict())
        else:
            print("📊 No trades executed yet")

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        trader.stop()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        trader.stop()


if __name__ == "__main__":
    test_mt5_system()
