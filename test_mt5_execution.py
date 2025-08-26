#!/usr/bin/env python3
"""
Test MT5 Execution
Test that trades are actually sent to MT5
"""

from live_trader_clean import MT5LiveTrader


def test_mt5_execution():
    """Test MT5 execution functionality."""

    print("🧪 Testing MT5 Execution...")
    print("=" * 40)

    try:
        # Create trader
        trader = MT5LiveTrader()

        print("✅ Trader created successfully")
        print(f"📊 Symbol: {trader.config.SYMBOL}")
        print(f"📊 Volume: {trader.config.VOLUME}")
        print(f"📊 Demo Mode: {trader.config.DEMO_MODE}")
        print(f"📊 MT5 Enabled: {trader.config.ENABLE_MT5}")
        print(f"📊 Magic Number: {trader.config.MAGIC}")

        # Test data generation
        print("\n📊 Testing data generation...")
        df = trader.data_manager.get_latest_data(100)
        print(f"✅ Data generated: {len(df)} rows")
        print(f"📊 Latest price: {df['close'].iloc[-1]:.2f}")

        # Test signal generation
        print("\n📊 Testing signal generation...")
        signal = trader.signal_generator.generate_enhanced_signal(df)
        print(f"✅ Signal generated: {signal}")
        print(f"📊 Signal: {signal['signal']}")
        print(f"📊 Confidence: {signal['confidence']:.3f}")

        # Test trade execution (if signal is valid)
        if signal['signal'] != 0 and signal['confidence'] >= trader.config.MIN_SIGNAL_CONFIDENCE:
            print("\n🚀 Testing MT5 trade execution...")
            print(f"📊 Will execute: {signal['signal']} with confidence {signal['confidence']:.3f}")

            # Execute trade
            trader._execute_trade(signal, df)

            print("✅ Trade execution test completed!")
            print("📊 Check MT5 terminal for actual orders!")
        else:
            print("\n⚠️ No valid signal for execution test")
            print(f"📊 Signal: {signal['signal']}, Confidence: {signal['confidence']:.3f}")
            print(f"📊 Required: Signal != 0, Confidence >= {trader.config.MIN_SIGNAL_CONFIDENCE}")

        print("\n✅ MT5 execution test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_mt5_execution()
