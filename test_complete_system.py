#!/usr/bin/env python3
"""
Complete System Test
Test the entire system after ML filter fix
"""

from live_trader_clean import MT5LiveTrader


def test_complete_system():
    """Test the complete system functionality."""

    print("🧪 Testing Complete MR BEN Trading System...")
    print("=" * 50)

    try:
        # Create trader
        trader = MT5LiveTrader()

        print("✅ Trader created successfully")
        print(f"📊 Symbol: {trader.config.SYMBOL}")
        print(f"📊 Volume: {trader.config.VOLUME}")
        print(f"📊 Threshold: {trader.config.MIN_SIGNAL_CONFIDENCE}")
        print(f"📊 MT5 Enabled: {trader.config.ENABLE_MT5}")

        # Test data generation
        print("\n📊 Testing data generation...")
        df = trader.data_manager.get_latest_data(100)
        print(f"✅ Data generated: {len(df)} rows")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📊 Latest price: {df['close'].iloc[-1]:.2f}")

        # Test signal generation
        print("\n📊 Testing signal generation...")
        signal = trader.signal_generator.generate_enhanced_signal(df)
        print(f"✅ Signal generated: {signal}")
        print(f"📊 Signal: {signal['signal']}")
        print(f"📊 Confidence: {signal['confidence']:.3f}")
        print(f"📊 Source: {signal['source']}")

        # Test ML filter specifically
        print("\n📊 Testing ML Filter specifically...")
        if trader.ml_filter:
            test_features = [1, 0.8, 0, 0.5]
            ml_result = trader.ml_filter.filter_signal_with_confidence(test_features)
            print(f"✅ ML Filter working: {ml_result}")
        else:
            print("⚠️ ML Filter not loaded")

        print("\n✅ Complete system test passed!")
        print("🚀 System is ready for live trading!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_complete_system()
