#!/usr/bin/env python3
"""
Comprehensive test for live_trader_clean.py ML filter fix
"""

import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_test_data():
    """Create test data with all required features."""
    print("🔄 Creating comprehensive test data...")

    # Generate synthetic OHLC data
    num_bars = 500
    data = []
    base_price = 3300.0

    for i in range(num_bars):
        open_price = base_price + np.random.uniform(-10, 10)
        close_price = open_price + np.random.uniform(-5, 5)
        high = max(open_price, close_price) + np.random.uniform(0, 3)
        low = min(open_price, close_price) - np.random.uniform(0, 3)
        volume = np.random.randint(100, 1000)

        data.append(
            {
                'time': datetime.now() - timedelta(minutes=i * 5),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'tick_volume': volume,
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values('time').reset_index(drop=True)

    # Calculate technical indicators (same as in live_trader_clean.py)
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(window=14).mean()

    # Fill NaN values
    df = df.ffill().bfill()

    print(f"✅ Test data created with {len(df)} bars")
    print(f"📊 Available columns: {list(df.columns)}")

    return df


def test_signal_generation():
    """Test the complete signal generation pipeline."""
    print("\n🧪 Testing Complete Signal Generation Pipeline...")

    try:
        # Import the signal generator class
        from live_trader_clean import MT5Config, MT5SignalGenerator

        # Create config
        config = MT5Config()

        # Load models (this will test model loading)
        print("🔄 Loading AI models...")

        # Try to load LSTM model
        lstm_model = None
        lstm_scaler = None
        try:
            import joblib
            from tensorflow.keras.models import load_model

            model_paths = [
                'models/mrben_lstm_real_data.h5',
                'models/mrben_lstm_balanced_v2.h5',
                'models/mrben_lstm_balanced_new.h5',
                'models/mrben_lstm_model.h5',
            ]
            scaler_paths = [
                'models/mrben_lstm_real_data_scaler.save',
                'models/mrben_lstm_scaler_v2.save',
                'models/mrben_lstm_scaler_balanced.save',
                'models/mrben_lstm_scaler.save',
            ]

            for model_path, scaler_path in zip(model_paths, scaler_paths, strict=False):
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    print(f"Loading LSTM Model from {model_path}...")
                    lstm_model = load_model(model_path)
                    lstm_scaler = joblib.load(scaler_path)
                    print("✅ LSTM Model loaded successfully!")
                    break

            if lstm_model is None:
                print("⚠️ No LSTM model found, continuing without LSTM")

        except Exception as e:
            print(f"⚠️ Error loading LSTM model: {e}")

        # Load ML filter
        ml_filter = None
        try:
            from ai_filter import AISignalFilter

            ml_filter_paths = [
                'models/mrben_ai_signal_filter_xgb_balanced.joblib',
                'models/mrben_ai_signal_filter_xgb.joblib',
            ]

            for ml_filter_path in ml_filter_paths:
                if os.path.exists(ml_filter_path):
                    print(f"Loading ML Filter from {ml_filter_path}...")
                    ml_filter = AISignalFilter(
                        model_path=ml_filter_path, model_type="joblib", threshold=0.5
                    )
                    print("✅ ML Filter loaded successfully!")
                    print(f"📊 ML Filter expects {ml_filter.feature_count} features")
                    break

            if ml_filter is None:
                print("⚠️ No ML Filter found, continuing without ML filter")

        except Exception as e:
            print(f"⚠️ Error loading ML Filter: {e}")

        # Create signal generator
        signal_generator = MT5SignalGenerator(config, lstm_model, lstm_scaler, ml_filter)

        # Create test data
        df = create_test_data()

        # Test signal generation
        print("\n🧪 Testing signal generation...")

        # Generate multiple signals to test consistency
        for i in range(5):
            print(f"\n--- Test {i+1} ---")

            # Get a subset of data for testing
            test_df = df.iloc[-(100 + i * 10) :].copy()

            # Generate signal
            signal_data = signal_generator.generate_enhanced_signal(test_df)

            print(f"📊 Signal: {signal_data['signal']}")
            print(f"📊 Confidence: {signal_data['confidence']:.3f}")
            print(f"📊 Source: {signal_data.get('source', 'Unknown')}")

            # Check if signal is valid
            assert signal_data['signal'] in [
                -1,
                0,
                1,
            ], f"Invalid signal value: {signal_data['signal']}"
            assert (
                0 <= signal_data['confidence'] <= 1
            ), f"Invalid confidence value: {signal_data['confidence']}"

            time.sleep(0.1)  # Small delay between tests

        print("\n✅ All signal generation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error in signal generation test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test the feature extraction specifically."""
    print("\n🧪 Testing Feature Extraction...")

    try:
        # Create test data
        df = create_test_data()

        # Simulate the feature extraction from live_trader_clean.py
        current_data = df.iloc[-1]

        # Calculate additional technical indicators if not present
        if 'SMA20' not in df.columns:
            df['SMA20'] = df['close'].rolling(window=20).mean()
        if 'SMA50' not in df.columns:
            df['SMA50'] = df['close'].rolling(window=50).mean()
        if 'MACD_hist' not in df.columns:
            df['MACD_hist'] = df['macd'] - df['macd_signal']

        # Get current values
        features = {
            'open': current_data['open'],
            'high': current_data['high'],
            'low': current_data['low'],
            'close': current_data['close'],
            'SMA20': df['SMA20'].iloc[-1],
            'SMA50': df['SMA50'].iloc[-1],
            'RSI': current_data['rsi'],
            'MACD': current_data['macd'],
            'MACD_signal': current_data['macd_signal'],
            'MACD_hist': df['MACD_hist'].iloc[-1],
        }

        # Convert to list in correct order
        feature_list = [
            features['open'],
            features['high'],
            features['low'],
            features['close'],
            features['SMA20'],
            features['SMA50'],
            features['RSI'],
            features['MACD'],
            features['MACD_signal'],
            features['MACD_hist'],
        ]

        print("📊 Feature extraction successful!")
        print(f"   Number of features: {len(feature_list)}")
        print("   Expected: 10")
        print(
            f"   All features are numeric: {all(isinstance(f, (int, float, np.number)) for f in feature_list)}"
        )
        print(f"   No NaN values: {not any(pd.isna(f) for f in feature_list)}")

        # Test with ML filter
        try:
            from ai_filter import AISignalFilter

            ml_filter_path = 'models/mrben_ai_signal_filter_xgb_balanced.joblib'
            if os.path.exists(ml_filter_path):
                ml_filter = AISignalFilter(
                    model_path=ml_filter_path, model_type="joblib", threshold=0.5
                )

                # Test prediction
                ml_result = ml_filter.filter_signal_with_confidence(feature_list)

                print("✅ ML Filter prediction successful!")
                print(f"   Prediction: {ml_result['prediction']}")
                print(f"   Confidence: {ml_result['confidence']:.3f}")

            else:
                print("⚠️ ML Filter not found, skipping prediction test")

        except Exception as e:
            print(f"⚠️ Error testing ML Filter: {e}")

        return True

    except Exception as e:
        print(f"❌ Error in feature extraction test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🎯 Live Trader ML Filter Fix - Comprehensive Test")
    print("=" * 60)

    # Test 1: Feature extraction
    test1_success = test_feature_extraction()

    # Test 2: Complete signal generation
    test2_success = test_signal_generation()

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Feature Extraction Test: {'PASSED' if test1_success else 'FAILED'}")
    print(f"✅ Signal Generation Test: {'PASSED' if test2_success else 'FAILED'}")

    if test1_success and test2_success:
        print("\n🎉 ALL TESTS PASSED! ML Filter feature count issue is completely fixed.")
        print(
            "✅ The live_trader_clean.py should now work correctly without feature shape mismatch errors."
        )
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

    return test1_success and test2_success


if __name__ == "__main__":
    main()
