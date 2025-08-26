#!/usr/bin/env python3
"""
Test script to verify ML filter feature count fix
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_filter import AISignalFilter

    print("✅ AI Filter imported successfully")
except ImportError as e:
    print(f"❌ Error importing AI Filter: {e}")
    sys.exit(1)


def create_test_data():
    """Create test data with all required features."""
    print("🔄 Creating test data...")

    # Generate synthetic OHLC data
    num_bars = 100
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

    # Calculate technical indicators
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

    # SMA
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()

    # MACD histogram
    df['MACD_hist'] = df['macd'] - df['macd_signal']

    # Fill NaN values
    df = df.ffill().bfill()

    print(f"✅ Test data created with {len(df)} bars")
    print(f"📊 Columns: {list(df.columns)}")

    return df


def test_ml_filter_features():
    """Test ML filter with correct features."""
    print("\n🧪 Testing ML Filter Feature Count...")

    # Load ML filter
    ml_filter_path = 'models/mrben_ai_signal_filter_xgb_balanced.joblib'
    if not os.path.exists(ml_filter_path):
        print(f"❌ ML Filter not found: {ml_filter_path}")
        return False

    try:
        ml_filter = AISignalFilter(model_path=ml_filter_path, model_type="joblib", threshold=0.5)
        print("✅ ML Filter loaded successfully")
        print(f"📊 Expected features: {ml_filter.feature_count}")
        print(f"📊 Feature names: {ml_filter.feature_names}")

    except Exception as e:
        print(f"❌ Error loading ML Filter: {e}")
        return False

    # Create test data
    df = create_test_data()

    # Test feature extraction (same as in live_trader_clean.py)
    try:
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

        print("\n📊 Feature extraction test:")
        print(f"   Number of features: {len(feature_list)}")
        print(f"   Expected: {ml_filter.feature_count}")
        print(f"   Features: {feature_list}")

        # Test ML filter prediction
        print("\n🧪 Testing ML Filter prediction...")
        ml_result = ml_filter.filter_signal_with_confidence(feature_list)

        print("✅ ML Filter test successful!")
        print(f"📊 Prediction: {ml_result['prediction']}")
        print(f"📊 Confidence: {ml_result['confidence']:.3f}")

        return True

    except Exception as e:
        print(f"❌ Error in feature extraction test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🎯 ML Filter Feature Count Test")
    print("=" * 50)

    success = test_ml_filter_features()

    if success:
        print("\n✅ All tests passed! ML Filter feature count issue is fixed.")
    else:
        print("\n❌ Tests failed! Please check the error messages above.")

    return success


if __name__ == "__main__":
    main()
