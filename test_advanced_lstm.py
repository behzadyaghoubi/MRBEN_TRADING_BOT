"""
Test script for Advanced LSTM Model
Verifies that the model works correctly with our data
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our advanced models
from advanced_models import build_advanced_lstm, build_enhanced_lstm_classifier, test_advanced_lstm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_with_real_data():
    """
    Test the advanced LSTM model with real trading data
    """
    try:
        print("🧪 Testing Advanced LSTM with Real Data...")

        # Try to load existing data
        data_files = ["XAUUSD_PRO_M15_history.csv", "adausd_data.csv", "ohlc_data.csv"]

        data = None
        for file in data_files:
            if os.path.exists(file):
                print(f"📊 Loading data from: {file}")
                data = pd.read_csv(file)
                break

        if data is None:
            print("⚠️ No real data found, generating synthetic data...")
            # Generate synthetic data
            np.random.seed(42)
            dates = pd.date_range(start='2024-01-01', periods=1000, freq='5T')

            data = pd.DataFrame(
                {
                    'time': dates,
                    'open': np.random.uniform(2000, 2100, 1000),
                    'high': np.random.uniform(2005, 2105, 1000),
                    'low': np.random.uniform(1995, 2095, 1000),
                    'close': np.random.uniform(2000, 2100, 1000),
                    'tick_volume': np.random.randint(100, 1000, 1000),
                }
            )

        print(f"📈 Data shape: {data.shape}")
        print(f"📋 Columns: {list(data.columns)}")

        # Prepare sequences
        timesteps = 50
        features = ['open', 'high', 'low', 'close', 'tick_volume']

        # Select features
        feature_data = data[features].values

        # Create sequences
        X, y = [], []
        for i in range(timesteps, len(feature_data)):
            X.append(feature_data[i - timesteps : i])
            y.append(feature_data[i, 3])  # Use close price as target

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Sequences prepared: X shape: {X.shape}, y shape: {y.shape}")

        # Test both models
        input_shape = (timesteps, len(features))

        # Test Advanced LSTM
        print("\n🔬 Testing Advanced LSTM with Attention...")
        model1 = build_advanced_lstm(input_shape, num_classes=3)
        if model1 is not None:
            print("✅ Advanced LSTM built successfully!")
            model1.summary()

            # Test prediction
            test_X = X[:10]  # Use first 10 sequences
            predictions = model1.predict(test_X)
            print(f"✅ Predictions shape: {predictions.shape}")
            print(f"✅ Sample predictions: {predictions[0]}")

        # Test Enhanced LSTM Classifier
        print("\n🔬 Testing Enhanced LSTM Classifier...")
        model2 = build_enhanced_lstm_classifier(input_shape, num_classes=3)
        if model2 is not None:
            print("✅ Enhanced LSTM Classifier built successfully!")
            model2.summary()

            # Test prediction
            predictions2 = model2.predict(test_X)
            print(f"✅ Predictions shape: {predictions2.shape}")
            print(f"✅ Sample predictions: {predictions2[0]}")

        print("\n🎉 All tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Error in test_with_real_data: {e}")
        return False


def main():
    """
    Main test function
    """
    print("🚀 Starting Advanced LSTM Model Tests...")
    print("=" * 50)

    # Test 1: Basic functionality
    print("\n📋 Test 1: Basic Model Functionality")
    success1 = test_advanced_lstm()

    # Test 2: Real data integration
    print("\n📋 Test 2: Real Data Integration")
    success2 = test_with_real_data()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"✅ Basic Functionality: {'PASS' if success1 else 'FAIL'}")
    print(f"✅ Real Data Integration: {'PASS' if success2 else 'FAIL'}")

    if success1 and success2:
        print("\n🎉 All tests passed! Advanced LSTM models are ready to use.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

    return success1 and success2


if __name__ == "__main__":
    main()
