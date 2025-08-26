#!/usr/bin/env python3
"""
Test LSTM Model with Enhanced Data
Verify that the LSTM model can generate proper signals with the enhanced XAUUSD.PRO data
"""

import os

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def test_lstm_with_enhanced_data():
    """Test LSTM model with enhanced XAUUSD.PRO data."""

    print("🧪 Testing LSTM Model with Enhanced Data...")

    # Load enhanced data
    data_file = 'data/XAUUSD_PRO_M5_enhanced.csv'
    if not os.path.exists(data_file):
        print(f"❌ Enhanced data file not found: {data_file}")
        return False

    df = pd.read_csv(data_file)
    print(f"📊 Loaded {len(df)} rows from enhanced data")
    print(f"📋 Columns: {list(df.columns)}")

    # Check if we have the required features
    required_features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd', 'atr']
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return False

    print("✅ All required features present")

    # Load LSTM model and scaler
    model_path = 'models/lstm_balanced_model.h5'
    scaler_path = 'models/lstm_balanced_scaler.save'

    if not os.path.exists(model_path):
        print(f"❌ LSTM model not found: {model_path}")
        return False

    if not os.path.exists(scaler_path):
        print(f"❌ LSTM scaler not found: {scaler_path}")
        return False

    try:
        lstm_model = load_model(model_path)
        lstm_scaler = joblib.load(scaler_path)
        print("✅ LSTM model and scaler loaded successfully")
        print(f"📐 Model input shape: {lstm_model.input_shape}")
        print(f"📐 Model output shape: {lstm_model.output_shape}")
        print(f"🔧 Scaler features: {lstm_scaler.n_features_in_}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    # Prepare data for prediction
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd', 'atr']
    data = df[features].values

    # Remove any NaN values
    data = data[~np.isnan(data).any(axis=1)]

    if len(data) < 60:  # Need at least 60 timesteps
        print(f"❌ Insufficient data: {len(data)} rows, need at least 60")
        return False

    print(f"📊 Using {len(data)} rows of clean data")

    # Scale the data
    try:
        scaled_data = lstm_scaler.transform(data)
        print("✅ Data scaled successfully")
    except Exception as e:
        print(f"❌ Error scaling data: {e}")
        return False

    # Test predictions on recent data
    timesteps = 60  # LSTM sequence length
    recent_data = scaled_data[-timesteps:]
    sequence = recent_data.reshape(1, timesteps, -1)

    try:
        prediction = lstm_model.predict(sequence, verbose=0)
        print("✅ LSTM prediction successful")

        # Analyze prediction
        signal_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        signal_name = signal_map[signal_class]

        print("🎯 Prediction Results:")
        print(f"   Signal Class: {signal_class} ({signal_name})")
        print(f"   Confidence: {confidence:.4f}")
        print(
            f"   Raw Probabilities: SELL={prediction[0][0]:.4f}, HOLD={prediction[0][1]:.4f}, BUY={prediction[0][2]:.4f}"
        )

        # Test multiple predictions
        print("\n🧪 Testing multiple predictions...")
        for i in range(5):
            start_idx = len(scaled_data) - timesteps - i * 10
            if start_idx < 0:
                break

            test_sequence = scaled_data[start_idx : start_idx + timesteps].reshape(1, timesteps, -1)
            test_pred = lstm_model.predict(test_sequence, verbose=0)
            test_signal = np.argmax(test_pred[0])
            test_conf = np.max(test_pred[0])

            print(f"   Sample {i+1}: {signal_map[test_signal]} (conf: {test_conf:.4f})")

        return True

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False


if __name__ == "__main__":
    success = test_lstm_with_enhanced_data()
    if success:
        print("\n✅ LSTM model test completed successfully!")
    else:
        print("\n❌ LSTM model test failed!")
