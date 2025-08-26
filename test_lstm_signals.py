#!/usr/bin/env python3
"""
Test LSTM Signals - تست سیگنال‌های LSTM
"""
import os

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def test_lstm_with_different_data():
    """Test LSTM with different market conditions."""

    print("🧪 Testing LSTM with Different Market Conditions")
    print("=" * 60)

    # Load LSTM model
    lstm_model_path = "models/advanced_lstm_model.h5"
    lstm_scaler_path = "models/advanced_lstm_scaler.save"

    if not os.path.exists(lstm_model_path):
        print(f"❌ LSTM Model not found: {lstm_model_path}")
        return

    lstm_model = load_model(lstm_model_path)
    lstm_scaler = joblib.load(lstm_scaler_path)

    # Test scenarios
    scenarios = [
        {'name': 'Strong Uptrend', 'trend': 'up', 'volatility': 'high', 'description': 'قوی صعودی'},
        {
            'name': 'Strong Downtrend',
            'trend': 'down',
            'volatility': 'high',
            'description': 'قوی نزولی',
        },
        {
            'name': 'Sideways Market',
            'trend': 'sideways',
            'volatility': 'low',
            'description': 'بازار خنثی',
        },
        {
            'name': 'Volatile Market',
            'trend': 'mixed',
            'volatility': 'high',
            'description': 'بازار پرنوسان',
        },
    ]

    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']} ({scenario['description']})")
        print("-" * 40)

        # Generate data based on scenario
        df = generate_scenario_data(scenario)

        # Get LSTM prediction
        try:
            features = df[['open', 'high', 'low', 'close', 'tick_volume']].values
            scaled_features = lstm_scaler.transform(features)
            lstm_input = scaled_features.reshape(1, -1, 5)

            prediction = lstm_model.predict(lstm_input, verbose=0)
            lstm_proba = prediction[0]

            signal_idx = np.argmax(lstm_proba)
            signal_map = {0: -1, 1: 0, 2: 1}  # SELL, HOLD, BUY
            lstm_signal = signal_map[signal_idx]
            lstm_confidence = float(np.max(lstm_proba))

            signal_names = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}

            print(f"   LSTM Prediction: {lstm_proba}")
            print(f"   Signal: {lstm_signal} ({signal_names[lstm_signal]})")
            print(f"   Confidence: {lstm_confidence:.3f}")
            print(f"   Price Range: {df['close'].min():.2f} - {df['close'].max():.2f}")

        except Exception as e:
            print(f"   ❌ Error: {e}")


def generate_scenario_data(scenario):
    """Generate synthetic data for different market scenarios."""
    np.random.seed(42)
    n_samples = 50
    base_price = 2000.0

    if scenario['trend'] == 'up':
        # Strong uptrend
        trend_factor = 0.02
        prices = [base_price]
        for i in range(1, n_samples):
            trend = np.random.normal(trend_factor, 0.01)
            new_price = prices[-1] * (1 + trend)
            prices.append(new_price)

    elif scenario['trend'] == 'down':
        # Strong downtrend
        trend_factor = -0.02
        prices = [base_price]
        for i in range(1, n_samples):
            trend = np.random.normal(trend_factor, 0.01)
            new_price = prices[-1] * (1 + trend)
            prices.append(new_price)

    elif scenario['trend'] == 'sideways':
        # Sideways market
        prices = [base_price]
        for i in range(1, n_samples):
            change = np.random.normal(0, 0.005)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

    else:  # mixed/volatile
        # Volatile market
        prices = [base_price]
        for i in range(1, n_samples):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

    # Add volatility
    if scenario['volatility'] == 'high':
        volatility_factor = 0.01
    else:
        volatility_factor = 0.003

    df = pd.DataFrame(
        {
            'open': [p * (1 + np.random.normal(0, volatility_factor)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, volatility_factor * 2))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, volatility_factor * 2))) for p in prices],
            'close': prices,
            'tick_volume': np.random.randint(100, 1000, n_samples),
        }
    )

    return df


if __name__ == "__main__":
    test_lstm_with_different_data()
