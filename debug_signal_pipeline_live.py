#!/usr/bin/env python3
"""
Debug Signal Pipeline Live
Test the complete signal pipeline to see what's happening with LSTM, TA, and ML filter
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def debug_signal_pipeline():
    """Debug the complete signal pipeline."""

    print("🔍 Debugging Signal Pipeline...")

    # Load enhanced data
    data_file = 'data/XAUUSD_PRO_M5_enhanced.csv'
    if not os.path.exists(data_file):
        print(f"❌ Enhanced data file not found: {data_file}")
        return False

    df = pd.read_csv(data_file)
    print(f"📊 Loaded {len(df)} rows from enhanced data")

    # Load LSTM model and scaler
    model_path = 'models/lstm_balanced_model.h5'
    scaler_path = 'models/lstm_balanced_scaler.save'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ LSTM model or scaler not found")
        return False

    try:
        lstm_model = load_model(model_path)
        lstm_scaler = joblib.load(scaler_path)
        print("✅ LSTM model and scaler loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    # Test LSTM Signal Generation
    print("\n🧪 Testing LSTM Signal Generation...")

    features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd', 'atr']
    data = df[features].values
    data = data[~np.isnan(data).any(axis=1)]

    if len(data) < 50:
        print(f"❌ Insufficient data: {len(data)} rows")
        return False

    # Scale data
    scaled_data = lstm_scaler.transform(data)

    # Test LSTM prediction
    timesteps = 50
    recent_data = scaled_data[-timesteps:]
    sequence = recent_data.reshape(1, timesteps, -1)

    prediction = lstm_model.predict(sequence, verbose=0)
    signal_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    signal_name = signal_map[signal_class]

    print("🎯 LSTM Results:")
    print(f"   Signal: {signal_class} ({signal_name})")
    print(f"   Confidence: {confidence:.4f}")
    print(
        f"   Probabilities: SELL={prediction[0][0]:.4f}, HOLD={prediction[0][1]:.4f}, BUY={prediction[0][2]:.4f}"
    )

    # Test Technical Analysis
    print("\n🧪 Testing Technical Analysis...")

    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd.values, macd_signal.values

    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.values, lower_band.values

    def calculate_atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr.values

    # Calculate technical indicators
    rsi = calculate_rsi(df['close'])
    macd, macd_signal = calculate_macd(df['close'])
    bb_upper, bb_lower = calculate_bollinger_bands(df['close'])
    atr = calculate_atr(df)

    # Get current values
    current_price = df['close'].iloc[-1]
    current_rsi = rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50
    current_macd = macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0
    current_macd_signal = (
        macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0
    )
    current_atr = atr[-1] if len(atr) > 0 and not np.isnan(atr[-1]) else 0

    print("📊 Technical Indicators:")
    print(f"   Price: {current_price:.2f}")
    print(f"   RSI: {current_rsi:.2f}")
    print(f"   MACD: {current_macd:.6f}")
    print(f"   MACD Signal: {current_macd_signal:.6f}")
    print(f"   ATR: {current_atr:.6f}")

    # Technical signal logic
    rsi_buy = current_rsi < 30
    rsi_sell = current_rsi > 70
    macd_buy = current_macd > current_macd_signal and current_macd > 0
    macd_sell = current_macd < current_macd_signal and current_macd < 0
    bb_buy = (
        current_price < bb_lower[-1] if len(bb_lower) > 0 and not np.isnan(bb_lower[-1]) else False
    )
    bb_sell = (
        current_price > bb_upper[-1] if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]) else False
    )

    buy_signals = sum([rsi_buy, macd_buy, bb_buy])
    sell_signals = sum([rsi_sell, macd_sell, bb_sell])

    if buy_signals >= 2:
        ta_signal = 1
        ta_confidence = 0.6 + (buy_signals * 0.1)
    elif sell_signals >= 2:
        ta_signal = -1
        ta_confidence = 0.6 + (sell_signals * 0.1)
    else:
        ta_signal = 0
        ta_confidence = 0.5

    print("🎯 Technical Analysis Results:")
    print(f"   Signal: {ta_signal}")
    print(f"   Confidence: {ta_confidence:.3f}")
    print(f"   Buy signals: {buy_signals}, Sell signals: {sell_signals}")

    # Test ML Filter (simplified)
    print("\n🧪 Testing ML Filter (simplified)...")

    # Simulate ML filter features
    ml_features = {
        'RSI': current_rsi,
        'MACD': current_macd,
        'ATR': current_atr,
        'Volume': df['tick_volume'].iloc[-1] if 'tick_volume' in df.columns else 0,
    }

    print("📊 ML Filter Features:")
    for key, value in ml_features.items():
        print(f"   {key}: {value:.6f}")

    # Simulate ML filter confidence (since we don't have the actual filter loaded)
    # This is a simplified simulation
    ml_confidence = 0.6  # Simulated value
    print("🎯 ML Filter Results:")
    print(f"   Confidence: {ml_confidence:.3f}")

    # Test Signal Combination Logic
    print("\n🧪 Testing Signal Combination Logic...")

    MIN_SIGNAL_CONFIDENCE = 0.5
    lstm_weight = 0.4
    ta_weight = 0.3
    ml_weight = 0.3

    # Calculate weighted confidence
    weighted_confidence = (
        confidence * lstm_weight + ta_confidence * ta_weight + ml_confidence * ml_weight
    )

    print("📊 Weighted Confidence Calculation:")
    print(f"   LSTM: {confidence:.4f} * {lstm_weight} = {confidence * lstm_weight:.4f}")
    print(f"   TA: {ta_confidence:.4f} * {ta_weight} = {ta_confidence * ta_weight:.4f}")
    print(f"   ML: {ml_confidence:.4f} * {ml_weight} = {ml_confidence * ml_weight:.4f}")
    print(f"   Total: {weighted_confidence:.4f}")

    # Determine final signal
    lstm_signal = (
        signal_map[signal_class] if signal_class == 2 else (-1 if signal_class == 0 else 0)
    )

    if lstm_signal == ta_signal and lstm_signal != 0:
        print("✅ Signals agree!")
        final_signal = lstm_signal
        if ml_confidence > MIN_SIGNAL_CONFIDENCE:
            final_confidence = weighted_confidence
            print(
                f"✅ ML filter approves (confidence {ml_confidence:.3f} > {MIN_SIGNAL_CONFIDENCE})"
            )
        else:
            final_signal = 0
            final_confidence = 0.0
            print(
                f"❌ ML filter rejects (confidence {ml_confidence:.3f} <= {MIN_SIGNAL_CONFIDENCE})"
            )
    else:
        print("⚠️ Signals disagree, using highest confidence")
        if confidence > ta_confidence:
            final_signal = lstm_signal
            final_confidence = confidence
            print(f"   Using LSTM signal (confidence {confidence:.4f})")
        else:
            final_signal = ta_signal
            final_confidence = ta_confidence
            print(f"   Using TA signal (confidence {ta_confidence:.4f})")

        if ml_confidence < MIN_SIGNAL_CONFIDENCE:
            final_signal = 0
            final_confidence = 0.0
            print(
                f"❌ ML filter rejects final signal (confidence {ml_confidence:.3f} < {MIN_SIGNAL_CONFIDENCE})"
            )

    print("\n🎯 Final Results:")
    print(f"   Signal: {final_signal}")
    print(f"   Confidence: {final_confidence:.4f}")

    return True


if __name__ == "__main__":
    success = debug_signal_pipeline()
    if success:
        print("\n✅ Signal pipeline debug completed!")
    else:
        print("\n❌ Signal pipeline debug failed!")
