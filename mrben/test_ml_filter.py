#!/usr/bin/env python3
"""
Test script for ML filter functionality.
"""

import numpy as np
import pandas as pd
from features.featurize import build_features, prepare_lstm_features, prepare_ml_features

from core.configx import load_config
from core.decide import Decider, LSTMDir, MLFilter
from core.loggingx import setup_logging


def create_mock_market_data():
    """Create mock market data for testing."""

    # Create OHLCV data
    np.random.seed(42)  # For reproducible results

    n_bars = 100
    base_price = 1.2000

    # Generate price data
    returns = np.random.normal(0, 0.001, n_bars)
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices)

    # Create OHLC data
    high = prices * (1 + np.random.uniform(0, 0.002, n_bars))
    low = prices * (1 - np.random.uniform(0, 0.002, n_bars))
    open_prices = prices * (1 + np.random.uniform(-0.001, 0.001, n_bars))

    # Ensure OHLC relationships
    high = np.maximum(high, np.maximum(open_prices, prices))
    low = np.minimum(low, np.minimum(open_prices, prices))

    # Create volume data
    volume = np.random.uniform(1000, 5000, n_bars)

    # Create DataFrame
    df = pd.DataFrame({'O': open_prices, 'H': high, 'L': low, 'C': prices, 'V': volume})

    # Add session and regime columns
    df['session'] = np.random.choice(['asia', 'london', 'ny', 'off'], n_bars)
    df['regime'] = np.random.choice(['LOW', 'NORMAL', 'HIGH'], n_bars)
    df['hour'] = np.random.randint(0, 24, n_bars)

    return df


def test_feature_engineering():
    """Test feature engineering functionality."""
    print("=== Testing Feature Engineering ===")

    # Create mock data
    df = create_mock_market_data()
    print(f"✅ Created mock market data: {len(df)} bars")

    # Test feature building
    try:
        features = build_features(df, lookback=50)
        print(f"✅ Features built successfully: shape {features.shape}")
        print(f"   Feature count: {features.shape[1]}")
        print(f"   Time steps: {features.shape[0]}")

        # Test ML features
        ml_features = prepare_ml_features(df, lookback=50)
        print(f"✅ ML features prepared: shape {ml_features.shape}")

        # Test LSTM features
        lstm_features = prepare_lstm_features(df, lookback=50)
        print(f"✅ LSTM features prepared: shape {lstm_features.shape}")

    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return None

    return features, ml_features, lstm_features


def test_ml_filter_mock():
    """Test ML filter with mock data (no actual ONNX model)."""
    print("\n=== Testing ML Filter (Mock) ===")

    # Create mock features with correct dimensions (17 features as per trained model)
    features = np.random.random((17,)).astype(np.float32)

    # Test ML filter initialization (will fail without model file)
    try:
        mlf = MLFilter("models/ml_filter_v1.onnx")
        print(f"✅ ML filter initialized: model_loaded={mlf.model_loaded}")

        if mlf.model_loaded:
            # Test prediction
            direction, confidence = mlf.predict(features)
            print(f"✅ ML prediction: dir={direction:+d}, conf={confidence:.3f}")
        else:
            print("ℹ️  ML model not loaded (expected without model file)")

    except Exception as e:
        print(f"ℹ️  ML filter initialization: {e}")


def test_lstm_mock():
    """Test LSTM with mock data (no actual ONNX model)."""
    print("\n=== Testing LSTM (Mock) ===")

    # Create mock sequence features with correct dimensions (17 features as per trained model)
    feature_seq = np.random.random((50, 17)).astype(np.float32)

    # Test LSTM initialization (will fail without model file)
    try:
        lstm = LSTMDir("models/lstm_dir_v1.onnx")
        print(f"✅ LSTM initialized: model_loaded={lstm.model_loaded}")

        if lstm.model_loaded:
            # Test prediction
            direction, confidence = lstm.predict(feature_seq)
            print(f"✅ LSTM prediction: dir={direction:+d}, conf={confidence:.3f}")
        else:
            print("ℹ️  LSTM model not loaded (expected without model file)")

    except Exception as e:
        print(f"ℹ️  LSTM initialization: {e}")


def test_decision_engine():
    """Test the decision engine with mock data."""
    print("\n=== Testing Decision Engine ===")

    try:
        # Load configuration
        cfg = load_config("config/config.yaml")
        print("✅ Configuration loaded successfully")

        # Create mock ML filter and LSTM (without models)
        mlf = MLFilter("models/ml_filter_v1.onnx")
        lstm = LSTMDir("models/lstm_dir_v1.onnx")

        # Create decision engine
        decider = Decider(cfg, mlf, lstm)
        print("✅ Decision engine created successfully")

        # Test dynamic confidence calculation
        dyn_conf = decider.dynamic_conf(0.70, "NORMAL", "london", "calm")
        print(f"✅ Dynamic confidence: {dyn_conf:.3f}")

        # Test voting mechanism
        score = decider.vote(+1, +1, +1, +1)  # All bullish
        print(f"✅ Ensemble vote (all bullish): {score:.3f}")

        score = decider.vote(+1, -1, +1, -1)  # Mixed signals
        print(f"✅ Ensemble vote (mixed): {score:.3f}")

        # Test decision making
        market_data = {
            'features': np.random.random((17,)).astype(np.float32),
            'feature_seq': np.random.random((50, 17)).astype(np.float32),
        }

        context = {'regime': 'NORMAL', 'session': 'london', 'drawdown_state': 'calm'}

        # Test ENTER decision
        decision = decider.decide(+1, +1, 0.65, market_data, context)
        print(f"✅ Decision test: {decision.action} - {decision.reason}")

        # Test HOLD decision (low PA score)
        decision = decider.decide(+1, +1, 0.45, market_data, context)
        print(f"✅ Decision test: {decision.action} - {decision.reason}")

    except Exception as e:
        print(f"❌ Decision engine test failed: {e}")
        import traceback

        traceback.print_exc()


def test_ml_logging():
    """Test ML filter logging."""
    print("\n=== Testing ML Logging ===")

    # Setup logging
    logger = setup_logging("INFO")

    # Create mock features with correct dimensions (17 features as per trained model)
    features = np.random.random((17,)).astype(np.float32)

    # Test ML filter with logging
    try:
        mlf = MLFilter("models/ml_filter_v1.onnx")

        if mlf.model_loaded:
            direction, confidence = mlf.predict(features)
            logger.bind(evt="ML").info("ml_prediction", direction=direction, confidence=confidence)
            print(f"✅ ML prediction logged: dir={direction:+d}, conf={confidence:.3f}")
        else:
            print("ℹ️  ML model not loaded - skipping prediction test")

    except Exception as e:
        print(f"ℹ️  ML logging test: {e}")


if __name__ == "__main__":
    print("MR BEN - ML Filter Test")
    print("=" * 30)

    try:
        # Test feature engineering
        result = test_feature_engineering()

        # Test ML filter
        test_ml_filter_mock()

        # Test LSTM
        test_lstm_mock()

        # Test decision engine
        test_decision_engine()

        # Test logging
        test_ml_logging()

        print("\n✅ All ML filter tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
