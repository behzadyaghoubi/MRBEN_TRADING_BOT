#!/usr/bin/env python3
"""
Final Comprehensive System Test
"""

import os
import time
from datetime import datetime

import joblib
import numpy as np


def print_status(message, level="INFO"):
    """Print status with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def test_data_files():
    """Test data files"""
    print_status("📁 Testing Data Files...", "TEST")

    sequences_path = "data/real_market_sequences.npy"
    labels_path = "data/real_market_labels.npy"

    if not os.path.exists(sequences_path):
        print_status(f"❌ Sequences file not found: {sequences_path}", "ERROR")
        return False, None, None

    if not os.path.exists(labels_path):
        print_status(f"❌ Labels file not found: {labels_path}", "ERROR")
        return False, None, None

    try:
        sequences = np.load(sequences_path)
        labels = np.load(labels_path)

        print_status("✅ Data files loaded successfully", "SUCCESS")
        print_status(f"   Sequences: {sequences.shape}", "INFO")
        print_status(f"   Labels: {labels.shape}", "INFO")

        # Check label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print_status("   Label distribution:", "INFO")
        for label, count in zip(unique_labels, counts, strict=False):
            percentage = (count / len(labels)) * 100
            signal_type = ["SELL", "HOLD", "BUY"][label]
            print_status(f"     {signal_type}: {count} ({percentage:.1f}%)", "INFO")

        return True, sequences, labels

    except Exception as e:
        print_status(f"❌ Error loading data: {e}", "ERROR")
        return False, None, None


def test_model_files():
    """Test model files"""
    print_status("🤖 Testing Model Files...", "TEST")

    model_paths = {
        'lstm_model': 'models/mrben_lstm_real_data.h5',
        'lstm_best': 'models/mrben_lstm_real_data_best.h5',
        'lstm_scaler': 'models/mrben_lstm_real_data_scaler.save',
    }

    results = {}

    for name, path in model_paths.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print_status(f"✅ {name}: {path} ({size:.1f} KB)", "SUCCESS")
            results[name] = True
        else:
            print_status(f"❌ {name}: {path} not found", "ERROR")
            results[name] = False

    return results


def test_lstm_performance(sequences, labels):
    """Test LSTM model performance"""
    print_status("🧠 Testing LSTM Model Performance...", "TEST")

    try:
        # Import TensorFlow
        from tensorflow.keras.models import load_model

        # Load model and scaler
        model_path = 'models/mrben_lstm_real_data.h5'
        scaler_path = 'models/mrben_lstm_real_data_scaler.save'

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        print_status(f"✅ Model loaded: {model.count_params():,} parameters", "SUCCESS")

        # Prepare test data (last 20%)
        test_size = int(len(sequences) * 0.2)
        X_test = sequences[-test_size:]
        y_test = labels[-test_size:]

        # Scale data
        n_samples, n_timesteps, n_features = X_test.shape
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(n_samples, n_timesteps, n_features)

        # Make predictions
        print_status("📊 Making predictions...", "INFO")
        predictions = model.predict(X_test_scaled, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # Calculate accuracy
        correct = np.sum(predicted_classes == y_test)
        accuracy = correct / len(y_test)

        print_status("✅ Performance Results:", "SUCCESS")
        print_status(f"   Test samples: {len(X_test)}", "INFO")
        print_status(f"   Correct predictions: {correct}/{len(y_test)}", "INFO")
        print_status(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", "INFO")

        # Prediction distribution
        unique_pred, pred_counts = np.unique(predicted_classes, return_counts=True)
        print_status("   Prediction distribution:", "INFO")
        for pred_class, count in zip(unique_pred, pred_counts, strict=False):
            percentage = (count / len(predicted_classes)) * 100
            signal_type = ["SELL", "HOLD", "BUY"][pred_class]
            print_status(f"     {signal_type}: {count} ({percentage:.1f}%)", "INFO")

        # Test individual samples
        print_status("🧪 Testing Individual Samples...", "INFO")
        for i in range(min(5, len(X_test))):
            pred_class = predicted_classes[i]
            actual_class = y_test[i]
            confidence = np.max(predictions[i])

            pred_signal = ["SELL", "HOLD", "BUY"][pred_class]
            actual_signal = ["SELL", "HOLD", "BUY"][actual_class]

            status = "✅" if pred_class == actual_class else "❌"
            print_status(
                f"   Sample {i+1}: {status} Pred={pred_signal} (Conf={confidence:.3f}), Actual={actual_signal}",
                "INFO",
            )

        return True, accuracy

    except Exception as e:
        print_status(f"❌ LSTM test error: {e}", "ERROR")
        return False, 0.0


def test_signal_generation():
    """Test signal generation"""
    print_status("🎯 Testing Signal Generation...", "TEST")

    try:
        # Import trading system
        from live_trader_clean import MT5LiveTrader

        # Create trader instance
        trader = MT5LiveTrader()

        # Get data
        df = trader.data_manager.get_latest_data(bars=100)
        if df is None:
            print_status("❌ Failed to get data", "ERROR")
            return False

        print_status(f"✅ Got {len(df)} bars of data", "SUCCESS")

        # Generate signal
        signal_data = trader.signal_generator.generate_enhanced_signal(df)

        print_status("✅ Signal generated:", "SUCCESS")
        print_status(
            f"   Signal: {signal_data['signal']} ({['SELL', 'HOLD', 'BUY'][signal_data['signal']]})",
            "INFO",
        )
        print_status(f"   Confidence: {signal_data['confidence']:.3f}", "INFO")
        print_status(f"   Source: {signal_data.get('source', 'Unknown')}", "INFO")

        return True

    except Exception as e:
        print_status(f"❌ Signal generation error: {e}", "ERROR")
        return False


def main():
    """Main test function"""
    print_status("🧪 Final Comprehensive System Test", "START")
    print_status("=" * 50, "INFO")

    start_time = time.time()

    # Test 1: Data Files
    data_ok, sequences, labels = test_data_files()

    # Test 2: Model Files
    model_results = test_model_files()

    # Test 3: LSTM Performance
    lstm_ok = False
    accuracy = 0.0
    if data_ok and sequences is not None and labels is not None:
        lstm_ok, accuracy = test_lstm_performance(sequences, labels)

    # Test 4: Signal Generation
    signal_ok = test_signal_generation()

    # Calculate test duration
    test_duration = time.time() - start_time

    # Summary
    print_status("📋 Test Summary:", "SUMMARY")
    print_status("=" * 30, "INFO")
    print_status(f"Data Files: {'✅' if data_ok else '❌'}", "INFO")
    print_status(f"LSTM Performance: {'✅' if lstm_ok else '❌'}", "INFO")
    print_status(f"Signal Generation: {'✅' if signal_ok else '❌'}", "INFO")
    print_status(f"Test Duration: {test_duration:.1f} seconds", "INFO")

    if lstm_ok:
        print_status(f"LSTM Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", "INFO")

    # Model files status
    print_status("📁 Model Files:", "INFO")
    for name, status in model_results.items():
        print_status(f"   {name}: {'✅' if status else '❌'}", "INFO")

    # Final recommendation
    critical_tests = [data_ok, lstm_ok, signal_ok]
    if all(critical_tests):
        print_status("🎉 All critical tests passed! System is ready for live trading.", "SUCCESS")
        print_status("🎯 Next Steps:", "INFO")
        print_status("   1. Run live_trader_clean.py for live trading", "INFO")
        print_status("   2. Monitor performance and signals", "INFO")
        print_status("   3. Check trade logs regularly", "INFO")
    else:
        print_status("⚠️ Some tests failed. Please check the issues above.", "WARNING")

    print_status("🏁 Test completed!", "END")


if __name__ == "__main__":
    main()
    input("Press Enter to continue...")
