#!/usr/bin/env python3
"""
Simple LSTM Model Test
"""

import os
from datetime import datetime

import joblib
import numpy as np
from tensorflow.keras.models import load_model


def print_status(message, level="INFO"):
    """Print status with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def main():
    """Simple LSTM model test"""
    print_status("🧪 Simple LSTM Model Test", "START")

    # Check model files
    model_path = "models/mrben_lstm_real_data.h5"
    scaler_path = "models/mrben_lstm_real_data_scaler.save"

    if not os.path.exists(model_path):
        print_status(f"❌ Model file not found: {model_path}", "ERROR")
        return False

    if not os.path.exists(scaler_path):
        print_status(f"❌ Scaler file not found: {scaler_path}", "ERROR")
        return False

    print_status("✅ Model files found", "SUCCESS")

    # Load model and scaler
    print_status("📊 Loading model...", "INFO")
    try:
        model = load_model(model_path)
        print_status(f"✅ Model loaded: {model.count_params():,} parameters", "SUCCESS")
    except Exception as e:
        print_status(f"❌ Error loading model: {e}", "ERROR")
        return False

    print_status("📊 Loading scaler...", "INFO")
    try:
        scaler = joblib.load(scaler_path)
        print_status("✅ Scaler loaded successfully", "SUCCESS")
    except Exception as e:
        print_status(f"❌ Error loading scaler: {e}", "ERROR")
        return False

    # Load test data
    sequences_path = "data/real_market_sequences.npy"
    labels_path = "data/real_market_labels.npy"

    if not os.path.exists(sequences_path):
        print_status(f"❌ Sequences file not found: {sequences_path}", "ERROR")
        return False

    print_status("📊 Loading test data...", "INFO")
    try:
        sequences = np.load(sequences_path)
        labels = np.load(labels_path)
        print_status(f"✅ Test data loaded: {sequences.shape}", "SUCCESS")
    except Exception as e:
        print_status(f"❌ Error loading test data: {e}", "ERROR")
        return False

    # Test predictions
    print_status("🧪 Testing predictions...", "INFO")
    try:
        # Use last 10 samples for testing
        test_samples = sequences[-10:]
        test_labels = labels[-10:]

        # Scale the data
        n_samples, n_timesteps, n_features = test_samples.shape
        test_reshaped = test_samples.reshape(-1, n_features)
        test_scaled = scaler.transform(test_reshaped)
        test_scaled = test_scaled.reshape(n_samples, n_timesteps, n_features)

        # Make predictions
        predictions = model.predict(test_scaled, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # Calculate accuracy
        correct = np.sum(predicted_classes == test_labels)
        accuracy = correct / len(test_labels)

        print_status("✅ Test Results:", "SUCCESS")
        print_status(f"   Test samples: {len(test_samples)}", "INFO")
        print_status(f"   Correct predictions: {correct}/{len(test_labels)}", "INFO")
        print_status(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", "INFO")

        # Show individual predictions
        print_status("📊 Individual Predictions:", "INFO")
        for i in range(len(test_samples)):
            pred_class = predicted_classes[i]
            actual_class = test_labels[i]
            confidence = np.max(predictions[i])

            pred_signal = ["SELL", "HOLD", "BUY"][pred_class]
            actual_signal = ["SELL", "HOLD", "BUY"][actual_class]

            status = "✅" if pred_class == actual_class else "❌"
            print_status(
                f"   Sample {i+1}: {status} Pred={pred_signal} (Conf={confidence:.3f}), Actual={actual_signal}",
                "INFO",
            )

        return True

    except Exception as e:
        print_status(f"❌ Prediction test error: {e}", "ERROR")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print_status("🎉 LSTM model test completed successfully!", "SUCCESS")
    else:
        print_status("❌ LSTM model test failed!", "ERROR")

    print_status("Press Enter to continue...", "INFO")
    input()
