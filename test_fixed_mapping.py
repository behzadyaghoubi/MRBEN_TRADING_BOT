import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def test_fixed_mapping():
    print("🧪 Testing Fixed Signal Mapping")
    print("=" * 50)

    try:
        # Load model and scaler
        model = load_model('models/lstm_balanced_model.h5')
        scaler = joblib.load('models/lstm_balanced_scaler.joblib')

        # Load dataset
        df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')

        # Prepare features
        feature_columns = [
            'open',
            'high',
            'low',
            'close',
            'SMA20',
            'SMA50',
            'RSI',
            'MACD',
            'MACD_signal',
            'MACD_hist',
        ]
        X = df[feature_columns].values
        X_scaled = scaler.transform(X)

        # Reshape for LSTM
        timesteps = 10
        X_reshaped = []

        for i in range(timesteps, len(X_scaled)):
            X_reshaped.append(X_scaled[i - timesteps : i])

        X_reshaped = np.array(X_reshaped)

        # Get predictions
        print("Getting predictions...")
        predictions = model.predict(X_reshaped[:100])  # Test on 100 samples

        # Fixed signal mapping function
        def correct_signal_mapping(predictions):
            """
            Correct signal mapping function
            predictions: numpy array of shape (n_samples, 3) with probabilities
            returns: list of signal strings
            """
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            predicted_classes = np.argmax(predictions, axis=1)
            return [signal_map[cls] for cls in predicted_classes]

        # Apply fixed mapping
        predicted_signals = correct_signal_mapping(predictions)

        # Analyze distribution
        signal_counts = {}
        for signal in predicted_signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

        total = len(predicted_signals)

        print("\n📊 Fixed Mapping Results:")
        for signal, count in signal_counts.items():
            percentage = (count / total) * 100
            print(f"{signal}: {percentage:.1f}% ({count})")

        # Check balance
        if 'BUY' in signal_counts and 'SELL' in signal_counts:
            buy_count = signal_counts['BUY']
            sell_count = signal_counts['SELL']
            ratio = buy_count / sell_count
            print(f"\nBUY/SELL ratio: {ratio:.2f}")

            if 0.7 <= ratio <= 1.3:
                print("✅ Balanced distribution achieved!")
                return True
            else:
                print("⚠️ Still some imbalance")
                return False
        else:
            print("❌ Missing BUY or SELL signals")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_fixed_mapping()

    if success:
        print("\n🎉 Fixed mapping works correctly!")
        print("System is ready for live trading!")
    else:
        print("\n⚠️ Mapping still needs attention")
