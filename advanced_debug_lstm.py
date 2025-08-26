import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def analyze_model_behavior():
    """Advanced analysis of LSTM model behavior"""

    print("🔍 Advanced LSTM Model Behavior Analysis")
    print("=" * 60)

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
        y_original = df['signal'].values

        # Scale features
        X_scaled = scaler.transform(X)

        # Create sequences
        timesteps = 10
        X_sequences = []
        y_sequences = []

        for i in range(timesteps, len(X_scaled)):
            X_sequences.append(X_scaled[i - timesteps : i])
            y_sequences.append(y_original[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        # Get predictions on all data
        print("📊 Getting predictions on all data...")
        predictions = model.predict(X_sequences, verbose=0)

        # Analyze raw probabilities
        print("\n📈 Raw Probability Analysis:")
        print("-" * 40)

        sell_probs = predictions[:, 0]
        hold_probs = predictions[:, 1]
        buy_probs = predictions[:, 2]

        print(
            f"SELL probabilities - Mean: {np.mean(sell_probs):.4f}, Std: {np.std(sell_probs):.4f}"
        )
        print(
            f"HOLD probabilities - Mean: {np.mean(hold_probs):.4f}, Std: {np.std(hold_probs):.4f}"
        )
        print(f"BUY probabilities - Mean: {np.mean(buy_probs):.4f}, Std: {np.std(buy_probs):.4f}")

        # Check probability ranges
        print("\n📊 Probability Ranges:")
        print(f"SELL: [{np.min(sell_probs):.4f}, {np.max(sell_probs):.4f}]")
        print(f"HOLD: [{np.min(hold_probs):.4f}, {np.max(hold_probs):.4f}]")
        print(f"BUY: [{np.min(buy_probs):.4f}, {np.max(buy_probs):.4f}]")

        # Analyze class predictions
        predicted_classes = np.argmax(predictions, axis=1)
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

        unique, counts = np.unique(predicted_classes, return_counts=True)

        print("\n🎯 Class Prediction Distribution:")
        for class_id, count in zip(unique, counts, strict=False):
            percentage = (count / len(predictions)) * 100
            signal_name = signal_map[class_id]
            print(f"  {signal_name}: {percentage:.1f}% ({count})")

        # Compare with original labels
        print("\n📊 Original vs Predicted Comparison:")
        original_counts = {}
        for signal in y_sequences:
            original_counts[signal] = original_counts.get(signal, 0) + 1

        print("Original distribution:")
        for signal, count in original_counts.items():
            percentage = (count / len(y_sequences)) * 100
            print(f"  {signal}: {percentage:.1f}% ({count})")

        # Find samples where predictions differ significantly
        print("\n🔍 Analyzing Prediction Confidence:")

        # Get confidence scores
        confidence_scores = np.max(predictions, axis=1)

        print(f"Average confidence: {np.mean(confidence_scores):.4f}")
        print(
            f"Confidence range: [{np.min(confidence_scores):.4f}, {np.max(confidence_scores):.4f}]"
        )

        # Find low confidence predictions
        low_conf_threshold = 0.4
        low_conf_indices = np.where(confidence_scores < low_conf_threshold)[0]

        print(f"Low confidence predictions (< {low_conf_threshold}): {len(low_conf_indices)}")

        if len(low_conf_indices) > 0:
            print("Sample low confidence predictions:")
            for i in low_conf_indices[:5]:
                pred = predictions[i]
                print(f"  Sample {i}: SELL={pred[0]:.4f}, HOLD={pred[1]:.4f}, BUY={pred[2]:.4f}")

        # Analyze feature importance (correlation with predictions)
        print("\n🔍 Feature-Prediction Correlation Analysis:")

        # Get the last timestep features (most recent data)
        last_features = X_sequences[:, -1, :]  # Shape: (n_samples, n_features)

        # Calculate correlations with each class probability
        feature_names = feature_columns

        for class_idx, class_name in enumerate(['SELL', 'HOLD', 'BUY']):
            class_probs = predictions[:, class_idx]
            correlations = []

            for feat_idx, feat_name in enumerate(feature_names):
                corr = np.corrcoef(last_features[:, feat_idx], class_probs)[0, 1]
                correlations.append((feat_name, corr))

            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            print(f"\n{class_name} class correlations:")
            for feat_name, corr in correlations[:5]:
                print(f"  {feat_name}: {corr:.4f}")

        return True

    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_with_extreme_inputs():
    """Test model with extreme input values to understand behavior"""

    print("\n🧪 Testing Model with Extreme Inputs...")
    print("=" * 50)

    try:
        model = load_model('models/lstm_balanced_model.h5')
        scaler = joblib.load('models/lstm_balanced_scaler.joblib')

        # Create extreme test cases
        timesteps = 10
        n_features = 10

        test_cases = [
            ("All High Values", np.ones((timesteps, n_features)) * 2.0),
            ("All Low Values", np.ones((timesteps, n_features)) * -2.0),
            ("Random Values", np.random.randn(timesteps, n_features)),
            ("Zero Values", np.zeros((timesteps, n_features))),
            ("Alternating High/Low", np.array([[1.0, -1.0] * 5] * timesteps)),
        ]

        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

        for case_name, test_data in test_cases:
            print(f"\n📊 Test Case: {case_name}")

            # Scale the test data
            test_scaled = scaler.transform(test_data)

            # Reshape for LSTM
            test_sequence = test_scaled.reshape(1, timesteps, n_features)

            # Get prediction
            prediction = model.predict(test_sequence, verbose=0)[0]

            # Analyze prediction
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            print(f"  Predicted: {signal_map[predicted_class]} (Class {predicted_class})")
            print(f"  Confidence: {confidence:.4f}")
            print(
                f"  Probabilities: SELL={prediction[0]:.4f}, HOLD={prediction[1]:.4f}, BUY={prediction[2]:.4f}"
            )

        return True

    except Exception as e:
        print(f"❌ Error in extreme testing: {e}")
        return False


def suggest_fixes():
    """Suggest specific fixes based on analysis"""

    print("\n💡 Suggested Fixes Based on Analysis:")
    print("=" * 50)

    print("1. 🎯 Model Architecture Issues:")
    print("   - Current model may be too complex for the data")
    print("   - Consider simpler architecture with fewer layers")
    print("   - Add batch normalization to stabilize training")

    print("\n2. 📊 Data Preprocessing Issues:")
    print("   - Check if scaling is appropriate for the data")
    print("   - Consider using StandardScaler instead of MinMaxScaler")
    print("   - Verify sequence creation logic")

    print("\n3. 🎓 Training Issues:")
    print("   - Model may be overfitting to training data")
    print("   - Increase dropout rates")
    print("   - Use different learning rate schedules")
    print("   - Implement learning rate warmup")

    print("\n4. 🔧 Immediate Actions:")
    print("   - Retrain with simpler architecture")
    print("   - Use different loss function (focal loss)")
    print("   - Implement data augmentation")
    print("   - Add regularization techniques")


if __name__ == "__main__":
    print("🚀 Starting Advanced LSTM Debug Analysis...")

    # Run advanced analysis
    analysis_success = analyze_model_behavior()

    if analysis_success:
        # Test with extreme inputs
        test_success = test_model_with_extreme_inputs()

        # Suggest fixes
        suggest_fixes()

        print("\n🎯 Advanced analysis completed!")
    else:
        print("\n❌ Advanced analysis failed.")
