import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model


def test_model_comprehensive():
    """Comprehensive test of the simple LSTM model"""

    print("🧪 Comprehensive Simple LSTM Model Test")
    print("=" * 60)

    try:
        # Load model and components
        model = load_model('models/lstm_simple_fixed.h5')
        scaler = joblib.load('models/lstm_simple_fixed_scaler.joblib')
        label_encoder = joblib.load('models/lstm_simple_fixed_label_encoder.joblib')

        # Load test data
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
        X = df[feature_columns].astype(float).values
        y_original = df['signal'].values

        # Encode labels
        y_encoded = label_encoder.transform(y_original)

        # Scale features
        X_scaled = scaler.transform(X)

        # Create sequences
        timesteps = 10
        X_sequences = []
        y_sequences = []

        for i in range(timesteps, len(X_scaled)):
            X_sequences.append(X_scaled[i - timesteps : i])
            y_sequences.append(y_encoded[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        # Get predictions
        print("📊 Getting predictions on all data...")
        predictions = model.predict(X_sequences, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # Analyze distribution
        print("\n📈 Prediction Distribution Analysis:")
        print("-" * 40)

        unique_pred, counts_pred = np.unique(predicted_classes, return_counts=True)
        unique_true, counts_true = np.unique(y_sequences, return_counts=True)

        print("True distribution:")
        for class_id, count in zip(unique_true, counts_true, strict=False):
            percentage = (count / len(y_sequences)) * 100
            signal_name = label_encoder.inverse_transform([class_id])[0]
            print(f"  {signal_name}: {percentage:.1f}% ({count})")

        print("\nPredicted distribution:")
        for class_id, count in zip(unique_pred, counts_pred, strict=False):
            percentage = (count / len(predictions)) * 100
            signal_name = label_encoder.inverse_transform([class_id])[0]
            print(f"  {signal_name}: {percentage:.1f}% ({count})")

        # Calculate accuracy
        accuracy = np.mean(predicted_classes == y_sequences)
        print(f"\n🎯 Overall Accuracy: {accuracy:.4f}")

        # Detailed classification report
        print("\n📊 Detailed Classification Report:")
        print("-" * 40)

        y_true_names = label_encoder.inverse_transform(y_sequences)
        y_pred_names = label_encoder.inverse_transform(predicted_classes)

        report = classification_report(y_true_names, y_pred_names, output_dict=True)
        print(classification_report(y_true_names, y_pred_names))

        # Confusion matrix
        print("\n🔍 Confusion Matrix:")
        print("-" * 40)

        cm = confusion_matrix(y_sequences, predicted_classes)
        cm_df = pd.DataFrame(
            cm,
            index=[label_encoder.inverse_transform([i])[0] for i in range(len(cm))],
            columns=[label_encoder.inverse_transform([i])[0] for i in range(len(cm))],
        )

        print(cm_df)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Simple LSTM Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/simple_lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Confidence analysis
        print("\n🔍 Confidence Analysis:")
        print("-" * 40)

        confidence_scores = np.max(predictions, axis=1)
        print(f"Average confidence: {np.mean(confidence_scores):.4f}")
        print(
            f"Confidence range: [{np.min(confidence_scores):.4f}, {np.max(confidence_scores):.4f}]"
        )
        print(f"Standard deviation: {np.std(confidence_scores):.4f}")

        # High confidence predictions
        high_conf_threshold = 0.7
        high_conf_mask = confidence_scores >= high_conf_threshold
        high_conf_accuracy = np.mean(
            predicted_classes[high_conf_mask] == y_sequences[high_conf_mask]
        )

        print(f"\nHigh confidence predictions (≥{high_conf_threshold}):")
        print(
            f"  Count: {np.sum(high_conf_mask)} ({np.sum(high_conf_mask)/len(predictions)*100:.1f}%)"
        )
        print(f"  Accuracy: {high_conf_accuracy:.4f}")

        # Test with extreme cases
        print("\n🧪 Extreme Case Testing:")
        print("-" * 40)

        test_cases = [
            ("Strong BUY Signal", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 85.0, 0.8, 0.6, 0.2]),
            ("Strong SELL Signal", [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 15.0, -0.8, -0.6, -0.2]),
            ("Neutral HOLD Signal", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0, 0.0, 0.0, 0.0]),
            ("Mixed Signals", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 60.0, 0.2, 0.1, 0.1]),
        ]

        for case_name, test_data in test_cases:
            print(f"\n📊 Test Case: {case_name}")

            # Create sequence
            test_sequence = np.array([test_data] * 10).reshape(1, 10, 10)

            # Scale
            test_scaled = scaler.transform(test_sequence.reshape(-1, 10)).reshape(1, 10, 10)

            # Predict
            prediction = model.predict(test_scaled, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            signal_name = label_encoder.inverse_transform([predicted_class])[0]
            print(f"  Predicted: {signal_name} (Class {predicted_class})")
            print(f"  Confidence: {confidence:.4f}")

            # Show all probabilities
            for i, prob in enumerate(prediction):
                signal = label_encoder.inverse_transform([i])[0]
                print(f"  {signal}: {prob:.4f}")

        # Balance assessment
        print("\n⚖️ Balance Assessment:")
        print("-" * 40)

        # Calculate balance ratio
        pred_distribution = np.bincount(predicted_classes, minlength=len(label_encoder.classes_))
        ideal_balance = len(predictions) / len(label_encoder.classes_)

        balance_ratios = pred_distribution / ideal_balance
        print("Balance ratios (1.0 = perfect balance):")
        for i, ratio in enumerate(balance_ratios):
            signal_name = label_encoder.inverse_transform([i])[0]
            print(f"  {signal_name}: {ratio:.3f}")

        overall_balance = 1 - np.std(balance_ratios)
        print(f"\nOverall balance score: {overall_balance:.4f}")

        # Final assessment
        print("\n✅ Final Assessment:")
        print("-" * 40)

        if accuracy >= 0.6:
            print("✅ Accuracy: GOOD")
        else:
            print("❌ Accuracy: NEEDS IMPROVEMENT")

        if overall_balance >= 0.8:
            print("✅ Balance: GOOD")
        else:
            print("❌ Balance: NEEDS IMPROVEMENT")

        if np.mean(confidence_scores) >= 0.5:
            print("✅ Confidence: GOOD")
        else:
            print("❌ Confidence: NEEDS IMPROVEMENT")

        print("\n🎯 Model Performance Summary:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balance Score: {overall_balance:.4f}")
        print(f"  Average Confidence: {np.mean(confidence_scores):.4f}")

        return True

    except Exception as e:
        print(f"❌ Error in comprehensive test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Comprehensive Model Test...")
    success = test_model_comprehensive()

    if success:
        print("\n✅ Comprehensive test completed successfully!")
    else:
        print("\n❌ Comprehensive test failed.")
