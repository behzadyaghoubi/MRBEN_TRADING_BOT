import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_final_lstm_model():
    """Create a final LSTM model with different approach"""

    print("🏗️ Creating Final LSTM Model...")

    model = Sequential(
        [
            # Input layer
            Input(shape=(10, 10)),
            # First LSTM layer with more units
            LSTM(64, return_sequences=True, activation='relu'),
            Dropout(0.4),
            BatchNormalization(),
            # Second LSTM layer
            LSTM(32, return_sequences=False, activation='relu'),
            Dropout(0.4),
            BatchNormalization(),
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dropout(0.2),
            # Output layer
            Dense(3, activation='softmax'),
        ]
    )

    # Compile with different settings
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    print("✅ Final model created successfully!")
    return model


def prepare_data_final():
    """Prepare data with different approach"""

    print("📊 Loading and preparing data...")

    # Load balanced dataset
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

    # Encode signal labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_original)

    print("📈 Signal distribution after encoding:")
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts, strict=False):
        signal_name = label_encoder.inverse_transform([label])[0]
        percentage = (count / len(y)) * 100
        print(f"  {signal_name} (Class {label}): {percentage:.1f}% ({count})")

    # Use StandardScaler instead of MinMaxScaler
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences with different approach
    timesteps = 10
    X_sequences = []
    y_sequences = []

    for i in range(timesteps, len(X_scaled)):
        X_sequences.append(X_scaled[i - timesteps : i])
        y_sequences.append(y[i])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    # Shuffle data before splitting
    indices = np.arange(len(X_sequences))
    np.random.shuffle(indices)
    X_sequences = X_sequences[indices]
    y_sequences = y_sequences[indices]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
    )

    print("📈 Data prepared:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[2]}")
    print(f"  Timesteps: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler, label_encoder


def train_final_model():
    """Train the final LSTM model"""

    print("🚀 Starting Final LSTM Training...")
    print("=" * 50)

    # Prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_data_final()

    # Create model
    model = create_final_lstm_model()

    # Callbacks with different settings
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',  # Monitor accuracy instead of loss
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=8, min_lr=1e-7, verbose=1),
    ]

    # Train model with different settings
    print("🎓 Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,  # More epochs
        batch_size=16,  # Smaller batch size
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate model
    print("\n📊 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Test predictions
    print("\n🔍 Testing predictions...")
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    # Analyze distribution
    unique, counts = np.unique(predicted_classes, return_counts=True)

    print("Prediction distribution:")
    for class_id, count in zip(unique, counts, strict=False):
        percentage = (count / len(predictions)) * 100
        signal_name = label_encoder.inverse_transform([class_id])[0]
        print(f"  {signal_name}: {percentage:.1f}% ({count})")

    # Save model, scaler, and label encoder
    print("\n💾 Saving model, scaler, and label encoder...")
    model.save('models/lstm_final_fixed.h5')
    joblib.dump(scaler, 'models/lstm_final_fixed_scaler.joblib')
    joblib.dump(label_encoder, 'models/lstm_final_fixed_label_encoder.joblib')

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    # Plot prediction distribution
    true_dist = np.bincount(y_test, minlength=3)
    pred_dist = np.bincount(predicted_classes, minlength=3)

    x = np.arange(3)
    width = 0.35

    plt.bar(x - width / 2, true_dist, width, label='True', alpha=0.8)
    plt.bar(x + width / 2, pred_dist, width, label='Predicted', alpha=0.8)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('True vs Predicted Distribution')
    plt.xticks(x, ['SELL', 'HOLD', 'BUY'])
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/lstm_final_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Training completed successfully!")
    return model, scaler, label_encoder


def test_final_model():
    """Test the final model with various inputs"""

    print("\n🧪 Testing Final Model...")
    print("=" * 40)

    try:
        # Load model and components
        from tensorflow.keras.models import load_model

        model = load_model('models/lstm_final_fixed.h5')
        scaler = joblib.load('models/lstm_final_fixed_scaler.joblib')
        label_encoder = joblib.load('models/lstm_final_fixed_label_encoder.joblib')

        # Test cases
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

        return True

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Final LSTM Fix Process...")
    print("=" * 60)

    # Train final model
    model, scaler, label_encoder = train_final_model()

    # Test model
    test_success = test_final_model()

    if test_success:
        print("\n✅ Final LSTM fix completed successfully!")
        print("🎯 Model should now have balanced predictions.")
    else:
        print("\n❌ Testing failed.")
