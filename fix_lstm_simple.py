import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_simple_lstm_model():
    """Create a simple LSTM model to avoid overfitting"""

    print("🏗️ Creating Simple LSTM Model...")

    model = Sequential(
        [
            # Simple LSTM layer
            LSTM(32, input_shape=(10, 10), return_sequences=False, activation='tanh'),
            Dropout(0.3),
            # Batch normalization
            BatchNormalization(),
            # Simple dense layers
            Dense(16, activation='relu'),
            Dropout(0.2),
            # Output layer
            Dense(3, activation='softmax'),
        ]
    )

    # Compile with simple settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    print("✅ Simple model created successfully!")
    return model


def prepare_data():
    """Prepare balanced dataset"""

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

    # Encode signal labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['signal'].values)

    print("📈 Signal distribution after encoding:")
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts, strict=False):
        signal_name = label_encoder.inverse_transform([label])[0]
        percentage = (count / len(y)) * 100
        print(f"  {signal_name} (Class {label}): {percentage:.1f}% ({count})")

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    timesteps = 10
    X_sequences = []
    y_sequences = []

    for i in range(timesteps, len(X_scaled)):
        X_sequences.append(X_scaled[i - timesteps : i])
        y_sequences.append(y[i])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

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


def train_simple_model():
    """Train the simple LSTM model"""

    print("🚀 Starting Simple LSTM Training...")
    print("=" * 50)

    # Prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_data()

    # Create model
    model = create_simple_lstm_model()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    # Train model
    print("🎓 Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
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
    model.save('models/lstm_simple_fixed.h5')
    joblib.dump(scaler, 'models/lstm_simple_fixed_scaler.joblib')
    joblib.dump(label_encoder, 'models/lstm_simple_fixed_label_encoder.joblib')

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/lstm_simple_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Training completed successfully!")
    return model, scaler, label_encoder


def test_simple_model():
    """Test the simple model with various inputs"""

    print("\n🧪 Testing Simple Model...")
    print("=" * 40)

    try:
        # Load model and components
        from tensorflow.keras.models import load_model

        model = load_model('models/lstm_simple_fixed.h5')
        scaler = joblib.load('models/lstm_simple_fixed_scaler.joblib')
        label_encoder = joblib.load('models/lstm_simple_fixed_label_encoder.joblib')

        # Test cases
        test_cases = [
            ("High RSI, High MACD", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 80.0, 0.5, 0.3, 0.2]),
            ("Low RSI, Low MACD", [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 20.0, -0.5, -0.3, -0.2]),
            ("Neutral Values", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0, 0.0, 0.0, 0.0]),
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
    print("🚀 Starting Simple LSTM Fix Process...")
    print("=" * 60)

    # Train simple model
    model, scaler, label_encoder = train_simple_model()

    # Test model
    test_success = test_simple_model()

    if test_success:
        print("\n✅ Simple LSTM fix completed successfully!")
        print("🎯 Model should now have balanced predictions.")
    else:
        print("\n❌ Testing failed.")
