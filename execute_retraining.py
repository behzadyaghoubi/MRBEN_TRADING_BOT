#!/usr/bin/env python3
"""
Execute LSTM Retraining with Real Data
Direct execution of the LSTM retraining process
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def load_real_data():
    """Load the collected real market data."""
    print("📊 Loading real market data...")

    sequences_path = "data/real_market_sequences.npy"
    labels_path = "data/real_market_labels.npy"

    if not os.path.exists(sequences_path) or not os.path.exists(labels_path):
        print("❌ Real market data files not found")
        return None, None

    sequences = np.load(sequences_path)
    labels = np.load(labels_path)

    print(f"✅ Loaded {len(sequences)} sequences with shape {sequences.shape}")
    print(f"   Labels shape: {labels.shape}")

    return sequences, labels


def prepare_data(sequences, labels):
    """Prepare data for training."""
    print("🔧 Preparing data for training...")

    # Reshape sequences for LSTM (samples, timesteps, features)
    X = sequences
    y = to_categorical(labels, num_classes=3)  # 3 classes: SELL(0), HOLD(1), BUY(2)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=labels
    )

    print("✅ Data prepared:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[2]}")
    print(f"   Timesteps: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def create_lstm_model(input_shape, num_classes=3):
    """Create LSTM model architecture."""
    print("🏗️ Creating LSTM model...")

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(num_classes, activation='softmax'),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']
    )

    print(f"✅ Model created with {model.count_params()} parameters")
    model.summary()

    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """Train the LSTM model."""
    print("🎯 Training LSTM model...")

    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        'models/mrben_lstm_real_data_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1,
    )

    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    print("📊 Evaluating model...")

    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_test_classes)

    # Class distribution
    unique_pred, pred_counts = np.unique(y_pred_classes, return_counts=True)
    unique_test, test_counts = np.unique(y_test_classes, return_counts=True)

    print("✅ Model Evaluation:")
    print(f"   Test Accuracy: {accuracy:.4f}")

    print("\n📈 Prediction Distribution:")
    for pred_class, count in zip(unique_pred, pred_counts, strict=False):
        percentage = (count / len(y_pred_classes)) * 100
        signal_type = ["SELL", "HOLD", "BUY"][pred_class]
        print(f"   {signal_type}: {count} ({percentage:.1f}%)")

    print("\n📈 Actual Distribution:")
    for test_class, count in zip(unique_test, test_counts, strict=False):
        percentage = (count / len(y_test_classes)) * 100
        signal_type = ["SELL", "HOLD", "BUY"][test_class]
        print(f"   {signal_type}: {count} ({percentage:.1f}%)")

    return accuracy, y_pred_classes, y_test_classes


def save_model_and_scaler(model, sequences):
    """Save the trained model and scaler."""
    print("💾 Saving model and scaler...")

    # Create scaler for the features
    scaler = MinMaxScaler()

    # Reshape sequences for fitting scaler
    n_samples, n_timesteps, n_features = sequences.shape
    sequences_reshaped = sequences.reshape(-1, n_features)

    # Fit scaler
    scaler.fit(sequences_reshaped)

    # Save model and scaler
    model.save('models/mrben_lstm_real_data.h5')
    joblib.dump(scaler, 'models/mrben_lstm_real_data_scaler.save')

    print("✅ Model and scaler saved:")
    print("   Model: models/mrben_lstm_real_data.h5")
    print("   Scaler: models/mrben_lstm_real_data_scaler.save")


def plot_training_history(history):
    """Plot training history."""
    print("📈 Plotting training history...")

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('models/lstm_real_data_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free memory

        print("✅ Training history saved: models/lstm_real_data_training_history.png")
    except Exception as e:
        print(f"⚠️ Could not save training plot: {e}")


def main():
    """Main function."""
    print("🎯 Retraining LSTM Model with Real Market Data")
    print("=" * 60)

    # Load real data
    sequences, labels = load_real_data()
    if sequences is None:
        return

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(sequences, labels)

    # Create model
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))

    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate model
    accuracy, y_pred, y_test_classes = evaluate_model(model, X_test, y_test)

    # Save model and scaler
    save_model_and_scaler(model, sequences)

    # Plot training history
    plot_training_history(history)

    print("\n🎉 LSTM Model Retraining Complete!")
    print(f"   Final Test Accuracy: {accuracy:.4f}")
    print("   Model saved: models/mrben_lstm_real_data.h5")
    print("   Scaler saved: models/mrben_lstm_real_data_scaler.save")

    print("\n🎯 Next Steps:")
    print("   1. Test the system with: test_complete_system_real_data.py")
    print("   2. Run live trading with: live_trader_clean.py")
    print("   3. Monitor signal distribution for balance")


if __name__ == "__main__":
    main()
