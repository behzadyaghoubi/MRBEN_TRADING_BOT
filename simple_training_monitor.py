#!/usr/bin/env python3
"""
Simple LSTM Training with Active Monitoring
"""

import os
import time
from datetime import datetime

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def print_status(message, level="INFO"):
    """Print status with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def main():
    """Main training function with active monitoring"""
    start_time = time.time()

    print_status("🎯 Starting LSTM Training with Active Monitoring", "START")
    print_status("=" * 60, "INFO")

    # Check data files
    sequences_path = "data/real_market_sequences.npy"
    labels_path = "data/real_market_labels.npy"

    if not os.path.exists(sequences_path):
        print_status(f"❌ Sequences file not found: {sequences_path}", "ERROR")
        return False

    if not os.path.exists(labels_path):
        print_status(f"❌ Labels file not found: {labels_path}", "ERROR")
        return False

    print_status("✅ Data files found", "SUCCESS")

    # Load data
    print_status("📊 Loading real market data...", "INFO")
    try:
        sequences = np.load(sequences_path)
        labels = np.load(labels_path)
        print_status("✅ Data loaded successfully", "SUCCESS")
        print_status(f"   Sequences: {sequences.shape}", "INFO")
        print_status(f"   Labels: {labels.shape}", "INFO")
    except Exception as e:
        print_status(f"❌ Error loading data: {e}", "ERROR")
        return False

    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print_status("   Label distribution:", "INFO")
    for label, count in zip(unique_labels, counts, strict=False):
        percentage = (count / len(labels)) * 100
        signal_type = ["SELL", "HOLD", "BUY"][label]
        print_status(f"     {signal_type}: {count} ({percentage:.1f}%)", "INFO")

    # Prepare data
    print_status("🔧 Preparing data for training...", "INFO")
    try:
        X = sequences
        y = to_categorical(labels, num_classes=3)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=labels
        )

        print_status("✅ Data prepared:", "SUCCESS")
        print_status(f"   Training: {len(X_train)} samples", "INFO")
        print_status(f"   Test: {len(X_test)} samples", "INFO")
        print_status(f"   Features: {X_train.shape[2]}", "INFO")
        print_status(f"   Timesteps: {X_train.shape[1]}", "INFO")
    except Exception as e:
        print_status(f"❌ Error preparing data: {e}", "ERROR")
        return False

    # Create model
    print_status("🏗️ Creating LSTM model...", "INFO")
    try:
        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dropout(0.1),
                Dense(3, activation='softmax'),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        print_status(f"✅ Model created with {model.count_params()} parameters", "SUCCESS")
    except Exception as e:
        print_status(f"❌ Error creating model: {e}", "ERROR")
        return False

    # Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
        print_status("✅ Created models directory", "SUCCESS")

    # Train model
    print_status("🎯 Starting LSTM training...", "INFO")
    print_status("🚀 Training will begin in 3 seconds...", "INFO")
    time.sleep(3)

    try:
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            'models/mrben_lstm_real_data_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        )

        print_status("🔥 Training started! Monitoring progress...", "TRAINING")

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1,
        )

        print_status("✅ Training completed successfully", "SUCCESS")
    except Exception as e:
        print_status(f"❌ Error during training: {e}", "ERROR")
        return False

    # Evaluate
    print_status("📊 Evaluating model...", "INFO")
    try:
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        accuracy = np.mean(y_pred_classes == y_test_classes)
        print_status(f"✅ Test Accuracy: {accuracy:.4f}", "SUCCESS")

        # Check prediction distribution
        unique_pred, pred_counts = np.unique(y_pred_classes, return_counts=True)
        print_status("   Prediction distribution:", "INFO")
        for pred_class, count in zip(unique_pred, pred_counts, strict=False):
            percentage = (count / len(y_pred_classes)) * 100
            signal_type = ["SELL", "HOLD", "BUY"][pred_class]
            print_status(f"     {signal_type}: {count} ({percentage:.1f}%)", "INFO")
    except Exception as e:
        print_status(f"❌ Error during evaluation: {e}", "ERROR")
        return False

    # Save model and scaler
    print_status("💾 Saving model and scaler...", "INFO")
    try:
        model.save('models/mrben_lstm_real_data.h5')

        scaler = MinMaxScaler()
        n_samples, n_timesteps, n_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, n_features)
        scaler.fit(sequences_reshaped)
        joblib.dump(scaler, 'models/mrben_lstm_real_data_scaler.save')

        print_status("✅ Model and scaler saved:", "SUCCESS")
        print_status("   Model: models/mrben_lstm_real_data.h5", "INFO")
        print_status("   Scaler: models/mrben_lstm_real_data_scaler.save", "INFO")
    except Exception as e:
        print_status(f"❌ Error saving model: {e}", "ERROR")
        return False

    # Calculate training duration
    training_duration = time.time() - start_time
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    seconds = int(training_duration % 60)

    # Final report
    print_status("🎉 LSTM Training Completed Successfully!", "SUCCESS")
    print_status("=" * 50, "INFO")
    print_status(f"   Final Test Accuracy: {accuracy:.4f}", "INFO")
    print_status(f"   Training Epochs: {len(history.history['loss'])}", "INFO")
    print_status(f"   Training Duration: {hours:02d}:{minutes:02d}:{seconds:02d}", "INFO")
    print_status("   Model saved: models/mrben_lstm_real_data.h5", "INFO")
    print_status("   Scaler saved: models/mrben_lstm_real_data_scaler.save", "INFO")

    print_status("🎯 Next Steps:", "INFO")
    print_status("   1. Test system: test_complete_system_real_data.py", "INFO")
    print_status("   2. Run live trading: live_trader_clean.py", "INFO")
    print_status("   3. Monitor performance", "INFO")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print_status("🎉 Training completed successfully!", "SUCCESS")
    else:
        print_status("❌ Training failed!", "ERROR")

    print_status("Press Enter to exit...", "INFO")
    input()
