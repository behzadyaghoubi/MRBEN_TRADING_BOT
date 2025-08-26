#!/usr/bin/env python3
"""
Run Final Training
Direct execution of final LSTM training
"""

import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def main():
    print("🎯 LSTM Retraining with Real Market Data")
    print("=" * 60)

    # Check data files
    sequences_path = "data/real_market_sequences.npy"
    labels_path = "data/real_market_labels.npy"

    if not os.path.exists(sequences_path):
        print(f"❌ Sequences file not found: {sequences_path}")
        return

    if not os.path.exists(labels_path):
        print(f"❌ Labels file not found: {labels_path}")
        return

    print("✅ Data files found")

    # Load data
    print("📊 Loading real market data...")
    sequences = np.load(sequences_path)
    labels = np.load(labels_path)

    print("✅ Data loaded:")
    print(f"   Sequences: {sequences.shape}")
    print(f"   Labels: {labels.shape}")

    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("   Label distribution:")
    for label, count in zip(unique_labels, counts, strict=False):
        percentage = (count / len(labels)) * 100
        signal_type = ["SELL", "HOLD", "BUY"][label]
        print(f"     {signal_type}: {count} ({percentage:.1f}%)")

    # Prepare data
    print("🔧 Preparing data for training...")
    X = sequences
    y = to_categorical(labels, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=labels
    )

    print("✅ Data prepared:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features: {X_train.shape[2]}")
    print(f"   Timesteps: {X_train.shape[1]}")

    # Create model
    print("🏗️ Creating LSTM model...")
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
        optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']
    )

    print(f"✅ Model created with {model.count_params()} parameters")

    # Train model
    print("🎯 Training LSTM model...")
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        'models/mrben_lstm_real_data_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1,
    )

    # Evaluate
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_classes == y_test_classes)
    print(f"✅ Test Accuracy: {accuracy:.4f}")

    # Check prediction distribution
    unique_pred, pred_counts = np.unique(y_pred_classes, return_counts=True)
    print("   Prediction distribution:")
    for pred_class, count in zip(unique_pred, pred_counts, strict=False):
        percentage = (count / len(y_pred_classes)) * 100
        signal_type = ["SELL", "HOLD", "BUY"][pred_class]
        print(f"     {signal_type}: {count} ({percentage:.1f}%)")

    # Save model and scaler
    print("💾 Saving model and scaler...")
    model.save('models/mrben_lstm_real_data.h5')

    scaler = MinMaxScaler()
    n_samples, n_timesteps, n_features = sequences.shape
    sequences_reshaped = sequences.reshape(-1, n_features)
    scaler.fit(sequences_reshaped)
    joblib.dump(scaler, 'models/mrben_lstm_real_data_scaler.save')

    print("✅ Model and scaler saved:")
    print("   Model: models/mrben_lstm_real_data.h5")
    print("   Scaler: models/mrben_lstm_real_data_scaler.save")

    # Final report
    print("\n🎉 LSTM Retraining Completed Successfully!")
    print(f"   Final Test Accuracy: {accuracy:.4f}")
    print(f"   Training Epochs: {len(history.history['loss'])}")
    print("   Model saved: models/mrben_lstm_real_data.h5")
    print("   Scaler saved: models/mrben_lstm_real_data_scaler.save")

    print("\n🎯 Next Steps:")
    print("   1. Test system: test_complete_system_real_data.py")
    print("   2. Run live trading: live_trader_clean.py")
    print("   3. Monitor performance")


if __name__ == "__main__":
    main()
