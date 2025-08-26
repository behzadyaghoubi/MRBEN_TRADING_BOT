#!/usr/bin/env python3
"""
Execute Training Final
Final execution of LSTM training with keyboard management
"""

import os
import sys
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Import keyboard manager
try:
    from keyboard_manager import ensure_english_keyboard
    KEYBOARD_MANAGER_AVAILABLE = True
except ImportError:
    KEYBOARD_MANAGER_AVAILABLE = False
    print("⚠️  Keyboard manager not available")

def main():
    """Main training function"""
    print("🎯 LSTM Retraining with Real Market Data")
    print("=" * 60)
    
    # Ensure English keyboard before starting
    if KEYBOARD_MANAGER_AVAILABLE:
        print("🔧 Ensuring English keyboard layout...")
        if not ensure_english_keyboard():
            print("❌ Cannot proceed without English keyboard")
            print("⚠️  Please manually switch to English and run again")
            return
        print("✅ Keyboard layout confirmed as English")
    else:
        print("⚠️  Please ensure keyboard is set to English before proceeding...")
        input("Press Enter when ready...")
    
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
    
    print(f"✅ Data loaded:")
    print(f"   Sequences: {sequences.shape}")
    print(f"   Labels: {labels.shape}")
    
    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"   Label distribution:")
    for label, count in zip(unique_labels, counts):
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
    
    print(f"✅ Data prepared:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features: {X_train.shape[2]}")
    print(f"   Timesteps: {X_train.shape[1]}")
    
    # Create model
    print("🏗️ Creating LSTM model...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created with {model.count_params()} parameters")
    
    # Train model
    print("🎯 Training LSTM model...")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'models/mrben_lstm_real_data_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
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
    print(f"   Prediction distribution:")
    for pred_class, count in zip(unique_pred, pred_counts):
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
    print(f"   Model: models/mrben_lstm_real_data.h5")
    print(f"   Scaler: models/mrben_lstm_real_data_scaler.save")
    
    # Final report
    print(f"\n🎉 LSTM Retraining Completed Successfully!")
    print(f"   Final Test Accuracy: {accuracy:.4f}")
    print(f"   Training Epochs: {len(history.history['loss'])}")
    print(f"   Model saved: models/mrben_lstm_real_data.h5")
    print(f"   Scaler saved: models/mrben_lstm_real_data_scaler.save")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Test system: test_complete_system_real_data.py")
    print(f"   2. Run live trading: live_trader_clean.py")
    print(f"   3. Monitor performance")

if __name__ == "__main__":
    main() 