#!/usr/bin/env python3
"""
Final LSTM Retraining Script
Complete retraining with real market data
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

def check_data():
    """Check if data files exist."""
    print("🔍 Checking data files...")
    
    sequences_path = "data/real_market_sequences.npy"
    labels_path = "data/real_market_labels.npy"
    
    if not os.path.exists(sequences_path):
        print(f"❌ Sequences file not found: {sequences_path}")
        return False
    
    if not os.path.exists(labels_path):
        print(f"❌ Labels file not found: {labels_path}")
        return False
    
    print("✅ Data files found")
    return True

def load_data():
    """Load the real market data."""
    print("📊 Loading real market data...")
    
    sequences = np.load("data/real_market_sequences.npy")
    labels = np.load("data/real_market_labels.npy")
    
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
    
    return sequences, labels

def prepare_data(sequences, labels):
    """Prepare data for training."""
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
    
    return X_train, X_test, y_train, y_test

def create_model(input_shape):
    """Create LSTM model."""
    print("🏗️ Creating LSTM model...")
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
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
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model."""
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
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    print("📊 Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    accuracy = np.mean(y_pred_classes == y_test_classes)
    
    print(f"✅ Model Evaluation:")
    print(f"   Test Accuracy: {accuracy:.4f}")
    
    # Check prediction distribution
    unique_pred, pred_counts = np.unique(y_pred_classes, return_counts=True)
    print(f"   Prediction distribution:")
    for pred_class, count in zip(unique_pred, pred_counts):
        percentage = (count / len(y_pred_classes)) * 100
        signal_type = ["SELL", "HOLD", "BUY"][pred_class]
        print(f"     {signal_type}: {count} ({percentage:.1f}%)")
    
    return accuracy

def save_model_and_scaler(model, sequences):
    """Save model and scaler."""
    print("💾 Saving model and scaler...")
    
    # Save model
    model.save('models/mrben_lstm_real_data.h5')
    
    # Create and save scaler
    scaler = MinMaxScaler()
    n_samples, n_timesteps, n_features = sequences.shape
    sequences_reshaped = sequences.reshape(-1, n_features)
    scaler.fit(sequences_reshaped)
    joblib.dump(scaler, 'models/mrben_lstm_real_data_scaler.save')
    
    print("✅ Model and scaler saved:")
    print(f"   Model: models/mrben_lstm_real_data.h5")
    print(f"   Scaler: models/mrben_lstm_real_data_scaler.save")

def main():
    """Main function."""
    print("🎯 LSTM Retraining with Real Market Data")
    print("=" * 60)
    
    # Check data
    if not check_data():
        return
    
    # Load data
    sequences, labels = load_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(sequences, labels)
    
    # Create model
    model = create_model((X_train.shape[1], X_train.shape[2]))
    
    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save model and scaler
    save_model_and_scaler(model, sequences)
    
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