#!/usr/bin/env python3
"""
Simple LSTM Retraining
Simplified version for direct execution
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

def main():
    print("🎯 Starting LSTM Retraining with Real Data")
    print("=" * 50)
    
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
    print("📊 Loading data...")
    sequences = np.load(sequences_path)
    labels = np.load(labels_path)
    
    print(f"✅ Loaded {len(sequences)} sequences")
    print(f"   Shape: {sequences.shape}")
    print(f"   Labels: {labels.shape}")
    
    # Prepare data
    print("🔧 Preparing data...")
    X = sequences
    y = to_categorical(labels, num_classes=3)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"✅ Data prepared:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Create model
    print("🏗️ Creating model...")
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
    print("🎯 Training model...")
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
    
    # Save model and scaler
    print("💾 Saving model...")
    model.save('models/mrben_lstm_real_data.h5')
    
    scaler = MinMaxScaler()
    n_samples, n_timesteps, n_features = sequences.shape
    sequences_reshaped = sequences.reshape(-1, n_features)
    scaler.fit(sequences_reshaped)
    joblib.dump(scaler, 'models/mrben_lstm_real_data_scaler.save')
    
    print("✅ Model and scaler saved!")
    print(f"   Model: models/mrben_lstm_real_data.h5")
    print(f"   Scaler: models/mrben_lstm_real_data_scaler.save")
    
    print(f"\n🎉 Retraining completed!")
    print(f"   Final Accuracy: {accuracy:.4f}")
    print(f"   Training Epochs: {len(history.history['loss'])}")

if __name__ == "__main__":
    main() 