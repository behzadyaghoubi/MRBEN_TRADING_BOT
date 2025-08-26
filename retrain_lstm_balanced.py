#!/usr/bin/env python3
"""
Retrain LSTM Model with Balanced Dataset
Create balanced dataset and retrain LSTM model to eliminate bias
"""

import json
import os
import sys
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_balanced_dataset():
    """Create a balanced dataset with BUY/SELL/HOLD signals."""
    print("🔧 Creating Balanced LSTM Dataset")
    print("=" * 50)
    
    # Parameters
    n_samples_per_class = 1000
    timesteps = 50
    features = 7
    
    print(f"📊 Creating {n_samples_per_class} samples per class")
    print(f"📊 Timesteps: {timesteps}, Features: {features}")
    
    # Generate BUY signals (class 2)
    buy_data = []
    for _ in range(n_samples_per_class):
        # Generate sequence with upward trend
        sequence = []
        base_price = np.random.uniform(3000, 3500)
        
        for t in range(timesteps):
            # Upward trend with some noise
            trend = 0.1 * t + np.random.normal(0, 0.5)
            price = base_price + trend
            
            # Generate OHLCV data
            open_price = price + np.random.normal(0, 0.1)
            high_price = open_price + abs(np.random.normal(0, 0.2))
            low_price = open_price - abs(np.random.normal(0, 0.2))
            close_price = price + np.random.normal(0, 0.1)
            volume = np.random.uniform(100, 1000)
            
            # Technical indicators for BUY
            rsi = np.random.uniform(50, 80)  # Neutral to overbought
            macd = np.random.uniform(0, 2)   # Positive MACD
            
            sequence.append([open_price, high_price, low_price, close_price, volume, rsi, macd])
        
        buy_data.append(sequence)
    
    # Generate SELL signals (class 0)
    sell_data = []
    for _ in range(n_samples_per_class):
        # Generate sequence with downward trend
        sequence = []
        base_price = np.random.uniform(3000, 3500)
        
        for t in range(timesteps):
            # Downward trend with some noise
            trend = -0.1 * t + np.random.normal(0, 0.5)
            price = base_price + trend
            
            # Generate OHLCV data
            open_price = price + np.random.normal(0, 0.1)
            high_price = open_price + abs(np.random.normal(0, 0.2))
            low_price = open_price - abs(np.random.normal(0, 0.2))
            close_price = price + np.random.normal(0, 0.1)
            volume = np.random.uniform(100, 1000)
            
            # Technical indicators for SELL
            rsi = np.random.uniform(20, 50)  # Oversold to neutral
            macd = np.random.uniform(-2, 0)  # Negative MACD
            
            sequence.append([open_price, high_price, low_price, close_price, volume, rsi, macd])
        
        sell_data.append(sequence)
    
    # Generate HOLD signals (class 1)
    hold_data = []
    for _ in range(n_samples_per_class):
        # Generate sequence with sideways movement
        sequence = []
        base_price = np.random.uniform(3000, 3500)
        
        for t in range(timesteps):
            # Sideways movement with noise
            trend = np.random.normal(0, 0.3)
            price = base_price + trend
            
            # Generate OHLCV data
            open_price = price + np.random.normal(0, 0.1)
            high_price = open_price + abs(np.random.normal(0, 0.2))
            low_price = open_price - abs(np.random.normal(0, 0.2))
            close_price = price + np.random.normal(0, 0.1)
            volume = np.random.uniform(100, 1000)
            
            # Technical indicators for HOLD
            rsi = np.random.uniform(40, 60)  # Neutral RSI
            macd = np.random.uniform(-0.5, 0.5)  # Neutral MACD
            
            sequence.append([open_price, high_price, low_price, close_price, volume, rsi, macd])
        
        hold_data.append(sequence)
    
    # Combine all data
    X = np.array(buy_data + sell_data + hold_data)
    y = np.array([2] * n_samples_per_class + [0] * n_samples_per_class + [1] * n_samples_per_class)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"✅ Dataset created successfully!")
    print(f"   Total samples: {len(X)}")
    print(f"   BUY signals: {np.sum(y == 2)}")
    print(f"   SELL signals: {np.sum(y == 0)}")
    print(f"   HOLD signals: {np.sum(y == 1)}")
    print(f"   Shape: {X.shape}")
    
        return X, y
    
def create_lstm_model(timesteps, features, classes):
    """Create LSTM model architecture."""
    print(f"\n🎯 Creating LSTM Model")
    print(f"📊 Input shape: ({timesteps}, {features})")
    print(f"📊 Output classes: {classes}")
        
        model = Sequential([
        LSTM(64, input_shape=(timesteps, features), return_sequences=False),
        Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
        Dense(classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    print("✅ Model created successfully!")
        model.summary()
        
        return model
    
def train_model(model, X_train, y_train, X_val, y_val):
        """Train the LSTM model."""
    print(f"\n🎯 Training LSTM Model")
    print("=" * 50)
        
        # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'models/mrben_lstm_balanced_v2_temp.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
    
    print("✅ Training completed!")
        
        return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    print(f"\n📊 Model Evaluation")
    print("=" * 50)
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📈 Test Accuracy: {accuracy:.3f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY']))
    
    # Confusion matrix
    print(f"\n🔍 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   SELL predicted as SELL: {cm[0,0]}")
    print(f"   SELL predicted as HOLD: {cm[0,1]}")
    print(f"   SELL predicted as BUY:  {cm[0,2]}")
    print(f"   HOLD predicted as SELL: {cm[1,0]}")
    print(f"   HOLD predicted as HOLD: {cm[1,1]}")
    print(f"   HOLD predicted as BUY:  {cm[1,2]}")
    print(f"   BUY predicted as SELL:  {cm[2,0]}")
    print(f"   BUY predicted as HOLD:  {cm[2,1]}")
    print(f"   BUY predicted as BUY:   {cm[2,2]}")
    
    return accuracy, y_pred

def save_model_and_scaler(model, scaler):
        """Save the trained model and scaler."""
    print(f"\n💾 Saving Model and Scaler")
    print("=" * 50)
        
        # Save model
    model_path = 'models/mrben_lstm_balanced_v2.h5'
    model.save(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Save scaler
    scaler_path = 'models/mrben_lstm_scaler_v2.save'
    joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved to: {scaler_path}")
        
        return model_path, scaler_path
    
def test_model_with_samples(model, scaler):
    """Test the model with sample inputs."""
    print(f"\n🧪 Testing Model with Sample Inputs")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("Bullish Pattern", "upward_trend"),
        ("Bearish Pattern", "downward_trend"),
        ("Sideways Pattern", "sideways_movement"),
        ("Random Pattern", "random_movement")
    ]
    
    for name, pattern in test_cases:
        # Generate test sequence
        if pattern == "upward_trend":
            sequence = np.random.rand(50, 7) * 0.5 + 0.5  # High values
        elif pattern == "downward_trend":
            sequence = np.random.rand(50, 7) * 0.5  # Low values
        elif pattern == "sideways_movement":
            sequence = np.random.rand(50, 7) * 0.3 + 0.35  # Medium values
        else:
            sequence = np.random.rand(50, 7)  # Random values
        
        # Scale sequence
        scaled_sequence = scaler.transform(sequence)
        input_sequence = scaled_sequence.reshape(1, 50, 7)
        
        # Make prediction
        prediction = model.predict(input_sequence, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Map class to signal
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        signal = signal_map[predicted_class]
        
        print(f"   {name}: {signal} (Class {predicted_class}), Confidence: {confidence:.3f}")
        print(f"      Raw predictions: SELL={prediction[0][0]:.3f}, HOLD={prediction[0][1]:.3f}, BUY={prediction[0][2]:.3f}")

def main():
    """Main function."""
    print("🎯 LSTM Model Retraining with Balanced Dataset")
    print("=" * 60)
    
    try:
        # Create balanced dataset
        X, y = create_balanced_dataset()
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        # Create and fit scaler
        print(f"\n🔧 Creating Scaler")
        scaler = MinMaxScaler()
        
        # Reshape data for scaler fitting
        X_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_reshaped)
        
        # Scale the data
        X_train_scaled = scaler.transform(X_reshaped).reshape(X_train.shape)
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        print(f"✅ Data scaled successfully!")
        
        # Create model
        model = create_lstm_model(timesteps=50, features=7, classes=3)
        
        # Train model
        history = train_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate model
        accuracy, y_pred = evaluate_model(model, X_test_scaled, y_test)
        
        # Save model and scaler
        model_path, scaler_path = save_model_and_scaler(model, scaler)
        
        # Test with samples
        test_model_with_samples(model, scaler)
        
        print(f"\n✅ LSTM Retraining completed successfully!")
        print(f"\n📝 Summary:")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Test Accuracy: {accuracy:.3f}")
        print(f"   Classes: SELL(0), HOLD(1), BUY(2)")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Update live_trader_clean.py to use new model")
        print(f"   2. Test the pipeline with balanced signals")
        print(f"   3. Monitor signal distribution")
        
    except Exception as e:
        print(f"❌ Error in LSTM retraining: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
