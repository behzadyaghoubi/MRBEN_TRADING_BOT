#!/usr/bin/env python3
"""
Fix LSTM Model Bias
Analyze and potentially fix the LSTM model that always predicts BUY
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def create_balanced_lstm_model():
    """Create a new LSTM model with better architecture to avoid bias"""
    
    print("🏗️ Creating New Balanced LSTM Model...")
    
    model = Sequential([
        # First LSTM layer with more units
        LSTM(128, return_sequences=True, input_shape=(10, 10), 
             activation='tanh', recurrent_dropout=0.2),
        Dropout(0.3),
        
        # Second LSTM layer
        LSTM(64, return_sequences=True, activation='tanh', recurrent_dropout=0.2),
        Dropout(0.3),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False, activation='tanh', recurrent_dropout=0.2),
        Dropout(0.3),
        
        # Dense layers with better initialization
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),
        
        Dense(32, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),
        
        # Output layer with softmax for balanced probabilities
        Dense(3, activation='softmax', kernel_initializer='glorot_normal')
    ])
    
    # Compile with better learning rate and loss function
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model created successfully!")
    return model

def prepare_balanced_data():
    """Prepare balanced dataset with proper labels"""
    
    print("📊 Preparing Balanced Dataset...")
    
    # Load the balanced dataset
    df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')
    
    # Convert signal labels to numeric
    signal_map = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
    df['signal_numeric'] = df['signal'].map(signal_map)
    
    # Check distribution
    signal_counts = df['signal'].value_counts()
    print(f"Dataset distribution:")
    for signal, count in signal_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {signal}: {count} ({percentage:.1f}%)")
    
    # Prepare features
    feature_columns = ['open', 'high', 'low', 'close', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
    X = df[feature_columns].values
    y = df['signal_numeric'].values
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences for LSTM
    timesteps = 10
    X_sequences = []
    y_sequences = []
    
    for i in range(timesteps, len(X_scaled)):
        X_sequences.append(X_scaled[i-timesteps:i])
        y_sequences.append(y[i])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"Sequences created: {X_sequences.shape}")
    print(f"Labels shape: {y_sequences.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

def train_balanced_model():
    """Train the model with balanced data and better parameters"""
    
    print("🚀 Starting Balanced Model Training...")
    
    # Create model
    model = create_balanced_lstm_model()
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_balanced_data()
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train with class weights to ensure balance
    class_weights = {
        0: 1.0,  # SELL
        1: 1.0,  # HOLD  
        2: 1.0   # BUY
    }
    
    print("🎯 Training with balanced class weights...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 Model Evaluation:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Test predictions distribution
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    unique, counts = np.unique(predicted_classes, return_counts=True)
    signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    
    print(f"\n📈 Prediction Distribution:")
    for class_id, count in zip(unique, counts):
        percentage = (count / len(predictions)) * 100
        signal_name = signal_map[class_id]
        print(f"  {signal_name}: {percentage:.1f}% ({count})")
    
    # Save model and scaler
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model.save('models/lstm_balanced_fixed.h5')
    joblib.dump(scaler, 'models/lstm_balanced_fixed_scaler.joblib')
    
    print("\n✅ Model and scaler saved successfully!")
    
    return model, scaler

def test_fixed_model():
    """Test the fixed model to ensure balanced predictions"""
    
    print("\n🧪 Testing Fixed Model...")
    
    try:
        # Load the fixed model
        model = load_model('models/lstm_balanced_fixed.h5')
        scaler = joblib.load('models/lstm_balanced_fixed_scaler.joblib')
        
        # Load test data
        df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')
        
        # Prepare features
        feature_columns = ['open', 'high', 'low', 'close', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
        X = df[feature_columns].values
        X_scaled = scaler.transform(X)
        
        # Create sequences
        timesteps = 10
        X_sequences = []
        
        for i in range(timesteps, len(X_scaled)):
            X_sequences.append(X_scaled[i-timesteps:i])
        
        X_sequences = np.array(X_sequences)
        
        # Get predictions on 100 samples
        test_samples = X_sequences[:100]
        predictions = model.predict(test_samples)
        
        # Analyze distribution
        predicted_classes = np.argmax(predictions, axis=1)
        unique, counts = np.unique(predicted_classes, return_counts=True)
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        
        print(f"📊 Fixed Model Distribution:")
        for class_id, count in zip(unique, counts):
            percentage = (count / len(predictions)) * 100
            signal_name = signal_map[class_id]
            print(f"  {signal_name}: {percentage:.1f}% ({count})")
        
        # Check balance
        if len(unique) >= 2:
            buy_count = counts[unique == 2][0] if 2 in unique else 0
            sell_count = counts[unique == 0][0] if 0 in unique else 0
            
            if buy_count > 0 and sell_count > 0:
                ratio = buy_count / sell_count
                print(f"\nBUY/SELL ratio: {ratio:.2f}")
                
                if 0.5 <= ratio <= 2.0:
                    print("✅ Balanced distribution achieved!")
                    return True
                else:
                    print("⚠️ Still some imbalance")
                    return False
            else:
                print("❌ Missing BUY or SELL signals")
                return False
        else:
            print("❌ Only one class predicted")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting LSTM Bias Fix Process...")
    print("=" * 60)
    
    # Train new balanced model
    model, scaler = train_balanced_model()
    
    # Test the fixed model
    success = test_fixed_model()
    
    if success:
        print("\n🎉 LSTM bias fixed successfully!")
        print("System is ready for live trading!")
    else:
        print("\n⚠️ Model still needs attention")
        print("Consider additional training or architecture changes") 