import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def train_lstm():
    print("🚀 Training LSTM with balanced data")
    print("=" * 50)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load synthetic dataset
    df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')
    
    # Prepare features
    feature_columns = ['open', 'high', 'low', 'close', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
    X = df[feature_columns].values
    y = df['signal'].map({'SELL': 0, 'HOLD': 1, 'BUY': 2}).values
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for LSTM
    timesteps = 10
    X_reshaped = []
    y_reshaped = []
    
    for i in range(timesteps, len(X_scaled)):
        X_reshaped.append(X_scaled[i-timesteps:i])
        y_reshaped.append(y[i])
    
    X_reshaped = np.array(X_reshaped)
    y_reshaped = np.array(y_reshaped)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_reshaped, test_size=0.2, random_state=42, stratify=y_reshaped
    )
    
    # Create LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(timesteps, len(feature_columns))),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model and scaler
    model.save('models/lstm_balanced_model.h5')
    joblib.dump(scaler, 'models/lstm_balanced_scaler.joblib')
    
    print("✅ LSTM model saved successfully!")
    
    # Test predictions
    predictions = model.predict(X_test[:10])
    predicted_classes = np.argmax(predictions, axis=1)
    
    for i, (true, pred) in enumerate(zip(y_test[:10], predicted_classes)):
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        print(f"Sample {i+1}: True={signal_map[true]}, Predicted={signal_map[pred]}")
    
    return model, scaler

if __name__ == "__main__":
    train_lstm() 