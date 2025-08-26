import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

def train_improved_lstm():
    print("🚀 Training Improved LSTM with Better Balance")
    print("=" * 60)
    
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
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_reshaped, test_size=0.2, random_state=42, stratify=y_reshaped
    )
    
    # Compute class weights for better balance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    
    # Create improved LSTM model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(timesteps, len(feature_columns))),
        Dropout(0.3),
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile model with better settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train model with more epochs and callbacks
    print("Training improved model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,  # More epochs
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,  # Use class weights
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save model and scaler
    model.save('models/lstm_balanced_model.h5')
    joblib.dump(scaler, 'models/lstm_balanced_scaler.joblib')
    
    print("✅ Improved LSTM model saved successfully!")
    
    # Test predictions for balance
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Analyze distribution
    unique, counts = np.unique(predicted_classes, return_counts=True)
    total = len(predicted_classes)
    
    signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    
    print("\n📊 Improved Model Predictions:")
    for class_id, count in zip(unique, counts):
        percentage = (count / total) * 100
        signal_name = signal_map[class_id]
        print(f"{signal_name}: {percentage:.1f}% ({count})")
    
    # Check for bias
    if len(unique) >= 2:
        buy_count = counts[unique == 2][0] if 2 in unique else 0
        sell_count = counts[unique == 0][0] if 0 in unique else 0
        
        if buy_count > 0 and sell_count > 0:
            ratio = buy_count / sell_count
            print(f"BUY/SELL ratio: {ratio:.2f}")
            
            if 0.7 <= ratio <= 1.3:
                print("✅ Balanced predictions achieved!")
            else:
                print("⚠️ Still some bias, but improved")
    
    return model, scaler

if __name__ == "__main__":
    train_improved_lstm() 