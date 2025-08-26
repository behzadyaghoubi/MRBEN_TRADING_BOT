#!/usr/bin/env python3
"""
MR BEN - LSTM Direction Training Script
Trains an LSTM model to predict trading direction
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import onnxmltools
from onnxmltools.utils import save_model
from pathlib import Path

from features.featurize import prepare_lstm_features
from core.loggingx import setup_logging

def generate_synthetic_lstm_data(n_samples: int = 10000, sequence_length: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for LSTM model"""
    np.random.seed(42)
    
    # Generate random OHLCV data with sequences
    base_price = 100.0
    all_prices = []
    all_volumes = []
    
    for i in range(n_samples + sequence_length):
        # Random price movement with some trend
        trend = np.sin(i * 0.1) * 0.01  # Cyclical trend
        noise = np.random.normal(0, 0.015)  # 1.5% daily volatility
        change = trend + noise
        
        base_price *= (1 + change)
        
        # Generate OHLC from base price
        high = base_price * (1 + abs(np.random.normal(0, 0.008)))
        low = base_price * (1 - abs(np.random.normal(0, 0.008)))
        open_price = base_price * (1 + np.random.normal(0, 0.004))
        close = base_price
        
        all_prices.append([open_price, high, low, close])
        all_volumes.append(np.random.randint(1000, 10000))
    
    # Create DataFrame
    df = pd.DataFrame(all_prices, columns=['O', 'H', 'L', 'C'])
    df['V'] = all_volumes
    
    # Prepare LSTM features
    features = prepare_lstm_features(df, sequence_length)
    
    # Generate labels (1 for up, 0 for down)
    # Use future price movement as label
    future_returns = df['C'].pct_change(5).shift(-5)  # 5-period future return
    labels = (future_returns > 0).astype(int)
    
    # Ensure both arrays have the same length
    min_length = min(len(features), len(labels))
    features = features[:min_length]
    labels = labels[:min_length]
    
    # Remove rows with NaN
    valid_mask = ~np.isnan(features).any(axis=1) & ~np.isnan(labels)
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    # Ensure we have enough data
    if len(features) < n_samples:
        n_samples = len(features)
    
    return features[:n_samples], labels[:n_samples]

def create_lstm_model(input_shape: tuple, num_classes: int = 2) -> keras.Model:
    """Create LSTM model architecture"""
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # LSTM layers
        layers.LSTM(128, return_sequences=True, dropout=0.2),
        layers.LSTM(64, return_sequences=False, dropout=0.2),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_lstm_model(features: np.ndarray, labels: np.ndarray,
                    test_size: float = 0.2, random_state: int = 42,
                    epochs: int = 50, batch_size: int = 32) -> keras.Model:
    """Train the LSTM model"""
    
    # Split data - ensure we maintain the sequence structure
    # For LSTM, we want to split the sequences, not individual time steps
    n_sequences = len(features)
    indices = np.arange(n_sequences)
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    
    X_train = features[train_indices]
    X_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {len(features)}")
    print(f"Sequence length: {features.shape[0]}")
    print(f"Features per timestep: {features.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Positive labels: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    print(f"Negative labels: {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
    
    # Create model - LSTM expects [T, F] input shape where T=time_steps, F=features
    input_shape = (features.shape[0], features.shape[1])
    model = create_lstm_model(input_shape=input_shape)
    
    print(f"\n=== Model Architecture ===")
    model.summary()
    
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    print(f"\n=== Training Model ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print(f"\n=== Model Performance ===")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    print(f"\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    
    return model

def export_lstm_to_onnx(model: keras.Model, output_path: str):
    """Export TensorFlow LSTM model to ONNX format"""
    
    try:
        # Convert TensorFlow model to ONNX
        onx = onnxmltools.convert_keras(model)
        
        # Save model
        save_model(onx, output_path)
        print(f"\n✅ LSTM model exported to ONNX: {output_path}")
        
    except Exception as e:
        print(f"❌ Failed to export LSTM to ONNX: {e}")
        print("This is expected for complex LSTM models. Using TensorFlow format instead.")
        raise

def main():
    """Main training function"""
    logger = setup_logging("INFO")
    logger.info("Starting LSTM Direction training...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data
    logger.info("Generating synthetic LSTM training data...")
    features, labels = generate_synthetic_lstm_data(n_samples=12000, sequence_length=50)
    
    # Train model
    logger.info("Training LSTM model...")
    model = train_lstm_model(features, labels, epochs=30)
    
    # Export to ONNX
    output_path = models_dir / "lstm_dir_v1.onnx"
    logger.info(f"Exporting LSTM model to ONNX: {output_path}")
    
    try:
        export_lstm_to_onnx(model, str(output_path))
        logger.info("LSTM model exported to ONNX successfully!")
        
        # Test the exported model
        logger.info("Testing exported ONNX model...")
        import onnxruntime as rt
        sess = rt.InferenceSession(str(output_path))
        input_name = sess.get_inputs()[0].name
        
        # Test with a single sample
        test_input = features[:1].astype(np.float32)
        prediction = sess.run(None, {input_name: test_input})
        
        print(f"\n✅ ONNX LSTM model test successful!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {prediction[0].shape}")
        
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
        logger.info("Saving model in TensorFlow format instead...")
        
        # Save as TensorFlow model
        tf_path = models_dir / "lstm_dir_v1"
        model.save(tf_path)
        logger.info(f"LSTM model saved as TensorFlow model: {tf_path}")
        
        # Update config to use TensorFlow model
        logger.info("Note: Update config to use TensorFlow model path: models/lstm_dir_v1")
    
    logger.info("LSTM Direction training completed!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
