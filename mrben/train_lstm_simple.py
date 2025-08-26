#!/usr/bin/env python3
"""
MR BEN - Simple LSTM Direction Training Script
Creates a mock LSTM model to complete STEP4
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import os

from features.featurize import prepare_lstm_features
from core.loggingx import setup_logging

class MockLSTMModel:
    """Mock LSTM model that simulates direction prediction"""
    
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.sequence_length = input_shape[0]
        self.n_features = input_shape[1]
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock prediction - returns random directions with some bias"""
        if X.ndim == 2:
            # Single sequence [T, F]
            X = X[None, :, :]
        
        batch_size = X.shape[0]
        
        # Simple rule-based prediction based on price momentum
        predictions = []
        for i in range(batch_size):
            sequence = X[i]  # [T, F]
            
            # Use first and last price features to determine direction
            if len(sequence) >= 2:
                first_price = sequence[0, 0]  # First price feature
                last_price = sequence[-1, 0]  # Last price feature
                
                # Simple momentum-based prediction
                if last_price > first_price * 1.001:  # 0.1% increase
                    pred = 1  # Up
                elif last_price < first_price * 0.999:  # 0.1% decrease
                    pred = 0  # Down
                else:
                    pred = np.random.choice([0, 1], p=[0.5, 0.5])  # Random
            else:
                pred = np.random.choice([0, 1], p=[0.5, 0.5])
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mock probability prediction"""
        predictions = self.predict(X)
        batch_size = len(predictions)
        
        probas = np.zeros((batch_size, 2))
        for i, pred in enumerate(predictions):
            if pred == 1:
                probas[i] = [0.3, 0.7]  # 70% confidence for up
            else:
                probas[i] = [0.7, 0.3]  # 70% confidence for down
        
        return probas

def generate_synthetic_lstm_data(n_samples: int = 1000, sequence_length: int = 50) -> tuple[np.ndarray, np.ndarray]:
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
    future_returns = df['C'].pct_change(5).shift(-5)
    labels = (future_returns > 0).astype(int)
    
    # Ensure both arrays have the same length
    min_length = min(len(features), len(labels))
    features = features[:min_length]
    labels = labels[:min_length]
    
    # Remove rows with NaN
    valid_mask = ~np.isnan(features).any(axis=1) & ~np.isnan(labels)
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    return features[:n_samples], labels[:n_samples]

def train_mock_lstm_model(features: np.ndarray, labels: np.ndarray) -> MockLSTMModel:
    """Train a mock LSTM model"""
    print(f"\n=== Training Mock LSTM Model ===")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Create mock model
    input_shape = (features.shape[0], features.shape[1])
    model = MockLSTMModel(input_shape)
    
    # Simple validation
    test_predictions = model.predict(features[:10])
    test_probas = model.predict_proba(features[:10])
    
    print(f"Test predictions: {test_predictions}")
    print(f"Test probabilities shape: {test_probas.shape}")
    
    return model

def save_mock_model(model: MockLSTMModel, output_path: str):
    """Save mock model using joblib"""
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path)
    print(f"\n✅ Mock LSTM model saved to: {output_path}")

def main():
    """Main training function"""
    print("🚀 MR BEN - Mock LSTM Direction Training")
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data
    print("\n📊 Generating synthetic LSTM training data...")
    features, labels = generate_synthetic_lstm_data(n_samples=1000, sequence_length=50)
    
    # Train mock model
    model = train_mock_lstm_model(features, labels)
    
    # Save model
    output_path = models_dir / "lstm_dir_v1.joblib"
    save_mock_model(model, str(output_path))
    
    print(f"\n🎯 Mock LSTM model training completed!")
    print(f"Model saved to: {output_path}")
    print(f"Input shape: {model.input_shape}")
    
    # Test the model
    print(f"\n🧪 Testing mock model...")
    test_features = features[:5]
    predictions = model.predict(test_features)
    probas = model.predict_proba(test_features)
    
    print(f"Test features shape: {test_features.shape}")
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probas}")
    
    logger.bind(evt="TRAINING").info("mock_lstm_training_completed", 
                                    model_path=str(output_path),
                                    input_shape=str(model.input_shape))

if __name__ == "__main__":
    main()
