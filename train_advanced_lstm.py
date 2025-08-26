"""
Advanced LSTM Training Script
Trains the advanced LSTM model with attention mechanism on real trading data
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our advanced models
from advanced_models import build_advanced_lstm, build_enhanced_lstm_classifier, get_callbacks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedLSTMTrainer:
    """
    Advanced LSTM Trainer for trading signals
    """
    
    def __init__(self, timesteps=50, features=None):
        self.timesteps = timesteps
        self.features = features or ['open', 'high', 'low', 'close', 'tick_volume']
        self.scaler = MinMaxScaler()
        self.model = None
        
    def load_and_prepare_data(self, data_file=None):
        """
        Load and prepare data for training
        """
        try:
            # Try to load data
            if data_file and os.path.exists(data_file):
                print(f"📊 Loading data from: {data_file}")
                data = pd.read_csv(data_file)
            else:
                # Try to find existing data files
                data_files = [
                    "XAUUSD_PRO_M15_history.csv",
                    "adausd_data.csv", 
                    "ohlc_data.csv",
                    "lstm_signals_features.csv"
                ]
                
                data = None
                for file in data_files:
                    if os.path.exists(file):
                        print(f"📊 Loading data from: {file}")
                        data = pd.read_csv(file)
                        break
                
                if data is None:
                    print("⚠️ No data found, generating synthetic data...")
                    data = self._generate_synthetic_data()
            
            print(f"📈 Data shape: {data.shape}")
            print(f"📋 Columns: {list(data.columns)}")
            
            # Prepare features - handle missing columns
            available_features = [f for f in self.features if f in data.columns]
            if len(available_features) < 4:  # Need at least 4 features
                print(f"Not enough features available. Found: {available_features}")
                return None, None
            feature_data = data[available_features].values
            
            # Scale features
            scaled_data = self.scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            print(f"✅ Prepared sequences: X shape: {X.shape}, y shape: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None
    
    def _generate_synthetic_data(self, n_samples=2000):
        """
        Generate synthetic trading data for testing
        """
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='5T')
        
        # Generate realistic price data with trends
        base_price = 2000.0
        prices = [base_price]
        for i in range(1, n_samples):
            # Add some trend and noise
            trend = np.sin(i * 0.01) * 10  # Cyclical trend
            noise = np.random.normal(0, 2)
            change = trend + noise
            new_price = prices[-1] + change
            prices.append(new_price)
        
        data = pd.DataFrame({
            'time': dates,
            'open': [p - np.random.uniform(0, 2) for p in prices],
            'high': [p + np.random.uniform(0, 3) for p in prices],
            'low': [p - np.random.uniform(0, 3) for p in prices],
            'close': prices,
            'tick_volume': np.random.randint(100, 1000, n_samples)
        })
        
        return data
    
    def _create_sequences(self, data):
        """
        Create sequences for LSTM training
        """
        X, y = [], []
        
        for i in range(self.timesteps, len(data)):
            X.append(data[i-self.timesteps:i])
            # For classification, create labels based on price movement
            current_price = data[i, 3]  # close price
            prev_price = data[i-1, 3]
            
            if current_price > prev_price * 1.001:  # 0.1% increase
                y.append(1)  # BUY
            elif current_price < prev_price * 0.999:  # 0.1% decrease
                y.append(2)  # SELL
            else:
                y.append(0)  # HOLD
        
        return np.array(X), np.array(y)
    
    def train_model(self, X, y, model_type='advanced', epochs=100, batch_size=32):
        """
        Train the LSTM model
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 Training data: {X_train.shape}")
            print(f"📊 Test data: {X_test.shape}")
            
            # Build model
            input_shape = (self.timesteps, len(self.features))
            
            if model_type == 'advanced':
                self.model = build_advanced_lstm(input_shape, num_classes=3)
            else:
                self.model = build_enhanced_lstm_classifier(input_shape, num_classes=3)
            
            if self.model is None:
                raise Exception("Failed to build model")
            
            print("🏗️ Model built successfully!")
            self.model.summary()
            
            # Get callbacks
            callbacks = get_callbacks(patience=15, min_lr=1e-7)
            
            # Train model
            print("🚀 Starting training...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"✅ Test Accuracy: {test_accuracy:.4f}")
            print(f"✅ Test Loss: {test_loss:.4f}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def save_model(self, model_path='models/advanced_lstm_model.h5', scaler_path='models/advanced_lstm_scaler.save'):
        """
        Save the trained model and scaler
        """
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save model
            if self.model:
                self.model.save(model_path)
                print(f"✅ Model saved to: {model_path}")
            
            # Save scaler
            joblib.dump(self.scaler, scaler_path)
            print(f"✅ Scaler saved to: {scaler_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def predict(self, X):
        """
        Make predictions with the trained model
        """
        if self.model is None:
            raise Exception("Model not trained yet")
        
        return self.model.predict(X)

def main():
    """
    Main training function
    """
    print("🚀 Advanced LSTM Training Script")
    print("=" * 50)
    
    # Initialize trainer
    trainer = AdvancedLSTMTrainer(timesteps=50)
    
    # Load and prepare data
    print("\n📊 Loading and preparing data...")
    X, y = trainer.load_and_prepare_data()
    
    if X is None or y is None:
        print("❌ Failed to load data")
        return False
    
    # Train model
    print("\n🏋️ Training Advanced LSTM Model...")
    history = trainer.train_model(X, y, model_type='advanced', epochs=50)
    
    if history is None:
        print("❌ Training failed")
        return False
    
    # Save model
    print("\n💾 Saving model...")
    success = trainer.save_model()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("✅ Model and scaler saved to models/ directory")
        
        # Test prediction
        print("\n🧪 Testing prediction...")
        test_X = X[:5]  # Test with first 5 sequences
        predictions = trainer.predict(test_X)
        print(f"✅ Predictions shape: {predictions.shape}")
        print(f"✅ Sample predictions: {predictions[0]}")
        
        return True
    else:
        print("❌ Failed to save model")
        return False

if __name__ == "__main__":
    main() 