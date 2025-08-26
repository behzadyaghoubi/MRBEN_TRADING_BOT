#!/usr/bin/env python3
"""
MR BEN LSTM Model Retraining Script
Uses latest trading logs and market data to retrain the LSTM model
"""

import logging
import os
import re
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LSTMRetrainer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = 50
        self.feature_columns = [
            'open',
            'high',
            'low',
            'close',
            'volume',
            'rsi',
            'macd',
            'macd_signal',
            'sma_20',
            'sma_50',
        ]

    def extract_data_from_logs(self, log_file_path):
        """Extract trading data from log files"""
        logger.info("Extracting data from trading logs...")

        data_points = []

        try:
            with open(log_file_path, encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                # Extract ML Filter Features
                if 'ML Filter Features:' in line:
                    try:
                        # Parse features from log line
                        features_match = re.search(r'ML Filter Features: \[(.*?)\]', line)
                        if features_match:
                            features_str = features_match.group(1)
                            features = []
                            for item in features_str.split(', '):
                                if 'np.float64(' in item:
                                    value = float(item.split('(')[1].split(')')[0])
                                    features.append(value)

                            if len(features) >= 10:
                                data_point = {
                                    'open': features[0],
                                    'high': features[1],
                                    'low': features[2],
                                    'close': features[3],
                                    'sma_20': features[4],
                                    'sma_50': features[5],
                                    'rsi': features[6],
                                    'macd': features[7],
                                    'macd_signal': features[8],
                                    'macd_hist': features[9],
                                    'volume': 1000,  # Default volume
                                }
                                data_points.append(data_point)
                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Error reading log file: {e}")

        logger.info(f"Extracted {len(data_points)} data points from logs")
        return pd.DataFrame(data_points)

    def get_mt5_data(self, symbol="XAUUSD.PRO", timeframe=mt5.TIMEFRAME_M5, bars=1000):
        """Get fresh market data from MT5"""
        logger.info(f"Fetching {bars} bars of {symbol} data from MT5...")

        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return None

        try:
            # Get price data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None:
                logger.error("Failed to get MT5 data")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Calculate technical indicators
            df = self.calculate_indicators(df)

            logger.info(f"Retrieved {len(df)} bars of market data")
            return df

        except Exception as e:
            logger.error(f"Error getting MT5 data: {e}")
            return None
        finally:
            mt5.shutdown()

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Volume (use tick volume as proxy)
        df['volume'] = df['tick_volume']

        return df

    def prepare_sequences(self, df):
        """Prepare LSTM sequences"""
        logger.info("Preparing LSTM sequences...")

        # Select features
        features = df[self.feature_columns].values

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Create sequences
        X, y = [], []

        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i - self.sequence_length : i])

            # Create target based on price movement
            current_price = features[i, 3]  # close price
            future_price = features[i - 1, 3] if i > 0 else current_price

            # Simple target: 0=down, 1=hold, 2=up
            if current_price > future_price * 1.001:  # 0.1% increase
                y.append(2)  # UP
            elif current_price < future_price * 0.999:  # 0.1% decrease
                y.append(0)  # DOWN
            else:
                y.append(1)  # HOLD

        X = np.array(X)
        y = np.array(y)

        # Convert to categorical
        y_categorical = tf.keras.utils.to_categorical(y, num_classes=3)

        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y_categorical

    def build_model(self, input_shape):
        """Build LSTM model"""
        logger.info("Building LSTM model...")

        model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax'),  # 3 classes: DOWN, HOLD, UP
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        logger.info("Model built successfully")
        return model

    def train_model(self, X, y):
        """Train the LSTM model"""
        logger.info("Starting model training...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
        ]

        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate model
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation Loss: {val_loss:.4f}")

        return history

    def save_model(self, model_path="models/mrben_lstm_real_data.h5"):
        """Save the trained model"""
        if not os.path.exists("models"):
            os.makedirs("models")

        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save scaler
        import joblib

        scaler_path = "models/mrben_lstm_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    def retrain_complete(self):
        """Complete retraining process"""
        logger.info("Starting complete LSTM retraining process...")

        # Get data from multiple sources
        data_sources = []

        # 1. Extract from logs
        if os.path.exists("lstm_trading_system.log"):
            log_data = self.extract_data_from_logs("lstm_trading_system.log")
            if not log_data.empty:
                data_sources.append(log_data)
                logger.info(f"Added {len(log_data)} data points from logs")

        # 2. Get fresh MT5 data
        mt5_data = self.get_mt5_data()
        if mt5_data is not None:
            data_sources.append(mt5_data)
            logger.info(f"Added {len(mt5_data)} data points from MT5")

        # Combine all data
        if not data_sources:
            logger.error("No data sources available")
            return False

        combined_df = pd.concat(data_sources, ignore_index=True)
        combined_df = combined_df.drop_duplicates()
        combined_df = combined_df.sort_values(
            'time' if 'time' in combined_df.columns else combined_df.index
        )

        logger.info(f"Combined dataset: {len(combined_df)} data points")

        # Prepare sequences
        X, y = self.prepare_sequences(combined_df)

        if len(X) < 100:
            logger.error("Insufficient data for training")
            return False

        # Train model
        history = self.train_model(X, y)

        # Save model
        self.save_model()

        # Generate training report
        self.generate_report(history, X, y)

        logger.info("LSTM retraining completed successfully!")
        return True

    def generate_report(self, history, X, y):
        """Generate training report"""
        logger.info("Generating training report...")

        # Calculate predictions
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)

        # Calculate metrics
        from sklearn.metrics import classification_report

        report = classification_report(
            true_classes, predicted_classes, target_names=['DOWN', 'HOLD', 'UP']
        )

        logger.info("Training Report:")
        logger.info("\n" + report)

        # Save report
        with open("models/lstm_training_report.txt", "w") as f:
            f.write("MR BEN LSTM Model Training Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Training Date: {datetime.now()}\n")
            f.write(f"Dataset Size: {len(X)} sequences\n")
            f.write("Model Architecture: LSTM(128) -> LSTM(64) -> Dense(32) -> Dense(3)\n")
            f.write("\nClassification Report:\n")
            f.write(report)


def main():
    """Main function"""
    logger.info("Starting MR BEN LSTM Model Retraining...")

    retrainer = LSTMRetrainer()
    success = retrainer.retrain_complete()

    if success:
        logger.info("✅ LSTM retraining completed successfully!")
        logger.info("📁 Model saved to: models/mrben_lstm_real_data.h5")
        logger.info("📁 Scaler saved to: models/mrben_lstm_scaler.pkl")
        logger.info("📄 Report saved to: models/lstm_training_report.txt")
    else:
        logger.error("❌ LSTM retraining failed!")


if __name__ == "__main__":
    main()
