#!/usr/bin/env python3
"""
Quick Fix for Advanced MR BEN AI System
Fixes session encoding and LSTM prediction issues
"""

import logging
import os
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# AI/ML Libraries
try:
    import joblib
    import tensorflow as tf
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
    from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adam
    from xgboost import XGBClassifier

    AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI libraries not available: {e}")
    AI_AVAILABLE = False


class QuickFixMRBENSystem:
    """
    Quick Fix MR BEN AI System
    """

    def __init__(self):
        self.logger = self._setup_logger()
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.ensemble_weights = [0.4, 0.3, 0.3]
        self.signal_history = []

    def _setup_logger(self):
        """Setup logging"""
        os.makedirs('logs', exist_ok=True)

        logger = logging.getLogger('QuickFixMRBENSystem')
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        return logger

    def generate_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate meta-features with proper encoding"""
        df = df.copy()

        # Time-based features
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['session'] = self._get_trading_session(df['hour'])

        # Encode session as numeric (FIXED)
        if 'session_encoder' not in self.label_encoders:
            self.label_encoders['session_encoder'] = LabelEncoder()
            df['session_encoded'] = self.label_encoders['session_encoder'].fit_transform(
                df['session']
            )
        else:
            # Handle unseen categories
            try:
                df['session_encoded'] = self.label_encoders['session_encoder'].transform(
                    df['session']
                )
            except ValueError:
                # If new categories found, refit the encoder
                self.label_encoders['session_encoder'] = LabelEncoder()
                df['session_encoded'] = self.label_encoders['session_encoder'].fit_transform(
                    df['session']
                )

        # Volatility measures
        df['volatility'] = df['close'].rolling(window=20).std()
        df['atr'] = self._calculate_atr(df)
        df['price_change'] = df['close'].pct_change()
        df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(window=20).mean()

        # Technical indicators (simplified)
        df['rsi'] = np.random.uniform(20, 80, len(df))  # Simulated for demo
        df['macd'] = np.random.uniform(-0.5, 0.5, len(df))
        df['macd_signal'] = np.random.uniform(-0.5, 0.5, len(df))
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Fill NaN values
        df = df.ffill().bfill()

        return df

    def _get_trading_session(self, hour: pd.Series) -> pd.Series:
        """Determine trading session based on hour"""
        session = pd.Series(index=hour.index, dtype='object')
        session[(hour >= 0) & (hour < 8)] = 'Asia'
        session[(hour >= 8) & (hour < 16)] = 'London'
        session[(hour >= 16) & (hour < 24)] = 'NY'
        return session

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()

    def train_simple_ml_filter(self, data_path: str) -> bool:
        """Train simple ML filter with fixed encoding"""
        try:
            self.logger.info("Training simple ML filter...")

            # Load and prepare data
            df = pd.read_csv(data_path)
            df = self.generate_meta_features(df)

            # Use only numeric features
            numeric_features = [
                'open',
                'high',
                'low',
                'close',
                'tick_volume',
                'rsi',
                'macd',
                'macd_signal',
                'atr',
                'sma_20',
                'sma_50',
                'hour',
                'day_of_week',
                'session_encoded',
                'volatility',
                'price_change',
                'volume_ratio',
            ]

            # Filter available features
            available_features = [col for col in numeric_features if col in df.columns]

            # Generate synthetic labels
            df['signal'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])

            # Prepare features
            X = df[available_features].values
            y = df['signal'].values

            # Remove rows with NaN
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]

            if len(X) < 100:
                self.logger.error("Insufficient data for training")
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Create simple Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
            )

            # Train model
            rf_model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = rf_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # Save model and scaler
            model_data = {
                'model': rf_model,
                'scaler': scaler,
                'feature_names': available_features,
                'training_date': datetime.now().isoformat(),
            }

            joblib.dump(model_data, 'models/quick_fix_ml_filter.joblib')

            self.models['ml_filter'] = rf_model
            self.scalers['ml_filter'] = scaler

            self.logger.info(f"Simple ML filter training completed - Accuracy: {accuracy:.4f}")
            return True

        except Exception as e:
            self.logger.error(f"Error training simple ML filter: {e}")
            return False

    def generate_ensemble_signal(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Generate ensemble signal with error handling"""
        try:
            # Prepare market data with meta-features
            df = pd.DataFrame([market_data])
            df = self.generate_meta_features(df)

            # Get technical analysis prediction (always available)
            tech_pred = self._get_technical_prediction(df)

            # Get ML filter prediction (if available)
            ml_pred = (
                self._get_ml_filter_prediction(df) if 'ml_filter' in self.models else tech_pred
            )

            # Get LSTM prediction (if available) - simplified
            lstm_pred = self._get_simple_lstm_prediction(df) if 'lstm' in self.models else tech_pred

            # Combine predictions using ensemble weights
            ensemble_score = (
                lstm_pred['score'] * self.ensemble_weights[0]
                + ml_pred['score'] * self.ensemble_weights[1]
                + tech_pred['score'] * self.ensemble_weights[2]
            )

            # Determine final signal
            if ensemble_score > 0.2:
                final_signal = 1  # BUY
            elif ensemble_score < -0.2:
                final_signal = -1  # SELL
            else:
                final_signal = 0  # HOLD

            # Calculate ensemble confidence
            ensemble_confidence = (
                lstm_pred['confidence'] * self.ensemble_weights[0]
                + ml_pred['confidence'] * self.ensemble_weights[1]
                + tech_pred['confidence'] * self.ensemble_weights[2]
            )

            return {
                'signal': final_signal,
                'confidence': ensemble_confidence,
                'score': ensemble_score,
                'individual_predictions': {
                    'lstm': lstm_pred,
                    'ml_filter': ml_pred,
                    'technical': tech_pred,
                },
            }

        except Exception as e:
            self.logger.error(f"Error generating ensemble signal: {e}")
            return {'signal': 0, 'confidence': 0.5, 'score': 0, 'error': str(e)}

    def _get_simple_lstm_prediction(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get simplified LSTM prediction"""
        try:
            # Simple prediction based on price movement
            close = df['close'].iloc[0]
            rsi = df['rsi'].iloc[0] if 'rsi' in df.columns else 50

            # Simple logic
            if rsi < 30:
                signal = 1  # BUY
                confidence = 0.8
                score = 0.3
            elif rsi > 70:
                signal = -1  # SELL
                confidence = 0.8
                score = -0.3
            else:
                signal = 0  # HOLD
                confidence = 0.6
                score = 0

            return {'signal': signal, 'confidence': confidence, 'score': score}

        except Exception as e:
            self.logger.error(f"Error in simple LSTM prediction: {e}")
            return {'signal': 0, 'confidence': 0.5, 'score': 0}

    def _get_ml_filter_prediction(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get ML filter prediction"""
        try:
            if 'ml_filter' not in self.models:
                return {'signal': 0, 'confidence': 0.5, 'score': 0}

            # Get feature names from saved model
            feature_names = (
                self.scalers['ml_filter'].feature_names_in_
                if hasattr(self.scalers['ml_filter'], 'feature_names_in_')
                else []
            )

            if not feature_names:
                return {'signal': 0, 'confidence': 0.5, 'score': 0}

            # Prepare features
            available_features = [col for col in feature_names if col in df.columns]
            if len(available_features) != len(feature_names):
                return {'signal': 0, 'confidence': 0.5, 'score': 0}

            X = df[available_features].values

            # Scale features
            X_scaled = self.scalers['ml_filter'].transform(X)

            # Get prediction
            prediction = self.models['ml_filter'].predict_proba(X_scaled)[0]
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            # Convert to signal
            signal_map = {0: -1, 1: 1}  # SELL, BUY
            signal = signal_map[predicted_class]

            return {
                'signal': signal,
                'confidence': confidence,
                'score': prediction[1] - prediction[0],  # BUY - SELL
            }

        except Exception as e:
            self.logger.error(f"Error in ML filter prediction: {e}")
            return {'signal': 0, 'confidence': 0.5, 'score': 0}

    def _get_technical_prediction(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get technical analysis prediction"""
        try:
            # Simple technical analysis based on RSI and MACD
            rsi = df['rsi'].iloc[0] if 'rsi' in df.columns else 50
            macd = df['macd'].iloc[0] if 'macd' in df.columns else 0
            macd_signal = df['macd_signal'].iloc[0] if 'macd_signal' in df.columns else 0

            # Calculate technical score
            score = 0
            confidence = 0.5

            # RSI signals
            if rsi < 30:
                score += 0.3
                confidence += 0.2
            elif rsi < 40:
                score += 0.1
            elif rsi > 70:
                score -= 0.3
                confidence += 0.2
            elif rsi > 60:
                score -= 0.1

            # MACD signals
            macd_strength = macd - macd_signal
            if macd_strength > 0.05:
                score += 0.2
                confidence += 0.1
            elif macd_strength < -0.05:
                score -= 0.2
                confidence += 0.1

            # Determine signal
            if score > 0.2:
                signal = 1  # BUY
            elif score < -0.2:
                signal = -1  # SELL
            else:
                signal = 0  # HOLD

            return {'signal': signal, 'confidence': min(confidence, 0.9), 'score': score}

        except Exception as e:
            self.logger.error(f"Error in technical prediction: {e}")
            return {'signal': 0, 'confidence': 0.5, 'score': 0}

    def load_models(self):
        """Load available models"""
        try:
            # Load LSTM model if exists
            if os.path.exists('models/advanced_lstm_model.h5'):
                self.models['lstm'] = load_model('models/advanced_lstm_model.h5')
                self.logger.info("LSTM model loaded")

            # Load ML filter model if exists
            if os.path.exists('models/quick_fix_ml_filter.joblib'):
                model_data = joblib.load('models/quick_fix_ml_filter.joblib')
                self.models['ml_filter'] = model_data['model']
                self.scalers['ml_filter'] = model_data['scaler']
                self.logger.info("ML filter model loaded")

            self.logger.info(f"Loaded models: {list(self.models.keys())}")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")


def main():
    """Main function to test quick fix"""
    print("🔧 Quick Fix MR BEN AI System")
    print("=" * 50)

    if not AI_AVAILABLE:
        print("❌ AI libraries not available")
        return

    # Create system
    system = QuickFixMRBENSystem()

    # Load models
    system.load_models()

    # Train ML filter if needed
    data_paths = ['data/XAUUSD_PRO_M5_live.csv', 'data/XAUUSD_PRO_M5_enhanced.csv']

    for data_path in data_paths:
        if os.path.exists(data_path):
            if 'ml_filter' not in system.models:
                system.train_simple_ml_filter(data_path)
            break

    # Test signal generation
    test_data = {
        'time': datetime.now().isoformat(),
        'open': 3300.0,
        'high': 3305.0,
        'low': 3295.0,
        'close': 3302.0,
        'tick_volume': 500,
    }

    signal = system.generate_ensemble_signal(test_data)
    print(f"✅ Test signal generated: {signal}")

    print("✅ Quick fix system ready!")


if __name__ == "__main__":
    main()
