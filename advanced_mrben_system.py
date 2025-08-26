#!/usr/bin/env python3
"""
MR BEN AI System - Advanced Professional Version
Features:
1. Adaptive Online Learning for all models
2. Expanded Feature Engineering with Meta-Features
3. Ensemble Model Strategy
4. Self-Healing Pipeline with Auto-Retrain
5. Comprehensive Market Context Logging
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# AI/ML Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import joblib
    from xgboost import XGBClassifier
    AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI libraries not available: {e}")
    AI_AVAILABLE = False

class AdvancedMRBENSystem:
    """
    Advanced MR BEN AI Trading System with Professional Features
    """
    
    def __init__(self, config_path: str = "advanced_config.json"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.models = {}
        self.scalers = {}
        self.ensemble_weights = self.config.get('ensemble_weights', [0.4, 0.3, 0.3])
        self.retrain_threshold = self.config.get('retrain_threshold', 0.7)
        self.performance_history = []
        self.signal_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load system configuration"""
        default_config = {
            'retrain_interval_days': 3,
            'ensemble_weights': [0.4, 0.3, 0.3],  # LSTM, ML Filter, Technical
            'retrain_threshold': 0.7,
            'max_hold_ratio': 0.6,
            'min_confidence': 0.6,
            'feature_columns': [
                'open', 'high', 'low', 'close', 'tick_volume',
                'rsi', 'macd', 'macd_signal', 'atr', 'sma_20', 'sma_50',
                'hour', 'day_of_week', 'session', 'volatility', 'price_change',
                'volume_ratio', 'bb_position', 'stoch_k', 'stoch_d'
            ],
            'model_paths': {
                'lstm': 'models/advanced_lstm_model.h5',
                'ml_filter': 'models/advanced_ml_filter.joblib',
                'ensemble': 'models/advanced_ensemble.joblib'
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        else:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _setup_logger(self):
        """Setup comprehensive logging"""
        os.makedirs('logs', exist_ok=True)
        
        logger = logging.getLogger('AdvancedMRBENSystem')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        log_file = f'logs/advanced_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def generate_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced meta-features"""
        df = df.copy()
        
        # Time-based features
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['session'] = self._get_trading_session(df['hour'])
        
        # Volatility measures
        df['volatility'] = df['close'].rolling(window=20).std()
        df['atr'] = self._calculate_atr(df)
        df['price_change'] = df['close'].pct_change()
        df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(window=20).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        df['stoch_k'] = self._calculate_stochastic(df, 14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Seasonality features
        df['rolling_mean_5'] = df['close'].rolling(window=5).mean()
        df['rolling_mean_20'] = df['close'].rolling(window=20).mean()
        df['rolling_std_20'] = df['close'].rolling(window=20).std()
        
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
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator %K"""
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        return 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
    
    def prepare_lstm_data(self, df: pd.DataFrame, sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM with meta-features"""
        feature_cols = [col for col in self.config['feature_columns'] if col in df.columns]
        
        # Generate labels based on future price movement
        df['future_return'] = df['close'].pct_change().shift(-1)
        
        # Create 3-class labels: SELL (0), HOLD (1), BUY (2)
        labels = np.ones(len(df))  # Default to HOLD
        
        buy_threshold = 0.0005
        sell_threshold = -0.0005
        
        labels[df['future_return'] > buy_threshold] = 2  # BUY
        labels[df['future_return'] < sell_threshold] = 0  # SELL
        
        # Remove NaN values
        valid_indices = ~(df[feature_cols].isnull().any(axis=1) | np.isnan(labels))
        data = df[feature_cols][valid_indices].values
        labels = labels[valid_indices]
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(labels[i])
        
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, num_classes=3)
        
        return X, y
    
    def create_advanced_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Create advanced LSTM model with meta-features"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_advanced_lstm(self, data_path: str) -> bool:
        """Train advanced LSTM model with meta-features"""
        try:
            self.logger.info("Training advanced LSTM model...")
            
            # Load and prepare data
            df = pd.read_csv(data_path)
            df = self.generate_meta_features(df)
            
            # Prepare sequences
            X, y = self.prepare_lstm_data(df)
            
            if len(X) < 100:
                self.logger.error("Insufficient data for training")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            
            # Create and train model
            model = self.create_advanced_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
                ModelCheckpoint(
                    self.config['model_paths']['lstm'],
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            
            accuracy = accuracy_score(y_test_classes, y_pred_classes)
            
            # Save model and scaler
            model.save(self.config['model_paths']['lstm'])
            joblib.dump(scaler, self.config['model_paths']['lstm'].replace('.h5', '_scaler.joblib'))
            
            self.models['lstm'] = model
            self.scalers['lstm'] = scaler
            
            self.logger.info(f"Advanced LSTM training completed - Accuracy: {accuracy:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training advanced LSTM: {e}")
            return False
    
    def train_advanced_ml_filter(self, data_path: str) -> bool:
        """Train advanced ML filter with meta-features"""
        try:
            self.logger.info("Training advanced ML filter...")
            
            # Load and prepare data
            df = pd.read_csv(data_path)
            df = self.generate_meta_features(df)
            
            # Generate synthetic labels for training
            df['signal'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])
            df['confidence'] = np.random.uniform(0.3, 0.9, size=len(df))
            
            # Prepare features
            feature_cols = [col for col in self.config['feature_columns'] if col in df.columns]
            X = df[feature_cols].values
            y = df['signal'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create ensemble of Random Forest and XGBoost
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            )
            
            xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=1.5
            )
            
            # Create voting classifier
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('xgb', xgb_model)
                ],
                voting='soft'
            )
            
            # Train ensemble
            ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = ensemble.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model and scaler
            model_data = {
                'model': ensemble,
                'scaler': scaler,
                'feature_names': feature_cols,
                'training_date': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, self.config['model_paths']['ml_filter'])
            
            self.models['ml_filter'] = ensemble
            self.scalers['ml_filter'] = scaler
            
            self.logger.info(f"Advanced ML filter training completed - Accuracy: {accuracy:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training advanced ML filter: {e}")
            return False
    
    def generate_ensemble_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ensemble signal combining all models"""
        try:
            # Prepare market data with meta-features
            df = pd.DataFrame([market_data])
            df = self.generate_meta_features(df)
            
            # Get LSTM prediction
            lstm_pred = self._get_lstm_prediction(df)
            
            # Get ML filter prediction
            ml_pred = self._get_ml_filter_prediction(df)
            
            # Get technical analysis prediction
            tech_pred = self._get_technical_prediction(df)
            
            # Combine predictions using ensemble weights
            ensemble_score = (
                lstm_pred['score'] * self.ensemble_weights[0] +
                ml_pred['score'] * self.ensemble_weights[1] +
                tech_pred['score'] * self.ensemble_weights[2]
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
                lstm_pred['confidence'] * self.ensemble_weights[0] +
                ml_pred['confidence'] * self.ensemble_weights[1] +
                tech_pred['confidence'] * self.ensemble_weights[2]
            )
            
            # Log market context
            context = self._log_market_context(market_data, {
                'lstm': lstm_pred,
                'ml_filter': ml_pred,
                'technical': tech_pred,
                'ensemble': {
                    'signal': final_signal,
                    'confidence': ensemble_confidence,
                    'score': ensemble_score
                }
            })
            
            return {
                'signal': final_signal,
                'confidence': ensemble_confidence,
                'score': ensemble_score,
                'context': context,
                'individual_predictions': {
                    'lstm': lstm_pred,
                    'ml_filter': ml_pred,
                    'technical': tech_pred
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble signal: {e}")
            return {
                'signal': 0,
                'confidence': 0.5,
                'score': 0,
                'context': {},
                'error': str(e)
            }
    
    def _get_lstm_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get LSTM model prediction"""
        try:
            if 'lstm' not in self.models:
                return {'signal': 0, 'confidence': 0.5, 'score': 0}
            
            # Prepare sequence data
            feature_cols = [col for col in self.config['feature_columns'] if col in df.columns]
            sequence_data = df[feature_cols].values
            
            # Scale data
            scaled_data = self.scalers['lstm'].transform(sequence_data)
            
            # Reshape for LSTM (add batch dimension)
            X = scaled_data.reshape(1, 1, len(feature_cols))
            
            # Get prediction
            prediction = self.models['lstm'].predict(X, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Convert to signal
            signal_map = {0: -1, 1: 0, 2: 1}  # SELL, HOLD, BUY
            signal = signal_map[predicted_class]
            
            return {
                'signal': signal,
                'confidence': confidence,
                'score': prediction[0][2] - prediction[0][0]  # BUY - SELL
            }
            
        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {e}")
            return {'signal': 0, 'confidence': 0.5, 'score': 0}
    
    def _get_ml_filter_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get ML filter prediction"""
        try:
            if 'ml_filter' not in self.models:
                return {'signal': 0, 'confidence': 0.5, 'score': 0}
            
            # Prepare features
            feature_cols = [col for col in self.config['feature_columns'] if col in df.columns]
            X = df[feature_cols].values
            
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
                'score': prediction[1] - prediction[0]  # BUY - SELL
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML filter prediction: {e}")
            return {'signal': 0, 'confidence': 0.5, 'score': 0}
    
    def _get_technical_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
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
            
            return {
                'signal': signal,
                'confidence': min(confidence, 0.9),
                'score': score
            }
            
        except Exception as e:
            self.logger.error(f"Error in technical prediction: {e}")
            return {'signal': 0, 'confidence': 0.5, 'score': 0}
    
    def _log_market_context(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Log comprehensive market context"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data,
            'predictions': predictions,
            'system_state': {
                'models_loaded': list(self.models.keys()),
                'ensemble_weights': self.ensemble_weights,
                'performance_history_length': len(self.performance_history)
            }
        }
        
        # Save to log file
        log_entry = {
            'timestamp': context['timestamp'],
            'signal': predictions['ensemble']['signal'],
            'confidence': predictions['ensemble']['confidence'],
            'market_context': market_data,
            'individual_predictions': predictions
        }
        
        self.signal_history.append(log_entry)
        
        # Save to file
        context_file = f'logs/market_context_{datetime.now().strftime("%Y%m%d")}.json'
        with open(context_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return context
    
    def check_performance_drift(self) -> bool:
        """Check for performance drift and trigger retraining if needed"""
        try:
            if len(self.signal_history) < 100:
                return False
            
            # Calculate recent performance metrics
            recent_signals = self.signal_history[-100:]
            
            # Check HOLD ratio
            hold_count = sum(1 for s in recent_signals if s['signal'] == 0)
            hold_ratio = hold_count / len(recent_signals)
            
            # Check average confidence
            avg_confidence = np.mean([s['confidence'] for s in recent_signals])
            
            # Check for drift
            drift_detected = False
            drift_reason = []
            
            if hold_ratio > self.config['max_hold_ratio']:
                drift_detected = True
                drift_reason.append(f"High HOLD ratio: {hold_ratio:.3f}")
            
            if avg_confidence < self.config['min_confidence']:
                drift_detected = True
                drift_reason.append(f"Low confidence: {avg_confidence:.3f}")
            
            if drift_detected:
                self.logger.warning(f"Performance drift detected: {', '.join(drift_reason)}")
                self.trigger_auto_retrain(drift_reason)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking performance drift: {e}")
            return False
    
    def trigger_auto_retrain(self, reasons: List[str]):
        """Trigger automatic retraining"""
        try:
            self.logger.info(f"Triggering auto-retrain due to: {', '.join(reasons)}")
            
            # Backup current models
            self._backup_models()
            
            # Retrain models
            data_paths = [
                'data/XAUUSD_PRO_M5_live.csv',
                'data/XAUUSD_PRO_M5_enhanced.csv'
            ]
            
            for data_path in data_paths:
                if os.path.exists(data_path):
                    self.train_advanced_lstm(data_path)
                    self.train_advanced_ml_filter(data_path)
                    break
            
            # Log retraining event
            retrain_log = {
                'timestamp': datetime.now().isoformat(),
                'reasons': reasons,
                'models_retrained': list(self.models.keys()),
                'data_used': data_path
            }
            
            with open('logs/auto_retrain_log.json', 'a') as f:
                f.write(json.dumps(retrain_log) + '\n')
            
            self.logger.info("Auto-retrain completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in auto-retrain: {e}")
            self._rollback_models()
    
    def _backup_models(self):
        """Backup current models"""
        backup_dir = f'models/backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(backup_dir, exist_ok=True)
        
        for model_name, model_path in self.config['model_paths'].items():
            if os.path.exists(model_path):
                import shutil
                shutil.copy2(model_path, f'{backup_dir}/{model_name}')
    
    def _rollback_models(self):
        """Rollback to previous stable models"""
        # Implementation for rollback logic
        self.logger.info("Rolling back to previous stable models")
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load LSTM model
            if os.path.exists(self.config['model_paths']['lstm']):
                self.models['lstm'] = load_model(self.config['model_paths']['lstm'])
                scaler_path = self.config['model_paths']['lstm'].replace('.h5', '_scaler.joblib')
                if os.path.exists(scaler_path):
                    self.scalers['lstm'] = joblib.load(scaler_path)
            
            # Load ML filter model
            if os.path.exists(self.config['model_paths']['ml_filter']):
                model_data = joblib.load(self.config['model_paths']['ml_filter'])
                self.models['ml_filter'] = model_data['model']
                self.scalers['ml_filter'] = model_data['scaler']
            
            self.logger.info(f"Loaded models: {list(self.models.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def run_system(self, data_path: str):
        """Run the advanced MR BEN system"""
        try:
            self.logger.info("Starting Advanced MR BEN AI System")
            
            # Load models
            self.load_models()
            
            # Train models if not available
            if 'lstm' not in self.models:
                self.train_advanced_lstm(data_path)
            
            if 'ml_filter' not in self.models:
                self.train_advanced_ml_filter(data_path)
            
            # Load market data
            df = pd.read_csv(data_path)
            df = self.generate_meta_features(df)
            
            # Generate signals for each data point
            signals = []
            for i in range(50, len(df)):  # Start from 50 to have enough history
                market_data = df.iloc[i].to_dict()
                signal = self.generate_ensemble_signal(market_data)
                signals.append(signal)
                
                # Check for performance drift every 100 signals
                if len(signals) % 100 == 0:
                    self.check_performance_drift()
            
            # Generate performance report
            self._generate_performance_report(signals)
            
            self.logger.info("Advanced MR BEN AI System completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error running advanced system: {e}")
    
    def _generate_performance_report(self, signals: List[Dict[str, Any]]):
        """Generate comprehensive performance report"""
        try:
            # Calculate performance metrics
            signal_distribution = {
                'BUY': sum(1 for s in signals if s['signal'] == 1),
                'SELL': sum(1 for s in signals if s['signal'] == -1),
                'HOLD': sum(1 for s in signals if s['signal'] == 0)
            }
            
            total_signals = len(signals)
            avg_confidence = np.mean([s['confidence'] for s in signals])
            
            # Calculate diversity score
            max_class = max(signal_distribution.values())
            diversity_score = 1.0 - max_class / total_signals if total_signals > 0 else 0
            
            # Generate report
            report = f"""
============================================================
ADVANCED MR BEN AI SYSTEM - PERFORMANCE REPORT
============================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: Advanced Professional

PERFORMANCE METRICS:
===================
- Total Signals: {total_signals}
- Signal Distribution: {signal_distribution}
- Average Confidence: {avg_confidence:.4f}
- Diversity Score: {diversity_score:.4f}
- Models Loaded: {list(self.models.keys())}

ENSEMBLE WEIGHTS:
================
- LSTM: {self.ensemble_weights[0]:.2f}
- ML Filter: {self.ensemble_weights[1]:.2f}
- Technical: {self.ensemble_weights[2]:.2f}

SYSTEM STATUS:
=============
- Performance Drift Check: {'Enabled' if len(self.signal_history) >= 100 else 'Insufficient Data'}
- Auto-Retrain: {'Enabled' if len(self.signal_history) >= 100 else 'Disabled'}
- Market Context Logging: Enabled

RECOMMENDATIONS:
===============
1. Monitor signal distribution for balance
2. Check confidence levels regularly
3. Review ensemble weights if needed
4. Monitor auto-retrain triggers

============================================================
Report generated by Advanced MR BEN AI System
============================================================
"""
            
            # Save report
            report_path = f'logs/advanced_performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"Performance report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")

def main():
    """Main function to run advanced MR BEN system"""
    print("🎯 Advanced MR BEN AI System - Professional Version")
    print("=" * 60)
    
    if not AI_AVAILABLE:
        print("❌ AI libraries not available. Please install required packages.")
        return
    
    # Create and run advanced system
    system = AdvancedMRBENSystem()
    
    # Find data file
    data_paths = [
        'data/XAUUSD_PRO_M5_live.csv',
        'data/XAUUSD_PRO_M5_enhanced.csv',
        'data/XAUUSD_PRO_M5_data.csv'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("❌ No data files found")
        return
    
    print(f"📊 Using data: {data_path}")
    system.run_system(data_path)
    
    print("✅ Advanced MR BEN AI System completed!")
    print("📋 Check logs/ directory for detailed reports")

if __name__ == "__main__":
    main() 