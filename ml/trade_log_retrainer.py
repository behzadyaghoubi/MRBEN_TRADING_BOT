#!/usr/bin/env python3
"""
Trade Log Retrainer for MRBEN AI System
Specialized for the actual trade log format we have
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

try:
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. LSTM training will be skipped.")

class TradeLogRetrainer:
    def __init__(self, 
                 trade_log_path="data/trade_log_clean.csv",
                 model_dir="models",
                 sequence_length=20):
        self.trade_log_path = trade_log_path
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Model paths
        self.lstm_path = os.path.join(model_dir, "advanced_lstm_model.h5")
        self.ml_filter_path = os.path.join(model_dir, "quick_fix_ml_filter.joblib")
        
    def load_and_prepare_data(self):
        """Load trade log and prepare features for training"""
        print("📊 Loading trade log data...")
        
        try:
            # Load the trade log - it has no headers, so we'll define them
            columns = [
                'timestamp', 'signal', 'entry_price', 'sl_price', 'tp_price', 
                'status', 'buy_proba', 'sell_proba', 'r_multiple', 'price'
            ]
            
            df = pd.read_csv(self.trade_log_path, header=None, names=columns)
            print(f"✅ Loaded {len(df)} trade records")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Basic data cleaning
            df = df.dropna()
            print(f"✅ After cleaning: {len(df)} records")
            
            # Show data info
            print(f"📊 Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"📊 Signal distribution: {df['signal'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def create_ml_features(self, df):
        """Create features for ML filter model"""
        print("🔧 Creating ML filter features...")
        
        try:
            features = []
            feature_names = []
            
            # 1. Price-based features
            if 'entry_price' in df.columns:
                features.append(df['entry_price'].values)
                feature_names.append('entry_price')
                
                features.append(df['sl_price'].values)
                feature_names.append('sl_price')
                
                features.append(df['tp_price'].values)
                feature_names.append('tp_price')
                
                # Calculate price ratios and distances
                sl_distance = np.abs(df['entry_price'] - df['sl_price'])
                tp_distance = np.abs(df['tp_price'] - df['entry_price'])
                risk_reward_ratio = tp_distance / (sl_distance + 1e-8)
                
                features.append(sl_distance)
                feature_names.append('sl_distance')
                
                features.append(tp_distance)
                feature_names.append('tp_distance')
                
                features.append(risk_reward_ratio)
                feature_names.append('risk_reward_ratio')
            
            # 2. Probability features
            if 'buy_proba' in df.columns:
                features.append(df['buy_proba'].values)
                feature_names.append('buy_proba')
                
                features.append(df['sell_proba'].values)
                feature_names.append('sell_proba')
                
                # Probability difference
                prob_diff = df['buy_proba'] - df['sell_proba']
                features.append(prob_diff.values)
                feature_names.append('prob_diff')
            
            # 3. R-multiple feature
            if 'r_multiple' in df.columns:
                features.append(df['r_multiple'].values)
                feature_names.append('r_multiple')
            
            # 4. Time-based features
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['month'] = df['timestamp'].dt.month
                
                features.extend([
                    df['hour'].values,
                    df['day_of_week'].values,
                    df['month'].values
                ])
                feature_names.extend(['hour', 'day_of_week', 'month'])
            
            # 5. Signal-based features
            if 'signal' in df.columns:
                features.append(df['signal'].values)
                feature_names.append('signal')
            
            # Combine all features
            X = np.column_stack(features)
            
            print(f"✅ ML features created: X shape {X.shape}")
            print(f"📋 Feature names: {feature_names}")
            
            # Create labels - this is where you'd implement proper labeling logic
            # For now, we'll use a simple heuristic based on signal direction
            # In a real scenario, you'd want to implement proper backtesting
            y = self._create_labels(df)
            
            return X, y, feature_names
            
        except Exception as e:
            print(f"❌ Error creating ML features: {e}")
            return None, None, None
    
    def _create_labels(self, df):
        """Create labels for training - this is a placeholder implementation"""
        print("⚠️ Using placeholder labels. Implement proper labeling logic for production.")
        
        # Placeholder: simple random labels for demonstration
        # In reality, you'd want to:
        # 1. Backtest the signals with actual price data
        # 2. Calculate whether each signal would have been profitable
        # 3. Consider slippage, spreads, and transaction costs
        
        np.random.seed(42)  # For reproducible results
        y = np.random.randint(0, 2, size=len(df))
        
        print(f"📊 Label distribution: {np.bincount(y)}")
        return y
    
    def create_lstm_features(self, df):
        """Create sequential features for LSTM model"""
        print("🔧 Creating LSTM features...")
        
        try:
            # For LSTM, we need sequential data
            basic_features = []
            feature_names = []
            
            # Price features
            if 'entry_price' in df.columns:
                basic_features.append(df['entry_price'].values)
                feature_names.append('entry_price')
            
            # Probability features
            if 'buy_proba' in df.columns:
                basic_features.append(df['buy_proba'].values)
                feature_names.append('buy_proba')
                
                basic_features.append(df['sell_proba'].values)
                feature_names.append('sell_proba')
            
            # R-multiple
            if 'r_multiple' in df.columns:
                basic_features.append(df['r_multiple'].values)
                feature_names.append('r_multiple')
            
            if not basic_features:
                print("❌ No suitable features found for LSTM")
                return None, None, None
            
            # Combine features
            X = np.column_stack(basic_features)
            
            # Create sequences
            X_seq, y_seq = [], []
            for i in range(len(X) - self.sequence_length):
                X_seq.append(X[i:i+self.sequence_length])
                # Simple label based on next signal (placeholder)
                y_seq.append(1 if df['signal'].iloc[i+self.sequence_length] > 0 else 0)
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            print(f"✅ LSTM features created: X shape {X_seq.shape}, y shape {y_seq.shape}")
            print(f"📋 LSTM feature names: {feature_names}")
            
            return X_seq, y_seq, feature_names
            
        except Exception as e:
            print(f"❌ Error creating LSTM features: {e}")
            return None, None, None
    
    def train_ml_filter(self, X, y, feature_names):
        """Train the ML filter model"""
        print("🤖 Training ML Filter model...")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 Training set: {len(X_train)} samples")
            print(f"📊 Test set: {len(X_test)} samples")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models and pick the best
            models = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5
                ),
                'XGBoost': xgb.XGBClassifier(
                    use_label_encoder=False, 
                    eval_metric='logloss', 
                    random_state=42,
                    max_depth=6,
                    learning_rate=0.1
                )
            }
            
            best_model = None
            best_score = 0
            best_model_name = None
            
            for name, model in models.items():
                print(f"  Training {name}...")
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                print(f"  {name} accuracy: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            
            print(f"✅ Best model: {best_model_name} with accuracy {best_score:.4f}")
            
            # Save the best model and scaler
            model_data = {
                'model': best_model,
                'scaler': scaler,
                'model_name': best_model_name,
                'accuracy': best_score,
                'feature_count': X.shape[1],
                'feature_names': feature_names
            }
            
            joblib.dump(model_data, self.ml_filter_path)
            print(f"✅ ML Filter model saved to {self.ml_filter_path}")
            
            # Print detailed metrics
            y_pred = best_model.predict(X_test_scaled)
            print("\n📊 Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Feature importance (for tree-based models)
            if hasattr(best_model, 'feature_importances_'):
                print("\n🔍 Feature Importance:")
                importance = best_model.feature_importances_
                for i, (name, imp) in enumerate(zip(feature_names, importance)):
                    print(f"  {name}: {imp:.4f}")
            
            return best_model, scaler, best_score
            
        except Exception as e:
            print(f"❌ Error training ML Filter: {e}")
            return None, None, 0
    
    def train_lstm(self, X, y, feature_names):
        """Train the LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ Skipping LSTM training - TensorFlow not available")
            return None, 0
        
        print("🧠 Training LSTM model...")
        
        try:
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"📊 LSTM Training set: {len(X_train)} sequences")
            print(f"📊 LSTM Test set: {len(X_test)} sequences")
            
            # Create LSTM model
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.3),
                LSTM(32, return_sequences=False),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print("🏗️ LSTM Model Architecture:")
            model.summary()
            
            # Train model
            print("\n🚀 Starting LSTM training...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=30,
                batch_size=16,
                verbose=1
            )
            
            # Evaluate
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"✅ LSTM test accuracy: {test_acc:.4f}")
            
            # Save model
            model.save(self.lstm_path)
            print(f"✅ LSTM model saved to {self.lstm_path}")
            
            return model, test_acc
            
        except Exception as e:
            print(f"❌ Error training LSTM: {e}")
            return None, 0
    
    def retrain_all(self):
        """Main method to retrain all models"""
        print("🚀 Starting Trade Log Retrainer...")
        print("=" * 60)
        
        # Load data
        df = self.load_and_prepare_data()
        if df is None:
            return
        
        # Train ML Filter
        print("\n" + "=" * 60)
        X_ml, y_ml, ml_features = self.create_ml_features(df)
        if X_ml is not None and y_ml is not None:
            ml_model, ml_scaler, ml_score = self.train_ml_filter(X_ml, y_ml, ml_features)
        else:
            print("❌ ML Filter training skipped due to data issues")
        
        # Train LSTM
        print("\n" + "=" * 60)
        X_lstm, y_lstm, lstm_features = self.create_lstm_features(df)
        if X_lstm is not None and y_lstm is not None:
            lstm_model, lstm_score = self.train_lstm(X_lstm, y_lstm, lstm_features)
        else:
            print("❌ LSTM training skipped due to data issues")
        
        print("\n" + "=" * 60)
        print("🎯 Retraining Complete!")
        
        if 'ml_model' in locals() and ml_model is not None:
            print(f"✅ ML Filter: {ml_score:.4f}")
        if 'lstm_model' in locals() and lstm_model is not None:
            print(f"✅ LSTM: {lstm_score:.4f}")
        
        print("\n💡 Next steps:")
        print("1. Test the models with live_trader_clean.py")
        print("2. Monitor performance and retrain as needed")
        print("3. Implement proper labeling logic for production use")
        print("4. Consider using actual backtest results for labels")

def main():
    """Main execution function"""
    retrainer = TradeLogRetrainer()
    retrainer.retrain_all()

if __name__ == "__main__":
    main()
