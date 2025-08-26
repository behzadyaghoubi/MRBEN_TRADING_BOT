#!/usr/bin/env python3
"""
Enhanced Retrainer for MRBEN AI System
Trains both LSTM and ML Filter models using live trading logs
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. LSTM training will be skipped.")


class EnhancedRetrainer:
    def __init__(
        self, trade_log_path="data/trade_log_clean.csv", model_dir="models", sequence_length=20
    ):
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
            # Load the trade log
            df = pd.read_csv(self.trade_log_path)
            print(f"✅ Loaded {len(df)} trade records")

            # Check column structure
            print(f"Columns: {list(df.columns)}")

            # Basic data cleaning
            df = df.dropna()
            print(f"✅ After cleaning: {len(df)} records")

            return df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def create_ml_features(self, df):
        """Create features for ML filter model"""
        print("🔧 Creating ML filter features...")

        try:
            # Extract basic features from trade log
            features = []

            # Price-based features
            if 'entry_price' in df.columns:
                features.append(df['entry_price'].values)
                features.append(df['sl_price'].values)
                features.append(df['tp_price'].values)

                # Calculate price ratios
                sl_distance = np.abs(df['entry_price'] - df['sl_price'])
                tp_distance = np.abs(df['tp_price'] - df['entry_price'])
                risk_reward_ratio = tp_distance / (sl_distance + 1e-8)
                features.append(risk_reward_ratio)

            # Confidence features
            if 'confidence' in df.columns:
                features.append(df['confidence'].values)

            # Volume features
            if 'volume' in df.columns:
                features.append(df['volume'].values)

            # Time-based features (if timestamp available)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                features.extend([df['hour'].values, df['day_of_week'].values])

            # Combine all features
            X = np.column_stack(features)

            # Create labels (simplified - you might want to enhance this)
            # For now, we'll use a simple heuristic based on price movement
            if 'entry_price' in df.columns and 'tp_price' in df.columns:
                # Simple label: if TP is reached, it's a win
                # This is a placeholder - you should implement proper labeling
                y = np.random.randint(0, 2, size=len(df))  # Placeholder
                print("⚠️ Using placeholder labels. Implement proper labeling logic.")
            else:
                y = np.random.randint(0, 2, size=len(df))  # Placeholder

            print(f"✅ ML features created: X shape {X.shape}, y shape {y.shape}")
            return X, y

        except Exception as e:
            print(f"❌ Error creating ML features: {e}")
            return None, None

    def create_lstm_features(self, df):
        """Create sequential features for LSTM model"""
        print("🔧 Creating LSTM features...")

        try:
            # For LSTM, we need sequential data
            # This is a simplified version - you might want to enhance it

            # Basic features
            basic_features = []

            if 'entry_price' in df.columns:
                basic_features.append(df['entry_price'].values)
            if 'confidence' in df.columns:
                basic_features.append(df['confidence'].values)
            if 'volume' in df.columns:
                basic_features.append(df['volume'].values)

            if not basic_features:
                print("❌ No suitable features found for LSTM")
                return None, None

            # Combine features
            X = np.column_stack(basic_features)

            # Create sequences
            X_seq, y_seq = [], []
            for i in range(len(X) - self.sequence_length):
                X_seq.append(X[i : i + self.sequence_length])
                # Simple label based on next price movement (placeholder)
                y_seq.append(np.random.randint(0, 2))

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            print(f"✅ LSTM features created: X shape {X_seq.shape}, y shape {y_seq.shape}")
            return X_seq, y_seq

        except Exception as e:
            print(f"❌ Error creating LSTM features: {e}")
            return None, None

    def train_ml_filter(self, X, y):
        """Train the ML filter model"""
        print("🤖 Training ML Filter model...")

        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train multiple models and pick the best
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBClassifier(
                    use_label_encoder=False, eval_metric='logloss', random_state=42
                ),
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
            }

            joblib.dump(model_data, self.ml_filter_path)
            print(f"✅ ML Filter model saved to {self.ml_filter_path}")

            # Print detailed metrics
            y_pred = best_model.predict(X_test_scaled)
            print("\n📊 Classification Report:")
            print(classification_report(y_test, y_pred))

            return best_model, scaler, best_score

        except Exception as e:
            print(f"❌ Error training ML Filter: {e}")
            return None, None, 0

    def train_lstm(self, X, y):
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

            # Create LSTM model
            model = Sequential(
                [
                    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25, activation='relu'),
                    Dense(1, activation='sigmoid'),
                ]
            )

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )

            # Train model
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                verbose=1,
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
        print("🚀 Starting Enhanced Retrainer...")
        print("=" * 50)

        # Load data
        df = self.load_and_prepare_data()
        if df is None:
            return

        # Train ML Filter
        print("\n" + "=" * 50)
        X_ml, y_ml = self.create_ml_features(df)
        if X_ml is not None and y_ml is not None:
            ml_model, ml_scaler, ml_score = self.train_ml_filter(X_ml, y_ml)
        else:
            print("❌ ML Filter training skipped due to data issues")

        # Train LSTM
        print("\n" + "=" * 50)
        X_lstm, y_lstm = self.create_lstm_features(df)
        if X_lstm is not None and y_lstm is not None:
            lstm_model, lstm_score = self.train_lstm(X_lstm, y_lstm)
        else:
            print("❌ LSTM training skipped due to data issues")

        print("\n" + "=" * 50)
        print("🎯 Retraining Complete!")

        if 'ml_model' in locals() and ml_model is not None:
            print(f"✅ ML Filter: {ml_score:.4f}")
        if 'lstm_model' in locals() and lstm_model is not None:
            print(f"✅ LSTM: {lstm_score:.4f}")

        print("\n💡 Next steps:")
        print("1. Test the models with live_trader_clean.py")
        print("2. Monitor performance and retrain as needed")
        print("3. Consider implementing proper labeling logic")


def main():
    """Main execution function"""
    retrainer = EnhancedRetrainer()
    retrainer.retrain_all()


if __name__ == "__main__":
    main()
