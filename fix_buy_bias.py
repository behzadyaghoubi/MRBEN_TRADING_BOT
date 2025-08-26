#!/usr/bin/env python3
"""
MR BEN - Fix BUY Bias Script
============================
This script fixes the BUY bias issue by:
1. Rebalancing the dataset
2. Retraining XGBoost model with balanced data
3. Retraining LSTM model with balanced data
4. Testing the balanced models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class BuyBiasFixer:
    def __init__(self):
        self.original_data = None
        self.balanced_data = None
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = None
        
    def load_and_analyze_data(self):
        """Load and analyze the current dataset."""
        print("🔍 Loading and analyzing current dataset...")
        
        # Load the dataset
        self.original_data = pd.read_csv('data/mrben_ai_signal_dataset.csv')
        
        # Analyze signal distribution
        signal_counts = self.original_data['signal'].value_counts()
        print(f"\n📊 Current Signal Distribution:")
        print(signal_counts)
        print(f"\nTotal signals: {len(self.original_data)}")
        
        # Calculate imbalance
        total = len(self.original_data)
        hold_pct = signal_counts.get('HOLD', 0) / total * 100
        buy_pct = signal_counts.get('BUY', 0) / total * 100
        sell_pct = signal_counts.get('SELL', 0) / total * 100
        
        print(f"\n📈 Distribution Percentages:")
        print(f"HOLD: {hold_pct:.1f}%")
        print(f"BUY: {buy_pct:.1f}%")
        print(f"SELL: {sell_pct:.1f}%")
        
        return signal_counts
    
    def create_balanced_dataset(self, target_ratio=0.3):
        """Create a balanced dataset with equal BUY/SELL signals."""
        print(f"\n⚖️ Creating balanced dataset with {target_ratio*100}% BUY/SELL each...")
        
        # Separate signals
        hold_data = self.original_data[self.original_data['signal'] == 'HOLD']
        buy_data = self.original_data[self.original_data['signal'] == 'BUY']
        sell_data = self.original_data[self.original_data['signal'] == 'SELL']
        
        print(f"Original counts - HOLD: {len(hold_data)}, BUY: {len(buy_data)}, SELL: {len(sell_data)}")
        
        # Calculate target counts
        total_target = len(self.original_data)
        target_count = int(total_target * target_ratio)
        
        # Balance BUY and SELL
        if len(buy_data) < target_count:
            # Oversample BUY data
            buy_data_balanced = buy_data.sample(n=target_count, replace=True, random_state=42)
        else:
            # Undersample BUY data
            buy_data_balanced = buy_data.sample(n=target_count, random_state=42)
            
        if len(sell_data) < target_count:
            # Oversample SELL data
            sell_data_balanced = sell_data.sample(n=target_count, replace=True, random_state=42)
        else:
            # Undersample SELL data
            sell_data_balanced = sell_data.sample(n=target_count, random_state=42)
        
        # Adjust HOLD data
        hold_target = total_target - len(buy_data_balanced) - len(sell_data_balanced)
        if len(hold_data) > hold_target:
            hold_data_balanced = hold_data.sample(n=hold_target, random_state=42)
        else:
            hold_data_balanced = hold_data.sample(n=hold_target, replace=True, random_state=42)
        
        # Combine balanced data
        self.balanced_data = pd.concat([hold_data_balanced, buy_data_balanced, sell_data_balanced])
        self.balanced_data = self.balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Verify balance
        balanced_counts = self.balanced_data['signal'].value_counts()
        print(f"\n✅ Balanced dataset created:")
        print(balanced_counts)
        
        return self.balanced_data
    
    def prepare_xgboost_data(self):
        """Prepare data for XGBoost training."""
        print("\n🤖 Preparing XGBoost training data...")
        
        # Select features
        feature_columns = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist']
        available_features = [col for col in feature_columns if col in self.balanced_data.columns]
        
        if not available_features:
            print("❌ No suitable features found!")
            return None, None
        
        # Prepare features and labels
        X = self.balanced_data[available_features].copy()
        
        # Create numeric labels
        label_map = {'HOLD': 0, 'SELL': 1, 'BUY': 2}
        y = self.balanced_data['signal'].map(label_map)
        
        print(f"Features: {available_features}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def train_balanced_xgboost(self, X, y):
        """Train XGBoost model with balanced data."""
        print("\n🚀 Training balanced XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        print(f"Class weights: {weight_dict}")
        
        # Train model
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        # Fit with sample weights
        sample_weights = np.array([weight_dict[label] for label in y_train])
        self.xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate
        y_pred = self.xgb_model.predict(X_test)
        
        print("\n📊 XGBoost Model Performance:")
        print(classification_report(y_test, y_pred, target_names=['HOLD', 'SELL', 'BUY']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📈 Confusion Matrix:")
        print(cm)
        
        # Test prediction distribution
        pred_counts = pd.Series(y_pred).value_counts()
        print(f"\n🎯 Prediction Distribution:")
        print(f"HOLD: {pred_counts.get(0, 0)}")
        print(f"SELL: {pred_counts.get(1, 0)}")
        print(f"BUY: {pred_counts.get(2, 0)}")
        
        return self.xgb_model
    
    def prepare_lstm_data(self):
        """Prepare data for LSTM training."""
        print("\n🧠 Preparing LSTM training data...")
        
        # Select features for LSTM
        feature_columns = ['open', 'high', 'low', 'close', 'RSI', 'MACD']
        available_features = [col for col in feature_columns if col in self.balanced_data.columns]
        
        if not available_features:
            print("❌ No suitable features found for LSTM!")
            return None, None
        
        # Prepare features
        X = self.balanced_data[available_features].values
        
        # Normalize features
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences (lookback=10)
        lookback = 10
        X_sequences = []
        y_sequences = []
        
        for i in range(lookback, len(X_scaled)):
            X_sequences.append(X_scaled[i-lookback:i])
            y_sequences.append(self.balanced_data.iloc[i]['signal'])
        
        X_sequences = np.array(X_sequences)
        
        # Convert labels to numeric
        label_map = {'HOLD': 0, 'SELL': 1, 'BUY': 2}
        y_sequences = np.array([label_map[label] for label in y_sequences])
        
        # Convert to one-hot encoding
        y_onehot = tf.keras.utils.to_categorical(y_sequences, num_classes=3)
        
        print(f"LSTM sequences shape: {X_sequences.shape}")
        print(f"LSTM labels shape: {y_onehot.shape}")
        
        return X_sequences, y_onehot
    
    def train_balanced_lstm(self, X_sequences, y_onehot):
        """Train LSTM model with balanced data."""
        print("\n🚀 Training balanced LSTM model...")
        
        # Split data
        split_idx = int(len(X_sequences) * 0.7)
        X_train = X_sequences[:split_idx]
        X_test = X_sequences[split_idx:]
        y_train = y_onehot[:split_idx]
        y_test = y_onehot[split_idx:]
        
        # Calculate class weights
        y_train_labels = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train_labels), y=y_train_labels
        )
        weight_dict = dict(zip(range(3), class_weights))
        
        print(f"LSTM Class weights: {weight_dict}")
        
        # Build LSTM model
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            class_weight=weight_dict,
            verbose=1
        )
        
        # Evaluate
        y_pred_proba = self.lstm_model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("\n📊 LSTM Model Performance:")
        print(classification_report(y_true, y_pred, target_names=['HOLD', 'SELL', 'BUY']))
        
        # Test prediction distribution
        pred_counts = pd.Series(y_pred).value_counts()
        print(f"\n🎯 LSTM Prediction Distribution:")
        print(f"HOLD: {pred_counts.get(0, 0)}")
        print(f"SELL: {pred_counts.get(1, 0)}")
        print(f"BUY: {pred_counts.get(2, 0)}")
        
        return self.lstm_model
    
    def save_models(self):
        """Save the trained models."""
        print("\n💾 Saving balanced models...")
        
        # Save XGBoost model
        if self.xgb_model:
            joblib.dump(self.xgb_model, 'models/mrben_ai_signal_filter_xgb_balanced.joblib')
            print("✅ XGBoost model saved as 'mrben_ai_signal_filter_xgb_balanced.joblib'")
        
        # Save LSTM model
        if self.lstm_model:
            self.lstm_model.save('models/mrben_lstm_balanced_new.h5')
            print("✅ LSTM model saved as 'mrben_lstm_balanced_new.h5'")
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, 'models/mrben_lstm_scaler_balanced.save')
            print("✅ LSTM scaler saved as 'mrben_lstm_scaler_balanced.save'")
    
    def test_models(self):
        """Test the models with sample data."""
        print("\n🧪 Testing models with sample data...")
        
        # Test XGBoost
        if self.xgb_model:
            print("\n🤖 Testing XGBoost model:")
            # Create sample features
            sample_features = np.array([[50, 0.5, 0.3, 0.2],  # Neutral
                                      [20, -1.0, -0.8, -0.2],  # Bearish
                                      [80, 1.0, 0.8, 0.2]])   # Bullish
            
            predictions = self.xgb_model.predict(sample_features)
            probabilities = self.xgb_model.predict_proba(sample_features)
            
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                signal_name = ['HOLD', 'SELL', 'BUY'][pred]
                print(f"Sample {i+1}: {signal_name} (prob: {prob[pred]:.3f})")
        
        # Test LSTM
        if self.lstm_model and self.scaler:
            print("\n🧠 Testing LSTM model:")
            # Create sample sequence
            sample_sequence = np.random.rand(10, 6)  # 10 timesteps, 6 features
            sample_sequence_scaled = self.scaler.transform(sample_sequence)
            sample_sequence_reshaped = sample_sequence_scaled.reshape(1, 10, 6)
            
            prediction = self.lstm_model.predict(sample_sequence_reshaped)
            predicted_class = np.argmax(prediction[0])
            signal_name = ['HOLD', 'SELL', 'BUY'][predicted_class]
            
            print(f"Sample sequence: {signal_name} (prob: {prediction[0][predicted_class]:.3f})")
    
    def run_complete_fix(self):
        """Run the complete BUY bias fix process."""
        print("🚀 Starting BUY Bias Fix Process...")
        print("=" * 50)
        
        # Step 1: Load and analyze data
        signal_counts = self.load_and_analyze_data()
        
        # Step 2: Create balanced dataset
        self.create_balanced_dataset()
        
        # Step 3: Train XGBoost
        X, y = self.prepare_xgboost_data()
        if X is not None:
            self.train_balanced_xgboost(X, y)
        
        # Step 4: Train LSTM
        X_sequences, y_onehot = self.prepare_lstm_data()
        if X_sequences is not None:
            self.train_balanced_lstm(X_sequences, y_onehot)
        
        # Step 5: Save models
        self.save_models()
        
        # Step 6: Test models
        self.test_models()
        
        print("\n🎉 BUY Bias Fix Process Completed!")
        print("=" * 50)

def main():
    """Main function."""
    fixer = BuyBiasFixer()
    fixer.run_complete_fix()

if __name__ == "__main__":
    main() 