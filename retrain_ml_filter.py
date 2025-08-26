#!/usr/bin/env python3
"""
MR BEN ML Filter Retraining Script
Uses latest trading logs to retrain the XGBoost ML filter
"""

import logging
import os
import re
from datetime import datetime

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLFilterRetrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'open',
            'high',
            'low',
            'close',
            'sma_20',
            'sma_50',
            'rsi',
            'macd',
            'macd_signal',
            'macd_hist',
        ]

    def extract_training_data_from_logs(self, log_file_path):
        """Extract training data from trading logs"""
        logger.info("Extracting ML filter training data from logs...")

        training_data = []

        try:
            with open(log_file_path, encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                # Look for ML Filter Features and corresponding signals
                if 'ML Filter Features:' in line:
                    try:
                        # Extract features
                        features_match = re.search(r'ML Filter Features: \[(.*?)\]', line)
                        if features_match:
                            features_str = features_match.group(1)
                            features = []
                            for item in features_str.split(', '):
                                if 'np.float64(' in item:
                                    value = float(item.split('(')[1].split(')')[0])
                                    features.append(value)

                            if len(features) >= 10:
                                # Look for corresponding signal in nearby lines
                                signal = self.find_signal_in_context(lines, i)

                                if signal is not None:
                                    data_point = {
                                        'features': features[:10],  # First 10 features
                                        'signal': signal,
                                        'timestamp': i,  # Use line number as timestamp
                                    }
                                    training_data.append(data_point)
                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Error reading log file: {e}")

        logger.info(f"Extracted {len(training_data)} training samples from logs")
        return training_data

    def find_signal_in_context(self, lines, current_line_idx, context=5):
        """Find signal in the context around the current line"""
        start_idx = max(0, current_line_idx - context)
        end_idx = min(len(lines), current_line_idx + context)

        for i in range(start_idx, end_idx):
            line = lines[i]

            # Look for signal information
            if 'Signal:' in line and 'Confidence:' in line:
                try:
                    # Extract signal value
                    signal_match = re.search(r'Signal: (\d+)', line)
                    if signal_match:
                        signal = int(signal_match.group(1))
                        return signal
                except:
                    continue

            # Look for LSTM result
            elif 'LSTM Result:' in line:
                try:
                    signal_match = re.search(r'Signal=(\d+)', line)
                    if signal_match:
                        signal = int(signal_match.group(1))
                        return signal
                except:
                    continue

            # Look for TA result
            elif 'TA Result:' in line:
                try:
                    signal_match = re.search(r'Signal=(\d+)', line)
                    if signal_match:
                        signal = int(signal_match.group(1))
                        return signal
                except:
                    continue

        return None

    def prepare_training_data(self, training_data):
        """Prepare training data for ML filter"""
        logger.info("Preparing ML filter training data...")

        X = []
        y = []

        for data_point in training_data:
            features = data_point['features']
            signal = data_point['signal']

            # Convert signal to binary (0=SELL/HOLD, 1=BUY)
            binary_signal = 1 if signal == 1 else 0

            X.append(features)
            y.append(binary_signal)

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Prepared {len(X)} samples with {len(X[0])} features")
        logger.info(f"Signal distribution: {np.bincount(y)}")

        return X, y

    def build_model(self):
        """Build XGBoost model"""
        logger.info("Building XGBoost ML filter model...")

        # Enhanced XGBoost parameters for better confidence
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'scale_pos_weight': 1.0,  # Handle class imbalance
            'min_child_weight': 3,
            'gamma': 0.1,
        }

        self.model = xgb.XGBClassifier(**params)
        logger.info("XGBoost model built successfully")
        return self.model

    def train_model(self, X, y):
        """Train the ML filter model"""
        logger.info("Starting ML filter training...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Build and train model
        self.model = self.build_model()

        # Train with early stopping
        eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]

        self.model.fit(
            X_train_scaled, y_train, eval_set=eval_set, early_stopping_rounds=20, verbose=True
        )

        # Evaluate model
        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)

        accuracy = accuracy_score(y_val, y_pred)
        logger.info(f"Validation Accuracy: {accuracy:.4f}")

        # Calculate confidence metrics
        confidence_scores = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(confidence_scores)
        logger.info(f"Average Confidence: {avg_confidence:.4f}")

        # Generate detailed report
        self.generate_report(X_val_scaled, y_val, y_pred, y_pred_proba)

        return accuracy, avg_confidence

    def generate_report(self, X_val, y_val, y_pred, y_pred_proba):
        """Generate detailed training report"""
        logger.info("Generating ML filter training report...")

        # Classification report
        report = classification_report(y_val, y_pred, target_names=['SELL/HOLD', 'BUY'])

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)

        # Confidence analysis
        confidence_scores = np.max(y_pred_proba, axis=1)
        confidence_stats = {
            'mean': np.mean(confidence_scores),
            'std': np.std(confidence_scores),
            'min': np.min(confidence_scores),
            'max': np.max(confidence_scores),
            'median': np.median(confidence_scores),
        }

        # Feature importance
        feature_importance = self.model.feature_importances_

        # Save report
        if not os.path.exists("models"):
            os.makedirs("models")

        with open("models/ml_filter_training_report.txt", "w") as f:
            f.write("MR BEN ML Filter Training Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Training Date: {datetime.now()}\n")
            f.write("Model Type: XGBoost Classifier\n")
            f.write(f"Dataset Size: {len(X_val)} validation samples\n")
            f.write(f"Features: {len(self.feature_names)}\n")
            f.write("\n" + "=" * 40 + "\n")
            f.write("CLASSIFICATION REPORT:\n")
            f.write(report)
            f.write("\n" + "=" * 40 + "\n")
            f.write("CONFUSION MATRIX:\n")
            f.write(str(cm))
            f.write("\n" + "=" * 40 + "\n")
            f.write("CONFIDENCE STATISTICS:\n")
            for stat, value in confidence_stats.items():
                f.write(f"{stat.capitalize()}: {value:.4f}\n")
            f.write("\n" + "=" * 40 + "\n")
            f.write("FEATURE IMPORTANCE:\n")
            for i, (feature, importance) in enumerate(
                zip(self.feature_names, feature_importance, strict=False)
            ):
                f.write(f"{feature}: {importance:.4f}\n")

        logger.info("Training report saved to models/ml_filter_training_report.txt")

    def save_model(self, model_path="models/mrben_ml_filter_updated.joblib"):
        """Save the trained model"""
        if not os.path.exists("models"):
            os.makedirs("models")

        # Save model and scaler together
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_date': datetime.now(),
        }

        joblib.dump(model_data, model_path)
        logger.info(f"ML Filter model saved to {model_path}")

    def retrain_complete(self):
        """Complete ML filter retraining process"""
        logger.info("Starting complete ML filter retraining process...")

        # Extract training data from logs
        if not os.path.exists("lstm_trading_system.log"):
            logger.error("Trading log file not found")
            return False

        training_data = self.extract_training_data_from_logs("lstm_trading_system.log")

        if len(training_data) < 50:
            logger.error("Insufficient training data")
            return False

        # Prepare training data
        X, y = self.prepare_training_data(training_data)

        # Train model
        accuracy, avg_confidence = self.train_model(X, y)

        # Save model
        self.save_model()

        logger.info("ML Filter retraining completed successfully!")
        logger.info(f"Final Accuracy: {accuracy:.4f}")
        logger.info(f"Average Confidence: {avg_confidence:.4f}")

        return True


def main():
    """Main function"""
    logger.info("Starting MR BEN ML Filter Retraining...")

    retrainer = MLFilterRetrainer()
    success = retrainer.retrain_complete()

    if success:
        logger.info("✅ ML Filter retraining completed successfully!")
        logger.info("📁 Model saved to: models/mrben_ml_filter_updated.joblib")
        logger.info("📄 Report saved to: models/ml_filter_training_report.txt")
    else:
        logger.error("❌ ML Filter retraining failed!")


if __name__ == "__main__":
    main()
