#!/usr/bin/env python3
"""
Parameter Optimization Script for MR BEN Trading System
Optimizes LSTM and ML Filter parameters for better performance
"""

import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_models import build_advanced_lstm


class ParameterOptimizer:
    """Parameter optimizer for trading models."""

    def __init__(self):
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging."""
        logger = logging.getLogger('ParameterOptimizer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
            logger.addHandler(ch)

        return logger

    def optimize_lstm_parameters(self, X_train, y_train, X_val, y_val):
        """Optimize LSTM model parameters."""
        self.logger.info("🔍 Optimizing LSTM parameters...")

        # Parameter grid for LSTM optimization
        param_grid = {
            'learning_rate': [0.001, 0.0005, 0.0001],
            'lstm_units': [64, 128, 256],
            'dropout_rate': [0.2, 0.3, 0.4],
            'batch_size': [16, 32, 64],
        }

        best_score = 0
        best_params = {}

        for lr in param_grid['learning_rate']:
            for units in param_grid['lstm_units']:
                for dropout in param_grid['dropout_rate']:
                    for batch_size in param_grid['batch_size']:
                        try:
                            # Build model with current parameters
                            model = build_advanced_lstm(
                                input_shape=(X_train.shape[1], X_train.shape[2]),
                                lstm_units=units,
                                dropout_rate=dropout,
                            )

                            # Compile with current learning rate
                            model.compile(
                                optimizer=Adam(learning_rate=lr),
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'],
                            )

                            # Train model
                            history = model.fit(
                                X_train,
                                y_train,
                                validation_data=(X_val, y_val),
                                epochs=20,  # Reduced for optimization
                                batch_size=batch_size,
                                verbose=0,
                            )

                            # Evaluate
                            val_accuracy = max(history.history['val_accuracy'])

                            if val_accuracy > best_score:
                                best_score = val_accuracy
                                best_params = {
                                    'learning_rate': lr,
                                    'lstm_units': units,
                                    'dropout_rate': dropout,
                                    'batch_size': batch_size,
                                }

                                self.logger.info(
                                    f"✅ New best score: {best_score:.4f} with params: {best_params}"
                                )

                        except Exception as e:
                            self.logger.warning(
                                f"Error with params {lr}, {units}, {dropout}, {batch_size}: {e}"
                            )
                            continue

        self.logger.info(f"🎯 Best LSTM parameters: {best_params}")
        self.logger.info(f"🎯 Best validation accuracy: {best_score:.4f}")

        return best_params, best_score

    def optimize_ml_filter_parameters(self, X_train, y_train):
        """Optimize ML Filter parameters."""
        self.logger.info("🔍 Optimizing ML Filter parameters...")

        try:
            from xgboost import XGBClassifier

            # Parameter grid for XGBoost optimization
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.2, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
            }

            # Grid search
            xgb_model = XGBClassifier(random_state=42)
            grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            self.logger.info(f"🎯 Best ML Filter parameters: {best_params}")
            self.logger.info(f"🎯 Best cross-validation accuracy: {best_score:.4f}")

            return best_params, best_score

        except Exception as e:
            self.logger.error(f"Error optimizing ML Filter: {e}")
            return {}, 0.0

    def evaluate_model_performance(self, model, X_test, y_test, model_type='lstm'):
        """Evaluate model performance with detailed metrics."""
        self.logger.info(f"📊 Evaluating {model_type.upper()} model performance...")

        try:
            if model_type == 'lstm':
                # LSTM evaluation
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                y_pred = np.argmax(model.predict(X_test), axis=1)

            else:
                # ML Filter evaluation
                y_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)

            # Calculate detailed metrics
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            metrics = {
                'accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
            }

            self.logger.info(f"📈 {model_type.upper()} Performance Metrics:")
            self.logger.info(f"   Accuracy: {test_accuracy:.4f}")
            self.logger.info(f"   Precision: {precision:.4f}")
            self.logger.info(f"   Recall: {recall:.4f}")
            self.logger.info(f"   F1-Score: {f1:.4f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Error evaluating {model_type} model: {e}")
            return {}

    def save_optimized_model(self, model, model_path, model_type='lstm'):
        """Save optimized model."""
        try:
            os.makedirs('models', exist_ok=True)

            if model_type == 'lstm':
                model.save(model_path)
            else:
                joblib.dump(model, model_path)

            self.logger.info(f"✅ Optimized {model_type.upper()} model saved to: {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving {model_type} model: {e}")
            return False

    def generate_optimization_report(self, lstm_params, lstm_score, ml_params, ml_score):
        """Generate optimization report."""
        report = f"""
🚀 MR BEN Parameter Optimization Report
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 LSTM Model Optimization Results:
   Best Validation Accuracy: {lstm_score:.4f}
   Optimal Parameters:
   - Learning Rate: {lstm_params.get('learning_rate', 'N/A')}
   - LSTM Units: {lstm_params.get('lstm_units', 'N/A')}
   - Dropout Rate: {lstm_params.get('dropout_rate', 'N/A')}
   - Batch Size: {lstm_params.get('batch_size', 'N/A')}

🔍 ML Filter Optimization Results:
   Best Cross-Validation Accuracy: {ml_score:.4f}
   Optimal Parameters:
   - N Estimators: {ml_params.get('n_estimators', 'N/A')}
   - Max Depth: {ml_params.get('max_depth', 'N/A')}
   - Learning Rate: {ml_params.get('learning_rate', 'N/A')}
   - Subsample: {ml_params.get('subsample', 'N/A')}
   - Colsample By Tree: {ml_params.get('colsample_bytree', 'N/A')}

💡 Recommendations:
1. Use optimized parameters for production models
2. Monitor performance with real market data
3. Re-optimize periodically with new data
4. Consider ensemble methods for better stability

🎯 Next Steps:
1. Train final models with optimized parameters
2. Deploy to live trading system
3. Monitor performance in real-time
4. Schedule regular re-optimization
"""

        # Save report
        with open('optimization_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        self.logger.info("📄 Optimization report saved to: optimization_report.txt")
        print(report)


def main():
    """Main optimization function."""
    print("🚀 MR BEN Parameter Optimization")
    print("=" * 50)

    optimizer = ParameterOptimizer()

    # Load data
    print("📊 Loading data for optimization...")

    # Try to load existing data
    data_files = [
        'data/lstm_signals_features.csv',
        'data/mrben_ai_signal_dataset.csv',
        'data/lstm_train_data.csv',
    ]

    data = None
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                data = pd.read_csv(file_path)
                print(f"✅ Loaded data from: {file_path}")
                break
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

    if data is None:
        print("❌ No suitable data found for optimization")
        return

    # Prepare data for optimization
    print("🔧 Preparing data for optimization...")

    # This is a simplified example - you'll need to adapt based on your data structure
    try:
        # For LSTM optimization
        if 'signal' in data.columns:
            # Prepare LSTM data
            features = ['open', 'high', 'low', 'close', 'tick_volume']
            available_features = [f for f in features if f in data.columns]

            if len(available_features) >= 4:
                X = data[available_features].values
                y = data['signal'].values

                # Split data
                split_idx = int(0.8 * len(X))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                # Reshape for LSTM
                timesteps = 50
                X_train_lstm = []
                y_train_lstm = []

                for i in range(timesteps, len(X_train)):
                    X_train_lstm.append(X_train[i - timesteps : i])
                    y_train_lstm.append(y_train[i])

                X_val_lstm = []
                y_val_lstm = []

                for i in range(timesteps, len(X_val)):
                    X_val_lstm.append(X_val[i - timesteps : i])
                    y_val_lstm.append(y_val[i])

                X_train_lstm = np.array(X_train_lstm)
                X_val_lstm = np.array(X_val_lstm)
                y_train_lstm = np.array(y_train_lstm)
                y_val_lstm = np.array(y_val_lstm)

                print(f"✅ LSTM data prepared: {X_train_lstm.shape}, {y_train_lstm.shape}")

                # Optimize LSTM parameters
                lstm_params, lstm_score = optimizer.optimize_lstm_parameters(
                    X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm
                )

                # Optimize ML Filter parameters (simplified)
                ml_params, ml_score = optimizer.optimize_ml_filter_parameters(X_train, y_train)

                # Generate report
                optimizer.generate_optimization_report(lstm_params, lstm_score, ml_params, ml_score)

            else:
                print("❌ Insufficient features for LSTM optimization")

        else:
            print("❌ No signal column found in data")

    except Exception as e:
        print(f"❌ Error in optimization: {e}")


if __name__ == "__main__":
    main()
