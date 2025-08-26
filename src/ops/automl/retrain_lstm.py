#!/usr/bin/env python3
"""
MR BEN AutoML - LSTM Filter Retraining Script
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/automl_lstm.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LSTMRetrainer:
    """LSTM Filter retraining with safe promotion"""

    def __init__(self, config_path: str = "config/pro_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.registry_path = "models/registry.json"
        self.registry = self._load_registry()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration"""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _load_registry(self) -> dict[str, Any]:
        """Load model registry"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path) as f:
                    return json.load(f)
            else:
                return self._get_default_registry()
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return self._get_default_registry()

    def _get_default_registry(self) -> dict[str, Any]:
        """Get default registry structure"""
        return {
            "ml": {
                "current": "models/ml_filter.pkl",
                "history": [],
                "last_retrain": None,
                "performance": {"current_auc": 0.0, "current_f1": 0.0, "current_calibration": 0.0},
            },
            "lstm": {
                "current": "models/lstm_model.h5",
                "history": [],
                "last_retrain": None,
                "performance": {"current_auc": 0.0, "current_f1": 0.0, "current_calibration": 0.0},
            },
        }

    def _prepare_sequence_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare sequence training dataset"""
        try:
            # This would normally load from your data pipeline
            # For now, create synthetic sequence data
            logger.info("Preparing LSTM sequence dataset...")

            # Generate synthetic sequence data
            n_samples = 5000
            seq_len = 50
            n_features = 10

            # Random sequences
            X = np.random.randn(n_samples, seq_len, n_features)

            # Simple rule-based labels (for demonstration)
            # Label based on trend in first feature
            y = np.zeros(n_samples)
            for i in range(n_samples):
                trend = np.mean(X[i, -10:, 0]) - np.mean(X[i, :10, 0])
                y[i] = 1 if trend > 0 else 0

            logger.info(
                f"Sequence dataset prepared: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features"
            )
            return X, y

        except Exception as e:
            logger.error(f"Failed to prepare sequence dataset: {e}")
            raise

    def _build_lstm_model(self, seq_len: int, n_features: int) -> tf.keras.Model:
        """Build LSTM model architecture"""
        try:
            model = Sequential(
                [
                    LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dropout(0.1),
                    Dense(1, activation='sigmoid'),
                ]
            )

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'auc'],
            )

            logger.info("LSTM model built successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to build LSTM model: {e}")
            raise

    def _train_lstm(self, X: np.ndarray, y: np.ndarray) -> tuple[tf.keras.Model, float, float]:
        """Train LSTM model and return model and metrics"""
        try:
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Build model
            seq_len, n_features = X.shape[1], X.shape[2]
            model = self._build_lstm_model(seq_len, n_features)

            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )

            # Train model
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1,
            )

            # Evaluate
            y_pred_proba = model.predict(X_val).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate metrics
            auc = self._calculate_auc(y_val, y_pred_proba)
            f1 = self._calculate_f1(y_val, y_pred)

            logger.info(f"LSTM training completed - AUC: {auc:.3f}, F1: {f1:.3f}")
            return model, auc, f1

        except Exception as e:
            logger.error(f"Failed to train LSTM: {e}")
            raise

    def _calculate_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate AUC score"""
        try:
            from sklearn.metrics import roc_auc_score

            return roc_auc_score(y_true, y_pred_proba)
        except ImportError:
            # Fallback calculation
            return 0.5

    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score"""
        try:
            from sklearn.metrics import f1_score

            return f1_score(y_true, y_pred)
        except ImportError:
            # Fallback calculation
            return 0.5

    def _evaluate_lstm(
        self, model: tf.keras.Model, X: np.ndarray, y: np.ndarray
    ) -> dict[str, float]:
        """Evaluate LSTM model performance"""
        try:
            y_pred_proba = model.predict(X).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

            auc = self._calculate_auc(y, y_pred_proba)
            f1 = self._calculate_f1(y, y_pred)

            # Simple calibration score
            pred_pos_rate = y_pred.mean()
            actual_pos_rate = y.mean()
            calibration = 1.0 - abs(pred_pos_rate - actual_pos_rate) if actual_pos_rate > 0 else 1.0

            return {"auc": auc, "f1": f1, "calibration": calibration}

        except Exception as e:
            logger.error(f"Failed to evaluate LSTM: {e}")
            return {"auc": 0.0, "f1": 0.0, "calibration": 0.0}

    def _should_promote(self, new_metrics: dict[str, float]) -> bool:
        """Determine if new LSTM model should be promoted"""
        try:
            current_metrics = self.registry["lstm"]["performance"]

            # Simple promotion criteria
            auc_improvement = new_metrics["auc"] - current_metrics["current_auc"]
            f1_improvement = new_metrics["f1"] - current_metrics["current_f1"]

            # Promote if significant improvement
            should_promote = (auc_improvement > 0.02) or (f1_improvement > 0.02)

            logger.info(
                f"LSTM promotion decision: AUC improvement {auc_improvement:.3f}, F1 improvement {f1_improvement:.3f}"
            )
            logger.info(f"Should promote: {should_promote}")

            return should_promote

        except Exception as e:
            logger.error(f"Failed to evaluate LSTM promotion: {e}")
            return False

    def _save_lstm(self, model: tf.keras.Model, metrics: dict[str, float]) -> str:
        """Save LSTM model with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)

            # Save model
            model_path = f"models/lstm_model_{timestamp}.h5"
            model.save(model_path)

            logger.info(f"LSTM model saved: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Failed to save LSTM model: {e}")
            raise

    def _update_registry(self, model_path: str, metrics: dict[str, float], promoted: bool):
        """Update model registry for LSTM"""
        try:
            timestamp = datetime.now().isoformat()

            # Add to history
            history_entry = {
                "timestamp": timestamp,
                "model_path": model_path,
                "metrics": metrics,
                "promoted": promoted,
            }

            self.registry["lstm"]["history"].append(history_entry)

            # Update current if promoted
            if promoted:
                self.registry["lstm"]["current"] = model_path
                self.registry["lstm"]["performance"] = metrics
                self.registry["lstm"]["last_retrain"] = timestamp
                logger.info("LSTM model promoted to current")
            else:
                self.registry["lstm"]["last_retrain"] = timestamp
                logger.info("LSTM model added to history (not promoted)")

            # Save registry
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)

            logger.info("Registry updated for LSTM")

        except Exception as e:
            logger.error(f"Failed to update LSTM registry: {e}")
            raise

    def retrain(self) -> bool:
        """Main LSTM retraining pipeline"""
        try:
            logger.info("Starting LSTM model retraining...")

            # Check dependencies
            if not TENSORFLOW_AVAILABLE:
                logger.error("TensorFlow not available")
                return False

            # Prepare dataset
            X, y = self._prepare_sequence_dataset()

            # Train LSTM
            model, auc, f1 = self._train_lstm(X, y)

            # Evaluate
            metrics = self._evaluate_lstm(model, X, y)
            logger.info(f"LSTM evaluation: {metrics}")

            # Check if should promote
            should_promote = self._should_promote(metrics)

            # Save model
            model_path = self._save_lstm(model, metrics)

            # Update registry
            self._update_registry(model_path, metrics, should_promote)

            logger.info("LSTM retraining completed successfully")
            return True

        except Exception as e:
            logger.error(f"LSTM retraining failed: {e}")
            return False


def main():
    """Main function"""
    try:
        retrainer = LSTMRetrainer()
        success = retrainer.retrain()

        if success:
            print("✅ LSTM retraining completed successfully")
            sys.exit(0)
        else:
            print("❌ LSTM retraining failed")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
