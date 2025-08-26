#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN AutoML - ML Filter Retraining Script
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automl_ml.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLRetrainer:
    """ML Filter retraining with safe promotion"""
    
    def __init__(self, config_path: str = "config/pro_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.registry_path = "models/registry.json"
        self.registry = self._load_registry()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            else:
                return self._get_default_registry()
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return self._get_default_registry()
    
    def _get_default_registry(self) -> Dict[str, Any]:
        """Get default registry structure"""
        return {
            "ml": {
                "current": "models/ml_filter.pkl",
                "history": [],
                "last_retrain": None,
                "performance": {"current_auc": 0.0, "current_f1": 0.0, "current_calibration": 0.0}
            },
            "lstm": {
                "current": "models/lstm_model.h5",
                "history": [],
                "last_retrain": None,
                "performance": {"current_auc": 0.0, "current_f1": 0.0, "current_calibration": 0.0}
            }
        }
    
    def _prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training dataset"""
        try:
            # This would normally load from your data pipeline
            # For now, create synthetic data
            logger.info("Preparing training dataset...")
            
            # Generate synthetic features and labels
            n_samples = 10000
            n_features = 20
            
            # Random features
            X = np.random.randn(n_samples, n_features)
            
            # Simple rule-based labels (for demonstration)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
            
            logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            raise
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, StandardScaler, float, float]:
        """Train ML model and return model, scaler, and metrics"""
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Choose best algorithm
            models = {}
            scores = {}
            
            if XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                xgb_model.fit(X_train_scaled, y_train)
                xgb_pred = xgb_model.predict_proba(X_val_scaled)[:, 1]
                xgb_auc = roc_auc_score(y_val, xgb_pred)
                xgb_f1 = f1_score(y_val, xgb_model.predict(X_val_scaled))
                models['xgboost'] = xgb_model
                scores['xgboost'] = (xgb_auc, xgb_f1)
                logger.info(f"XGBoost - AUC: {xgb_auc:.3f}, F1: {xgb_f1:.3f}")
            
            if LIGHTGBM_AVAILABLE:
                lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                lgb_model.fit(X_train_scaled, y_train)
                lgb_pred = lgb_model.predict_proba(X_val_scaled)[:, 1]
                lgb_auc = roc_auc_score(y_val, lgb_pred)
                lgb_f1 = f1_score(y_val, lgb_model.predict(X_val_scaled))
                models['lightgbm'] = lgb_model
                scores['lightgbm'] = (lgb_auc, lgb_f1)
                logger.info(f"LightGBM - AUC: {lgb_auc:.3f}, F1: {lgb_f1:.3f}")
            
            if SKLEARN_AVAILABLE:
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train_scaled, y_train)
                rf_pred = rf_model.predict_proba(X_val_scaled)[:, 1]
                rf_auc = roc_auc_score(y_val, rf_pred)
                rf_f1 = f1_score(y_val, rf_model.predict(X_val_scaled))
                models['random_forest'] = rf_model
                scores['random_forest'] = (rf_auc, rf_f1)
                logger.info(f"Random Forest - AUC: {rf_auc:.3f}, F1: {rf_f1:.3f}")
            
            # Select best model
            best_model_name = max(scores.keys(), key=lambda k: scores[k][0])
            best_model = models[best_model_name]
            best_auc, best_f1 = scores[best_model_name]
            
            logger.info(f"Best model: {best_model_name} (AUC: {best_auc:.3f}, F1: {best_f1:.3f})")
            
            # Calibrate model
            calibrated_model = CalibratedClassifierCV(best_model, cv=5)
            calibrated_model.fit(X_train_scaled, y_train)
            
            return calibrated_model, scaler, best_auc, best_f1
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
    
    def _evaluate_model(self, model: Any, scaler: StandardScaler, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            X_scaled = scaler.transform(X)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            y_pred = model.predict(X_scaled)
            
            auc = roc_auc_score(y, y_pred_proba)
            f1 = f1_score(y, y_pred)
            
            # Simple calibration score (ratio of predicted vs actual positive rate)
            pred_pos_rate = y_pred.mean()
            actual_pos_rate = y.mean()
            calibration = 1.0 - abs(pred_pos_rate - actual_pos_rate) if actual_pos_rate > 0 else 1.0
            
            return {
                "auc": auc,
                "f1": f1,
                "calibration": calibration
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            return {"auc": 0.0, "f1": 0.0, "calibration": 0.0}
    
    def _should_promote(self, new_metrics: Dict[str, float]) -> bool:
        """Determine if new model should be promoted"""
        try:
            current_metrics = self.registry["ml"]["performance"]
            
            # Simple promotion criteria
            auc_improvement = new_metrics["auc"] - current_metrics["current_auc"]
            f1_improvement = new_metrics["f1"] - current_metrics["current_f1"]
            
            # Promote if significant improvement
            should_promote = (auc_improvement > 0.02) or (f1_improvement > 0.02)
            
            logger.info(f"Promotion decision: AUC improvement {auc_improvement:.3f}, F1 improvement {f1_improvement:.3f}")
            logger.info(f"Should promote: {should_promote}")
            
            return should_promote
            
        except Exception as e:
            logger.error(f"Failed to evaluate promotion: {e}")
            return False
    
    def _save_model(self, model: Any, scaler: StandardScaler, metrics: Dict[str, float]) -> str:
        """Save model with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Save model files
            model_path = f"models/ml_filter_{timestamp}.pkl"
            scaler_path = f"models/scaler_{timestamp}.pkl"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"Model saved: {model_path}")
            logger.info(f"Scaler saved: {scaler_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def _update_registry(self, model_path: str, metrics: Dict[str, float], promoted: bool):
        """Update model registry"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Add to history
            history_entry = {
                "timestamp": timestamp,
                "model_path": model_path,
                "metrics": metrics,
                "promoted": promoted
            }
            
            self.registry["ml"]["history"].append(history_entry)
            
            # Update current if promoted
            if promoted:
                self.registry["ml"]["current"] = model_path
                self.registry["ml"]["performance"] = metrics
                self.registry["ml"]["last_retrain"] = timestamp
                logger.info("Model promoted to current")
            else:
                self.registry["ml"]["last_retrain"] = timestamp
                logger.info("Model added to history (not promoted)")
            
            # Save registry
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            
            logger.info("Registry updated")
            
        except Exception as e:
            logger.error(f"Failed to update registry: {e}")
            raise
    
    def retrain(self) -> bool:
        """Main retraining pipeline"""
        try:
            logger.info("Starting ML model retraining...")
            
            # Check dependencies
            if not any([SKLEARN_AVAILABLE, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE]):
                logger.error("No ML libraries available")
                return False
            
            # Prepare dataset
            X, y = self._prepare_dataset()
            
            # Train model
            model, scaler, auc, f1 = self._train_model(X, y)
            
            # Evaluate
            metrics = self._evaluate_model(model, scaler, X, y)
            logger.info(f"Model evaluation: {metrics}")
            
            # Check if should promote
            should_promote = self._should_promote(metrics)
            
            # Save model
            model_path = self._save_model(model, scaler, metrics)
            
            # Update registry
            self._update_registry(model_path, metrics, should_promote)
            
            logger.info("ML retraining completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ML retraining failed: {e}")
            return False

def main():
    """Main function"""
    try:
        retrainer = MLRetrainer()
        success = retrainer.retrain()
        
        if success:
            print("✅ ML retraining completed successfully")
            sys.exit(0)
        else:
            print("❌ ML retraining failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
