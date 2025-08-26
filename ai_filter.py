import logging
import os
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

try:
    import pickle
except ImportError:
    pickle = None

try:
    from tensorflow import keras
except ImportError:
    keras = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# -------- Logging Config --------
logger = logging.getLogger("AI_Filter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
    logger.addHandler(ch)


class AISignalFilter:
    """
    Professional AI signal filter with feature shape check and robust error reporting.
    Supports: joblib (scikit-learn/XGBoost), pickle, keras, xgboost native.
    """

    def __init__(self, model_path: str, model_type: str = "joblib", threshold: float = 0.5):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.threshold = threshold
        self.model = self._load_model()
        self.feature_names = []
        self.feature_count = None
        self._extract_feature_names()
        logger.info(
            f"AI Filter Initialized | Model: {self.model_type} | Threshold: {self.threshold}"
        )

    def _load_model(self) -> Any:
        logger.info(f"Loading model: {self.model_path} (type={self.model_type})")
        if self.model_type == "joblib":
            if not joblib:
                raise ImportError("joblib is not installed.")
            return joblib.load(self.model_path)
        elif self.model_type == "pickle":
            if not pickle:
                raise ImportError("pickle is not installed.")
            with open(self.model_path, "rb") as f:
                return pickle.load(f)
        elif self.model_type == "keras":
            if not keras:
                raise ImportError("keras is not installed.")
            return keras.models.load_model(self.model_path)
        elif self.model_type == "xgboost":
            if not xgb:
                raise ImportError("xgboost is not installed.")
            model = xgb.Booster()
            model.load_model(self.model_path)
            return model
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _extract_feature_names(self):
        """Try to extract feature names/count if model supports it."""
        try:
            if hasattr(self.model, "feature_names_in_"):
                self.feature_names = list(self.model.feature_names_in_)
                self.feature_count = len(self.feature_names)
            elif hasattr(self.model, "feature_importances_"):
                self.feature_count = len(self.model.feature_importances_)
                self.feature_names = [f"feature_{i}" for i in range(self.feature_count)]
            elif hasattr(self.model, "n_features_in_"):
                self.feature_count = int(self.model.n_features_in_)
            else:
                self.feature_count = None
        except Exception as e:
            logger.warning(f"Feature names extraction failed: {e}")

    def _prepare_features(self, X: dict | list | np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Converts X into correct shape for model input.
        Handles: dict, list, 1d/2d numpy, DataFrame.
        """
        # Dict (feature_name: value) -> array in correct order
        if isinstance(X, dict):
            if self.feature_names:
                arr = np.array([[X.get(k, 0) for k in self.feature_names]])
            else:
                arr = np.array([list(X.values())])
        elif isinstance(X, (list, tuple)):
            arr = np.array(X)
        elif isinstance(X, pd.DataFrame):
            arr = X.values
        elif isinstance(X, np.ndarray):
            arr = X
        else:
            raise ValueError("Unsupported feature input type")

        # 1D -> 2D
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Check shape!
        if self.feature_count is not None and arr.shape[1] != self.feature_count:
            logger.error(
                f"Feature shape mismatch, expected: {self.feature_count} ({self.feature_names if self.feature_names else ''}), got: {arr.shape[1]}"
            )
            raise ValueError(
                f"Feature shape mismatch, expected: {self.feature_count}, got: {arr.shape[1]}"
            )
        return arr

    def predict(
        self, X: pd.DataFrame | np.ndarray | list | dict, return_confidence: bool = False
    ) -> int | float:
        """
        Predict signal (0/1) or probability with the AI model.
        If return_confidence=True, returns probability/confidence (float)
        """
        arr = self._prepare_features(X)
        try:
            if self.model_type in ["joblib", "pickle"]:
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(arr)
                    confidence = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]
                    if return_confidence:
                        return float(confidence)
                    return int(confidence >= self.threshold)
                else:
                    pred = self.model.predict(arr)
                    if return_confidence:
                        return float(pred[0])
                    return int(pred[0])
            elif self.model_type == "keras":
                preds = self.model.predict(arr)
                confidence = preds[0][0] if preds.ndim > 1 else preds[0]
                if return_confidence:
                    return float(confidence)
                return int(confidence >= self.threshold)
            elif self.model_type == "xgboost":
                dmatrix = xgb.DMatrix(arr)
                preds = self.model.predict(dmatrix)
                confidence = preds[0]
                if return_confidence:
                    return float(confidence)
                return int(confidence >= self.threshold)
            else:
                raise ValueError("Unsupported model type for prediction.")
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0 if not return_confidence else 0.0

    def filter_signal(self, features: pd.DataFrame | np.ndarray | list | dict) -> int:
        """
        Passes signal through AI filter.
        Returns: 1 if signal passes, otherwise 0.
        """
        try:
            result = self.predict(features)
            logger.info(f"Signal filter result: {result}")
            return int(result)
        except Exception as e:
            logger.error(f"Signal filter failed: {e}")
            return 0

    def filter_signal_with_confidence(
        self, features: pd.DataFrame | np.ndarray | list | dict
    ) -> dict[str, int | float]:
        """
        Passes signal through AI filter and returns both prediction and confidence.
        Returns: {"prediction": int, "confidence": float}
        """
        try:
            prediction = self.filter_signal(features)
            confidence = self.get_confidence(features)

            return {"prediction": prediction, "confidence": confidence}
        except Exception as e:
            logger.error(f"Signal filter with confidence failed: {e}")
            return {"prediction": 0, "confidence": 0.0}

    def get_confidence(self, features) -> float:
        """Returns model probability/confidence score."""
        try:
            return float(self.predict(features, return_confidence=True))
        except Exception as e:
            logger.error(f"Confidence fetch failed: {e}")
            return 0.0

    def reload_model(self, new_path: str | None = None, new_type: str | None = None):
        """Reloads model from file (optionally with new path/type)."""
        if new_path:
            self.model_path = new_path
        if new_type:
            self.model_type = new_type.lower()
        self.model = self._load_model()
        self._extract_feature_names()
        logger.info("Model reloaded.")

    def set_threshold(self, new_threshold: float):
        self.threshold = float(new_threshold)
        logger.info(f"Threshold updated to {self.threshold}")

    def get_feature_importance(self) -> dict[str, float]:
        """Returns feature importances if available."""
        try:
            if hasattr(self.model, "feature_importances_") and self.feature_names:
                return dict(zip(self.feature_names, self.model.feature_importances_, strict=False))
            elif hasattr(self.model, "feature_importances_"):
                return {
                    f"feature_{i}": imp for i, imp in enumerate(self.model.feature_importances_)
                }
        except Exception as e:
            logger.warning(f"Cannot extract feature importance: {e}")
        return {}


# Example Usage
if __name__ == "__main__":
    MODEL_PATH = os.getenv("AI_FILTER_MODEL_PATH", "mrben_ai_signal_filter_xgb.joblib")
    MODEL_TYPE = os.getenv("AI_FILTER_MODEL_TYPE", "joblib")
    ai_filter = AISignalFilter(model_path=MODEL_PATH, model_type=MODEL_TYPE, threshold=0.55)

    # !!! توجه: حتماً سایز فیچر رو با مدل هماهنگ تست کن !!!
    test_features = np.random.rand(1, ai_filter.feature_count or 6)
    result = ai_filter.filter_signal(test_features)
    print("Signal passed AI filter:", result)
    print("Confidence score:", ai_filter.get_confidence(test_features))
