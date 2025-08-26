import os
import json
import numpy as np
import joblib
from typing import Tuple, Dict, Optional

class ConformalGate:
    """
    Conformal Prediction filter for trading signals
    
    Provides statistically valid prediction intervals and accept/reject decisions
    based on conformal prediction theory.
    """
    
    def __init__(self, meta_model_path: str = "models/meta_filter.joblib", 
                 conf_path: str = "models/conformal.json"):
        """
        Initialize the Conformal Gate
        
        Args:
            meta_model_path: Path to the trained meta-model
            conf_path: Path to conformal calibration parameters
        """
        self.meta = None
        self.conf = None
        
        # Load meta-model
        if meta_model_path and os.path.exists(meta_model_path):
            try:
                self.meta = joblib.load(meta_model_path)
                print(f"✅ Loaded meta-model from {meta_model_path}")
            except Exception as e:
                print(f"❌ Failed to load meta-model: {e}")
                self.meta = None
        else:
            print(f"⚠️  Meta-model not found at {meta_model_path}")
        
        # Load conformal parameters
        if conf_path and os.path.exists(conf_path):
            try:
                with open(conf_path, "r") as f:
                    self.conf = json.load(f)
                print(f"✅ Loaded conformal parameters from {conf_path}")
            except Exception as e:
                print(f"❌ Failed to load conformal parameters: {e}")
                self.conf = None
        else:
            print(f"⚠️  Conformal parameters not found at {conf_path}")
    
    def is_available(self) -> bool:
        """Check if the conformal gate is properly loaded and available"""
        return self.meta is not None and self.conf is not None
    
    def accept(self, x_dict: Dict[str, float]) -> Tuple[bool, float, float]:
        """
        Apply conformal prediction to accept or reject a trading signal
        
        Args:
            x_dict: Dictionary of feature values
            
        Returns:
            Tuple of (accepted, probability, nonconformity_score)
        """
        if not self.is_available():
            # If conformal gate is not available, default to accept with low confidence
            return True, 0.5, 0.5
        
        try:
            # Extract features in the correct order
            feats = self.conf["features"]
            X = np.array([[x_dict.get(k, 0.0) for k in feats]], dtype=float)
            
            # Transform features using the scaler
            Xs = self.meta["scaler"].transform(X)
            
            # Get probability prediction
            p = float(self.meta["model"].predict_proba(Xs)[0, 1])
            
            # Calculate nonconformity score
            nonconf = 1.0 - p
            
            # Apply conformal threshold
            threshold = float(self.conf["nonconf_threshold"])
            accepted = nonconf <= threshold
            
            return accepted, p, nonconf
            
        except Exception as e:
            print(f"❌ Error in conformal prediction: {e}")
            # On error, default to reject for safety
            return False, 0.0, 1.0
    
    def get_prediction_interval(self, x_dict: Dict[str, float], alpha: float = None) -> Tuple[float, float]:
        """
        Get prediction interval for the given features
        
        Args:
            x_dict: Dictionary of feature values
            alpha: Significance level (if None, uses the calibrated alpha)
            
        Returns:
            Tuple of (lower_bound, upper_bound) for the prediction interval
        """
        if not self.is_available():
            return 0.0, 1.0
        
        if alpha is None:
            alpha = self.conf.get("alpha", 0.1)
        
        try:
            accepted, p, nonconf = self.accept(x_dict)
            
            # Calculate prediction interval based on conformal theory
            # This is a simplified implementation
            margin = 1.96 * np.sqrt(p * (1 - p))  # Approximate using normal theory
            
            lower_bound = max(0.0, p - margin)
            upper_bound = min(1.0, p + margin)
            
            return lower_bound, upper_bound
            
        except Exception as e:
            print(f"❌ Error calculating prediction interval: {e}")
            return 0.0, 1.0
    
    def get_confidence_level(self) -> float:
        """Get the confidence level (1 - alpha) of the conformal predictor"""
        if not self.is_available():
            return 0.5
        return 1.0 - self.conf.get("alpha", 0.1)
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics about the conformal predictor"""
        if not self.is_available():
            return {"confidence_level": 0.5, "threshold": 0.5, "features_count": 0}
        
        return {
            "confidence_level": self.get_confidence_level(),
            "threshold": self.conf.get("nonconf_threshold", 0.5),
            "alpha": self.conf.get("alpha", 0.1),
            "features_count": len(self.conf.get("features", []))
        }


class RegimeAwareConformalGate(ConformalGate):
    """
    Extended conformal gate that adapts thresholds based on market regime
    """
    
    def __init__(self, meta_model_path: str = "models/meta_filter.joblib", 
                 conf_path: str = "models/conformal.json"):
        super().__init__(meta_model_path, conf_path)
        
        # Regime-specific threshold adjustments
        self.regime_adjustments = {
            "UPTREND": 0.95,    # More lenient in uptrends
            "DOWNTREND": 1.05,  # Slightly more strict in downtrends  
            "RANGE": 1.1,       # More strict in ranging markets
            "UNKNOWN": 1.2      # Most strict when regime is unknown
        }
    
    def accept(self, x_dict: Dict[str, float], regime: str = "UNKNOWN") -> Tuple[bool, float, float]:
        """
        Apply regime-aware conformal prediction
        
        Args:
            x_dict: Dictionary of feature values
            regime: Current market regime
            
        Returns:
            Tuple of (accepted, probability, nonconformity_score)
        """
        if not self.is_available():
            return True, 0.5, 0.5
        
        # Get base conformal decision
        accepted, p, nonconf = super().accept(x_dict)
        
        # Apply regime-specific adjustment to threshold
        adjustment = self.regime_adjustments.get(regime, 1.0)
        adjusted_threshold = self.conf["nonconf_threshold"] * adjustment
        
        # Re-evaluate acceptance with adjusted threshold
        regime_accepted = nonconf <= adjusted_threshold
        
        return regime_accepted, p, nonconf