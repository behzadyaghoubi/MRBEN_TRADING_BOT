"""
Strategy scorer for adaptive confidence thresholds.

Combines model prediction confidence with market regime awareness to provide
adaptive decision thresholds that adjust based on market conditions.
"""

import numpy as np
from typing import Dict, Optional, Union, Any
from pydantic import BaseModel, Field
import logging

from ..ai.regime import RegimeLabel, RegimeSnapshot

logger = logging.getLogger(__name__)


class AdaptiveDecision(BaseModel):
    """Result of adaptive confidence scoring."""
    raw_conf: float = Field(..., description="Raw model confidence")
    regime: RegimeLabel = Field(..., description="Current market regime")
    adj_conf: float = Field(..., description="Adjusted confidence after regime consideration")
    threshold: float = Field(..., description="Final decision threshold")
    allow_trade: bool = Field(..., description="Whether trade is allowed")
    notes: str = Field("", description="Additional notes about the decision")
    
    # Regime-specific adjustments
    trend_bonus: float = Field(0.0, description="Confidence bonus for trend regime")
    chop_penalty: float = Field(0.0, description="Confidence penalty for range/chop regime")
    vol_penalty: float = Field(0.0, description="Confidence penalty for high volatility")
    session_filter: str = Field("", description="Session filter applied")
    
    class Config:
        arbitrary_types_allowed = True


class AdaptiveConfig(BaseModel):
    """Configuration for adaptive confidence scoring."""
    base_conf: float = Field(0.60, description="Base confidence threshold")
    trend_bonus: float = Field(0.10, description="Confidence bonus for trend regime")
    chop_penalty: float = Field(0.15, description="Confidence penalty for range/chop regime")
    high_vol_penalty: float = Field(0.10, description="Confidence penalty for high volatility")
    
    # Session filters
    session_filters: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: {
            "london": {"open": "07:00", "close": "11:00"},
            "ny": {"open": "12:30", "close": "16:00"}
        }
    )
    
    # Confidence bounds
    min_conf_floor: float = Field(0.50, description="Minimum confidence floor")
    max_conf_cap: float = Field(0.90, description="Maximum confidence cap")
    
    # Regime-specific thresholds
    regime_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "trend": 0.65,
            "range": 0.75,
            "high_vol": 0.80,
            "illiquid": 0.85,
            "unknown": 0.70
        }
    )
    
    # Position size constraints by regime
    position_limits: Dict[str, float] = Field(
        default_factory=lambda: {
            "trend": 1.0,      # Full position size
            "range": 0.5,      # Half position size
            "high_vol": 0.3,   # Reduced position size
            "illiquid": 0.1,   # Minimal position size
            "unknown": 0.7     # Moderate position size
        }
    )


class AdaptiveScorer:
    """
    Adaptive confidence scorer that adjusts thresholds based on market regime.
    """
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """
        Initialize adaptive scorer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or AdaptiveConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not (0 <= self.config.base_conf <= 1):
            raise ValueError("Base confidence must be between 0 and 1")
        
        if not (0 <= self.config.min_conf_floor <= 1):
            raise ValueError("Min confidence floor must be between 0 and 1")
        
        if not (0 <= self.config.max_conf_cap <= 1):
            raise ValueError("Max confidence cap must be between 0 and 1")
        
        if self.config.min_conf_floor >= self.config.max_conf_cap:
            raise ValueError("Min confidence floor must be less than max confidence cap")
        
        # Validate regime thresholds
        for regime, threshold in self.config.regime_thresholds.items():
            if not (0 <= threshold <= 1):
                raise ValueError(f"Regime threshold for {regime} must be between 0 and 1")
        
        # Validate position limits
        for regime, limit in self.config.position_limits.items():
            if not (0 <= limit <= 1):
                raise ValueError(f"Position limit for {regime} must be between 0 and 1")
    
    def adapt_confidence(self, raw_conf: float, regime: RegimeSnapshot) -> AdaptiveDecision:
        """
        Adapt confidence based on market regime.
        
        Args:
            raw_conf: Raw model confidence (0-1)
            regime: Current market regime snapshot
        
        Returns:
            AdaptiveDecision with adjusted confidence and thresholds
        """
        if not (0 <= raw_conf <= 1):
            raise ValueError("Raw confidence must be between 0 and 1")
        
        # Start with base confidence
        adj_conf = self.config.base_conf
        
        # Apply regime-specific adjustments
        trend_bonus = 0.0
        chop_penalty = 0.0
        vol_penalty = 0.0
        session_filter = ""
        
        if regime.label == RegimeLabel.TREND:
            trend_bonus = self.config.trend_bonus
            adj_conf += trend_bonus
            logger.debug(f"Trend regime: applying +{trend_bonus} bonus")
        
        elif regime.label == RegimeLabel.RANGE:
            chop_penalty = self.config.chop_penalty
            adj_conf -= chop_penalty
            logger.debug(f"Range regime: applying -{chop_penalty} penalty")
        
        elif regime.label == RegimeLabel.HIGH_VOL:
            vol_penalty = self.config.high_vol_penalty
            adj_conf -= vol_penalty
            logger.debug(f"High volatility regime: applying -{vol_penalty} penalty")
        
        elif regime.label == RegimeLabel.ILLIQUID:
            # Severe penalty for illiquid conditions
            illiquid_penalty = 0.25
            adj_conf -= illiquid_penalty
            logger.debug(f"Illiquid regime: applying -{illiquid_penalty} penalty")
        
        # Apply session filters if available
        if regime.session in self.config.session_filters:
            session_filter = f"session_{regime.session}"
            # Additional penalty for non-optimal sessions
            if regime.session not in ["london", "ny"]:
                session_penalty = 0.05
                adj_conf -= session_penalty
                logger.debug(f"Non-optimal session {regime.session}: applying -{session_penalty} penalty")
        
        # Apply confidence bounds
        adj_conf = max(adj_conf, self.config.min_conf_floor)
        adj_conf = min(adj_conf, self.config.max_conf_cap)
        
        # Get regime-specific threshold
        regime_threshold = self.config.regime_thresholds.get(regime.label.value, self.config.base_conf)
        
        # Final threshold is the maximum of adjusted confidence and regime threshold
        final_threshold = max(adj_conf, regime_threshold)
        
        # Determine if trade is allowed
        allow_trade = raw_conf >= final_threshold
        
        # Generate notes
        notes = self._generate_notes(regime, adj_conf, final_threshold, allow_trade)
        
        return AdaptiveDecision(
            raw_conf=raw_conf,
            regime=regime.label,
            adj_conf=round(adj_conf, 3),
            threshold=round(final_threshold, 3),
            allow_trade=allow_trade,
            notes=notes,
            trend_bonus=round(trend_bonus, 3),
            chop_penalty=round(chop_penalty, 3),
            vol_penalty=round(vol_penalty, 3),
            session_filter=session_filter
        )
    
    def _generate_notes(self, regime: RegimeSnapshot, adj_conf: float, 
                        threshold: float, allow_trade: bool) -> str:
        """Generate human-readable notes about the decision."""
        notes_parts = []
        
        # Regime information
        notes_parts.append(f"Regime: {regime.label.value} (conf: {regime.confidence:.2f})")
        
        # Confidence adjustments
        if adj_conf != self.config.base_conf:
            notes_parts.append(f"Adjusted from {self.config.base_conf:.2f} to {adj_conf:.2f}")
        
        # Threshold comparison
        if threshold > adj_conf:
            notes_parts.append(f"Regime threshold {threshold:.2f} applied")
        
        # Trade decision
        if allow_trade:
            notes_parts.append("Trade allowed")
        else:
            notes_parts.append(f"Trade blocked: {regime.label.value} regime requires {threshold:.2f} confidence")
        
        # Session information
        if regime.session:
            notes_parts.append(f"Session: {regime.session}")
        
        return "; ".join(notes_parts)
    
    def get_position_size_multiplier(self, regime: RegimeLabel) -> float:
        """
        Get position size multiplier for current regime.
        
        Args:
            regime: Current market regime
        
        Returns:
            Position size multiplier (0-1)
        """
        return self.config.position_limits.get(regime.value, 0.7)
    
    def get_regime_summary(self) -> Dict:
        """Get summary of regime-specific thresholds and limits."""
        return {
            "base_confidence": self.config.base_conf,
            "regime_thresholds": self.config.regime_thresholds,
            "position_limits": self.config.position_limits,
            "adjustments": {
                "trend_bonus": self.config.trend_bonus,
                "chop_penalty": self.config.chop_penalty,
                "high_vol_penalty": self.config.high_vol_penalty
            },
            "bounds": {
                "min_floor": self.config.min_conf_floor,
                "max_cap": self.config.max_conf_cap
            }
        }
    
    def update_config(self, new_config: Dict):
        """
        Update configuration parameters.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        # Create new config object
        updated_config = AdaptiveConfig(**{**self.config.dict(), **new_config})
        
        # Validate new config
        updated_config._validate_config()
        
        # Update current config
        self.config = updated_config
        logger.info("Adaptive scorer configuration updated")
    
    def reset(self):
        """Reset scorer to default configuration."""
        self.config = AdaptiveConfig()
        logger.info("Adaptive scorer reset to default configuration")


def create_adaptive_scorer(config_dict: Optional[Dict] = None) -> AdaptiveScorer:
    """
    Factory function to create an adaptive scorer.
    
    Args:
        config_dict: Optional configuration dictionary
    
    Returns:
        Configured AdaptiveScorer instance
    """
    if config_dict is None:
        config = AdaptiveConfig()
    else:
        config = AdaptiveConfig(**config_dict)
    
    return AdaptiveScorer(config)


# Convenience functions for quick scoring
def quick_adapt_confidence(raw_conf: float, regime_label: str, 
                          config_dict: Optional[Dict] = None) -> AdaptiveDecision:
    """
    Quick confidence adaptation without full regime snapshot.
    
    Args:
        raw_conf: Raw confidence (0-1)
        regime_label: Regime label string
        config_dict: Optional configuration
    
    Returns:
        AdaptiveDecision
    """
    # Create minimal regime snapshot
    import pandas as pd
    from ..ai.regime import RegimeSnapshot, RegimeLabel
    regime = RegimeSnapshot(
        label=RegimeLabel(regime_label),
        scores={},
        session="unknown",
        ts=pd.Timestamp.now(),
        confidence=0.8
    )
    
    scorer = create_adaptive_scorer(config_dict)
    return scorer.adapt_confidence(raw_conf, regime)


def get_regime_penalty(regime_label: str) -> float:
    """
    Get the confidence penalty for a specific regime.
    
    Args:
        regime_label: Regime label string
    
    Returns:
        Penalty value (positive number)
    """
    penalties = {
        "trend": 0.0,      # No penalty
        "range": 0.15,     # Chop penalty
        "high_vol": 0.10,  # Volatility penalty
        "illiquid": 0.25,  # Severe penalty
        "unknown": 0.05    # Small penalty
    }
    
    return penalties.get(regime_label, 0.05)


# Standalone function for easy import
def adapt_confidence(raw_conf: float, regime_label: Union[str, RegimeLabel], 
                     config_dict: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Standalone function to adapt confidence based on regime.
    
    Args:
        raw_conf: Raw model confidence (0.0 to 1.0)
        regime_label: Regime label string or RegimeLabel enum
        config_dict: Optional configuration dictionary
    
    Returns:
        Dictionary with adjusted confidence and decision
    """
    try:
        # Convert string to RegimeLabel if needed
        if isinstance(regime_label, str):
            try:
                regime_label = RegimeLabel(regime_label)
            except ValueError:
                regime_label = RegimeLabel.UNKNOWN
        
        # Create scorer with config
        config = AdaptiveConfig(**(config_dict or {}))
        scorer = AdaptiveScorer(config)
        
        # Create a minimal RegimeSnapshot for compatibility
        from ..ai.regime import RegimeSnapshot
        import pandas as pd
        
        dummy_snapshot = RegimeSnapshot(
            label=regime_label,
            scores={},
            session="unknown",
            ts=pd.Timestamp.now(),
            confidence=0.8
        )
        
        # Get decision
        decision = scorer.adapt_confidence(raw_conf, dummy_snapshot)
        
        return {
            'adj_conf': decision.adj_conf,
            'threshold': decision.threshold,
            'allow_trade': decision.allow_trade,
            'regime': regime_label.value,
            'notes': decision.notes
        }
        
    except Exception as e:
        logger.warning(f"Confidence adaptation failed: {e}, using fallback")
        # Fallback: basic regime-based adjustment
        if regime_label in [RegimeLabel.TREND, "trend"]:
            adj_conf = min(raw_conf * 1.1, 0.9)
            threshold = 0.65
        elif regime_label in [RegimeLabel.RANGE, "range"]:
            adj_conf = max(raw_conf * 0.9, 0.5)
            threshold = 0.75
        elif regime_label in [RegimeLabel.HIGH_VOL, "high_vol"]:
            adj_conf = max(raw_conf * 0.8, 0.5)
            threshold = 0.80
        elif regime_label in [RegimeLabel.ILLIQUID, "illiquid"]:
            adj_conf = max(raw_conf * 0.7, 0.5)
            threshold = 0.85
        else:
            adj_conf = raw_conf
            threshold = 0.70
        
        return {
            'adj_conf': adj_conf,
            'threshold': threshold,
            'allow_trade': adj_conf >= threshold,
            'regime': str(regime_label),
            'notes': 'fallback calculation'
        }
