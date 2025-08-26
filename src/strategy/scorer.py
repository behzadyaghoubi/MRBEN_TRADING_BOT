#!/usr/bin/env python3
"""
MR BEN Strategy Scorer Module
"""

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RegimeLabel(Enum):
    """Market regime labels"""

    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"


def adapt_confidence(
    base_confidence: float, regime: RegimeLabel, config: dict[str, Any]
) -> dict[str, Any]:
    """
    Adapt confidence based on market regime

    Args:
        base_confidence: Base confidence score (0.0 to 1.0)
        regime: Market regime label
        config: Configuration dictionary with regime adjustments

    Returns:
        Dictionary with adjusted confidence and threshold
    """
    try:
        # Get regime-specific adjustments from config
        regime_adjustments = config.get(regime.value, {})

        # Default adjustments if not specified
        default_adjustments = {
            RegimeLabel.BULL_TREND.value: {"conf_mult": 1.1, "thr_add": -0.05},
            RegimeLabel.BEAR_TREND.value: {"conf_mult": 1.1, "thr_add": -0.05},
            RegimeLabel.SIDEWAYS.value: {"conf_mult": 0.9, "thr_add": 0.05},
            RegimeLabel.HIGH_VOL.value: {"conf_mult": 0.8, "thr_add": 0.1},
            RegimeLabel.LOW_VOL.value: {"conf_mult": 1.05, "thr_add": -0.05},
            RegimeLabel.NORMAL.value: {"conf_mult": 1.0, "thr_add": 0.0},
        }

        # Use config adjustments or defaults
        adjustments = regime_adjustments or default_adjustments.get(
            regime.value, {"conf_mult": 1.0, "thr_add": 0.0}
        )

        # Apply adjustments
        conf_mult = adjustments.get("conf_mult", 1.0)
        thr_add = adjustments.get("thr_add", 0.0)

        # Calculate adjusted values
        adjusted_confidence = base_confidence * conf_mult
        adjusted_threshold = 0.5 + thr_add  # Base threshold is 0.5

        # Ensure confidence stays within bounds
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        adjusted_threshold = max(0.1, min(0.9, adjusted_threshold))

        # Determine if trade should be allowed
        allow_trade = adjusted_confidence >= adjusted_threshold

        logger.info(
            f"Confidence adaptation: {base_confidence:.3f} -> {adjusted_confidence:.3f} "
            f"(regime: {regime.value}, mult: {conf_mult:.2f}, thr: {adjusted_threshold:.3f})"
        )

        return {
            'adj_conf': adjusted_confidence,
            'threshold': adjusted_threshold,
            'allow_trade': allow_trade,
            'regime': regime.value,
            'adjustments': adjustments,
        }

    except Exception as e:
        logger.error(f"Error in confidence adaptation: {e}")
        # Return safe fallback
        return {
            'adj_conf': base_confidence,
            'threshold': 0.5,
            'allow_trade': base_confidence >= 0.5,
            'regime': regime.value if hasattr(regime, 'value') else 'UNKNOWN',
            'adjustments': {'conf_mult': 1.0, 'thr_add': 0.0},
        }


def score_signal(signal_data: dict[str, Any], config: dict[str, Any]) -> float:
    """
    Score a trading signal based on multiple factors

    Args:
        signal_data: Dictionary containing signal information
        config: Configuration dictionary

    Returns:
        Signal score between 0.0 and 1.0
    """
    try:
        score = 0.0
        weights = config.get('weights', {})

        # Technical analysis score
        if 'technical_score' in signal_data:
            tech_score = signal_data['technical_score']
            tech_weight = weights.get('technical', 0.3)
            score += tech_score * tech_weight

        # Price action score
        if 'pa_score' in signal_data:
            pa_score = signal_data['pa_score']
            pa_weight = weights.get('price_action', 0.25)
            score += pa_score * pa_weight

        # Volume score
        if 'volume_score' in signal_data:
            vol_score = signal_data['volume_score']
            vol_weight = weights.get('volume', 0.2)
            score += vol_score * vol_weight

        # Trend score
        if 'trend_score' in signal_data:
            trend_score = signal_data['trend_score']
            trend_weight = weights.get('trend', 0.25)
            score += trend_score * trend_weight

        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))

        logger.debug(f"Signal scored: {score:.3f} from {signal_data}")
        return score

    except Exception as e:
        logger.error(f"Error in signal scoring: {e}")
        return 0.5  # Safe fallback score


def validate_signal(signal_data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a trading signal

    Args:
        signal_data: Dictionary containing signal information
        config: Configuration dictionary

    Returns:
        Validation result dictionary
    """
    try:
        validation_result = {'valid': False, 'score': 0.0, 'issues': [], 'warnings': []}

        # Check required fields
        required_fields = ['signal_type', 'confidence', 'symbol']
        for field in required_fields:
            if field not in signal_data:
                validation_result['issues'].append(f"Missing required field: {field}")

        # Check confidence bounds
        confidence = signal_data.get('confidence', 0.0)
        if not (0.0 <= confidence <= 1.0):
            validation_result['issues'].append(f"Invalid confidence: {confidence}")

        # Check signal type
        signal_type = signal_data.get('signal_type')
        if signal_type not in [-1, 0, 1]:
            validation_result['issues'].append(f"Invalid signal type: {signal_type}")

        # Calculate validation score
        if not validation_result['issues']:
            validation_result['valid'] = True
            validation_result['score'] = confidence

        # Add warnings for low confidence
        if confidence < 0.6:
            validation_result['warnings'].append(f"Low confidence: {confidence:.3f}")

        logger.debug(f"Signal validation: {validation_result}")
        return validation_result

    except Exception as e:
        logger.error(f"Error in signal validation: {e}")
        return {'valid': False, 'score': 0.0, 'issues': [f"Validation error: {e}"], 'warnings': []}
