#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Regime Detection Module
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RegimeLabel(Enum):
    """Market regime labels"""
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"

@dataclass
class RegimeSnapshot:
    """Market regime snapshot"""
    label: RegimeLabel
    confidence: float
    scores: Dict[str, float]
    timestamp: float

def infer_regime(bars: List[Dict[str, Any]], micro: Optional[Dict[str, Any]], config: Dict[str, Any]) -> RegimeSnapshot:
    """Infer market regime from price data"""
    try:
        if not bars or len(bars) < 20:
            return RegimeSnapshot(RegimeLabel.NORMAL, 0.5, {}, 0.0)
        
        # Simple regime detection based on price movement
        prices = [float(bar['close']) for bar in bars[-20:]]
        
        # Calculate trend
        trend_score = (prices[-1] - prices[0]) / prices[0]
        
        # Calculate volatility
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = sum(abs(r) for r in returns) / len(returns)
        
        # Determine regime
        if abs(trend_score) > 0.02:  # Strong trend
            if trend_score > 0:
                label = RegimeLabel.BULL_TREND
                confidence = min(0.9, 0.5 + abs(trend_score) * 10)
        else:
                label = RegimeLabel.BEAR_TREND
                confidence = min(0.9, 0.5 + abs(trend_score) * 10)
        elif volatility > 0.015:  # High volatility
            label = RegimeLabel.HIGH_VOL
            confidence = min(0.8, 0.5 + volatility * 20)
        elif volatility < 0.005:  # Low volatility
            label = RegimeLabel.LOW_VOL
            confidence = min(0.8, 0.5 + (0.01 - volatility) * 20)
            else:
            label = RegimeLabel.NORMAL
            confidence = 0.6
        
        scores = {
            "trend": trend_score,
            "volatility": volatility,
            "regime_confidence": confidence
        }
        
        return RegimeSnapshot(label, confidence, scores, 0.0)
        
    except Exception as e:
        logger.error(f"Error in regime detection: {e}")
        return RegimeSnapshot(RegimeLabel.NORMAL, 0.5, {"error": str(e)}, 0.0)
