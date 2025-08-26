#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Pro Strategy - Price Action Validation Module
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class PAResult:
    """Price Action validation result"""
    ok: bool
    score: float
    tags: List[str]

def pa_validate(df: pd.DataFrame, side: int, cfg: Dict[str, Any]) -> PAResult:
    """Check Engulfing/Pin/Inside/Sweep around last 3-5 candles; return score/tags."""
    try:
        if len(df) < 5:
            return PAResult(ok=False, score=0.0, tags=["insufficient_data"])
        
        # Get recent candles
        recent = df.tail(5)
        
        # Calculate ATR for scaling
        atr = _calculate_atr(recent, window=14)
        atr_value = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.01
        
        tags = []
        score = 0.0
        
        # Check for Engulfing pattern
        if _check_engulfing(recent, side):
            tags.append("ENGULF")
            score += 0.3
        
        # Check for Pin bar pattern
        if _check_pin_bar(recent, side, atr_value):
            tags.append("PIN")
            score += 0.25
        
        # Check for Inside bar pattern
        if _check_inside_bar(recent):
            tags.append("INSIDE")
            score += 0.2
        
        # Check for Sweep pattern
        if _check_sweep(recent, side, atr_value):
            tags.append("SWEEP")
            score += 0.25
        
        # Determine if patterns are favorable
        ok = score >= 0.3 and len(tags) >= 1
        
        return PAResult(ok=ok, score=score, tags=tags)
        
    except Exception as e:
        logger.error(f"Error in PA validation: {e}")
        return PAResult(ok=False, score=0.0, tags=["error"])

def _calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series([0.01] * len(df))

def _check_engulfing(df: pd.DataFrame, side: int) -> bool:
    """Check for engulfing pattern"""
    try:
        if len(df) < 2:
            return False
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        if side == 1:  # BUY signal
            # Bullish engulfing: current body engulfs previous body
            current_body = current['close'] - current['open']
            prev_body = previous['close'] - previous['open']
            
            return (current_body > 0 and  # Current is bullish
                   prev_body < 0 and      # Previous is bearish
                   current['open'] < previous['close'] and
                   current['close'] > previous['open'])
        
        elif side == -1:  # SELL signal
            # Bearish engulfing: current body engulfs previous body
            current_body = current['close'] - current['open']
            prev_body = previous['close'] - previous['open']
            
            return (current_body < 0 and  # Current is bearish
                   prev_body > 0 and      # Previous is bullish
                   current['open'] > previous['close'] and
                   current['close'] < previous['open'])
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking engulfing: {e}")
        return False

def _check_pin_bar(df: pd.DataFrame, side: int, atr: float) -> bool:
    """Check for pin bar pattern"""
    try:
        if len(df) < 1:
            return False
        
        current = df.iloc[-1]
        
        # Calculate body and shadow sizes
        body_size = abs(current['close'] - current['open'])
        upper_shadow = current['high'] - max(current['open'], current['close'])
        lower_shadow = min(current['open'], current['close']) - current['low']
        
        # Pin bar criteria
        if side == 1:  # BUY signal - hammer
            return (lower_shadow >= 2 * body_size and
                   lower_shadow >= 0.5 * atr and
                   upper_shadow <= 0.5 * body_size)
        
        elif side == -1:  # SELL signal - shooting star
            return (upper_shadow >= 2 * body_size and
                   upper_shadow >= 0.5 * atr and
                   lower_shadow <= 0.5 * body_size)
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking pin bar: {e}")
        return False

def _check_inside_bar(df: pd.DataFrame) -> bool:
    """Check for inside bar pattern"""
    try:
        if len(df) < 2:
            return False
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Current high/low is inside previous high/low
        return (current['high'] <= previous['high'] and
               current['low'] >= previous['low'])
        
    except Exception as e:
        logger.error(f"Error checking inside bar: {e}")
        return False

def _check_sweep(df: pd.DataFrame, side: int, atr: float) -> bool:
    """Check for sweep pattern"""
    try:
        if len(df) < 3:
            return False
        
        current = df.iloc[-1]
        
        # Look for long shadow that sweeps previous levels
        if side == 1:  # BUY signal
            lower_shadow = min(current['open'], current['close']) - current['low']
            return (lower_shadow >= 1.5 * atr and
                   current['close'] > current['open'])  # Closes bullish
        
        elif side == -1:  # SELL signal
            upper_shadow = current['high'] - max(current['open'], current['close'])
            return (upper_shadow >= 1.5 * atr and
                   current['close'] < current['open'])  # Closes bearish
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking sweep: {e}")
        return False
