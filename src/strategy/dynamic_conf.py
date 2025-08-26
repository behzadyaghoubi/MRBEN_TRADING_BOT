#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Pro Strategy - Dynamic Confidence Module
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class DynamicConfidence:
    """Dynamic confidence adjustment based on ATR, regime, and session"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy_cfg = config.get('strategy', {})
        self.session_cfg = config.get('session', {})
        
    def adjust_confidence(self, 
                         base_conf: float, 
                         df: pd.DataFrame, 
                         current_time: datetime = None) -> Tuple[float, float, Dict[str, Any]]:
        """Adjust confidence based on multiple factors"""
        try:
            if current_time is None:
                current_time = datetime.now()
            
            # Start with base confidence
            adjusted_conf = base_conf
            threshold = 0.5  # Default threshold
            reasons = {}
            
            # 1. ATR-based adjustment
            atr_conf = self._adjust_by_atr(base_conf, df)
            adjusted_conf = atr_conf['confidence']
            reasons['atr'] = atr_conf
            
            # 2. Session-based adjustment
            session_conf = self._adjust_by_session(adjusted_conf, current_time)
            adjusted_conf = session_conf['confidence']
            reasons['session'] = session_conf
            
            # 3. Regime-based adjustment
            regime_conf = self._adjust_by_regime(adjusted_conf, df)
            adjusted_conf = regime_conf['confidence']
            threshold += regime_conf['threshold_adjustment']
            reasons['regime'] = regime_conf
            
            # Ensure confidence stays within bounds
            min_conf = self.strategy_cfg.get('min_conf', 0.55)
            adjusted_conf = max(min_conf, min(1.0, adjusted_conf))
            
            return adjusted_conf, threshold, reasons
            
        except Exception as e:
            logger.error(f"Error in dynamic confidence adjustment: {e}")
            return base_conf, 0.5, {"error": str(e)}
    
    def _adjust_by_atr(self, base_conf: float, df: pd.DataFrame) -> Dict[str, Any]:
        """Adjust confidence based on ATR volatility"""
        try:
            atr_window = self.strategy_cfg.get('atr_window', 14)
            atr_scale = self.strategy_cfg.get('atr_conf_scale', [0.5, 0.9])
            
            if len(df) < atr_window:
                return {"confidence": base_conf, "atr_value": 0.0, "factor": 1.0}
            
            # Calculate ATR
            atr = self._calculate_atr(df, atr_window)
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
            
            # Get ATR percentiles from recent data
            if len(atr) >= 200:
                atr_percentiles = atr.tail(200)
            else:
                atr_percentiles = atr
            
            low_percentile = np.nanpercentile(atr_percentiles, 30)
            high_percentile = np.nanpercentile(atr_percentiles, 70)
            
            # Interpolate ATR factor
            if high_percentile > low_percentile:
                atr_factor = np.interp(current_atr, [low_percentile, high_percentile], atr_scale)
            else:
                atr_factor = 1.0
            
            adjusted_conf = base_conf * atr_factor
            
            return {
                "confidence": adjusted_conf,
                "atr_value": current_atr,
                "factor": atr_factor,
                "low_percentile": low_percentile,
                "high_percentile": high_percentile
            }
            
        except Exception as e:
            logger.error(f"Error in ATR adjustment: {e}")
            return {"confidence": base_conf, "atr_value": 0.0, "factor": 1.0}
    
    def _adjust_by_session(self, base_conf: float, current_time: datetime) -> Dict[str, Any]:
        """Adjust confidence based on trading session"""
        try:
            session_windows = self.session_cfg.get('windows', {})
            block_outside = self.session_cfg.get('block_outside', False)
            
            if not session_windows:
                return {"confidence": base_conf, "session": "UNKNOWN", "factor": 1.0}
            
            # Convert current time to UTC if needed
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=None)
            
            current_hour = current_time.hour
            current_minute = current_time.minute
            current_time_minutes = current_hour * 60 + current_minute
            
            # Check if current time is within any session window
            in_session = False
            session_name = "OUTSIDE"
            
            for session, window in session_windows.items():
                start_time = self._parse_time(window[0])
                end_time = self._parse_time(window[1])
                
                if start_time <= current_time_minutes <= end_time:
                    in_session = True
                    session_name = session
                    break
            
            # Apply session adjustment
            if in_session:
                factor = 1.0
                adjusted_conf = base_conf
            else:
                if block_outside:
                    factor = 0.0
                    adjusted_conf = 0.0
                else:
                    factor = 0.8  # Reduce confidence outside sessions
                    adjusted_conf = base_conf * factor
            
            return {
                "confidence": adjusted_conf,
                "session": session_name,
                "factor": factor,
                "in_session": in_session
            }
            
        except Exception as e:
            logger.error(f"Error in session adjustment: {e}")
            return {"confidence": base_conf, "session": "ERROR", "factor": 1.0}
    
    def _adjust_by_regime(self, base_conf: float, df: pd.DataFrame) -> Dict[str, Any]:
        """Adjust confidence based on market regime"""
        try:
            regime_config = self.strategy_cfg.get('regime', {})
            
            if not regime_config:
                return {"confidence": base_conf, "regime": "UNKNOWN", "factor": 1.0, "threshold_adjustment": 0.0}
            
            # Simple regime detection based on recent volatility
            if len(df) < 20:
                return {"confidence": base_conf, "regime": "UNKNOWN", "factor": 1.0, "threshold_adjustment": 0.0}
            
            # Calculate recent volatility
            returns = df['close'].pct_change().tail(20)
            volatility = returns.std()
            
            # Determine regime
            if volatility > 0.002:  # High volatility
                regime = "HIGH_VOL"
                regime_cfg = regime_config.get("HIGH_VOL", {"conf_mult": 0.8, "thr_add": 0.05})
            elif volatility < 0.0005:  # Low volatility
                regime = "LOW_VOL"
                regime_cfg = regime_config.get("LOW_VOL", {"conf_mult": 1.05, "thr_add": -0.05})
            else:
                regime = "NORMAL"
                regime_cfg = {"conf_mult": 1.0, "thr_add": 0.0}
            
            # Apply regime adjustments
            conf_mult = regime_cfg.get("conf_mult", 1.0)
            threshold_adjustment = regime_cfg.get("thr_add", 0.0)
            
            adjusted_conf = base_conf * conf_mult
            
            return {
                "confidence": adjusted_conf,
                "regime": regime,
                "factor": conf_mult,
                "threshold_adjustment": threshold_adjustment,
                "volatility": volatility
            }
            
        except Exception as e:
            logger.error(f"Error in regime adjustment: {e}")
            return {"confidence": base_conf, "regime": "ERROR", "factor": 1.0, "threshold_adjustment": 0.0}
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
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
    
    def _parse_time(self, time_str: str) -> int:
        """Parse time string (HH:MM) to minutes since midnight"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return hour * 60 + minute
        except Exception as e:
            logger.error(f"Error parsing time {time_str}: {e}")
            return 0
