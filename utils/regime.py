import numpy as np
import pandas as pd

def detect_regime(row: pd.Series) -> str:
    """
    Lightweight regime detection based on technical indicators
    
    Args:
        row: pandas Series containing technical indicators
        
    Returns:
        str: One of "UPTREND", "DOWNTREND", "RANGE", or "UNKNOWN"
    """
    # بسیار سبک: slope SMA و فاصله macd
    sma20 = row.get('sma_20', np.nan)
    sma50 = row.get('sma_50', np.nan)
    macd = row.get('macd', np.nan)
    macds = row.get('macd_signal', np.nan)
    
    if pd.isna(sma20) or pd.isna(sma50) or pd.isna(macd) or pd.isna(macds):
        return "UNKNOWN"
    
    # Check for trending behavior based on MACD divergence
    trend = abs(macd - macds) > (row.get('close', 1.0) * 0.0008)
    
    if trend and sma20 > sma50:
        return "UPTREND"
    if trend and sma20 < sma50:
        return "DOWNTREND"
    return "RANGE"


def get_regime_multipliers(regime: str) -> dict:
    """
    Get risk/reward multipliers based on market regime
    
    Args:
        regime: Market regime string
        
    Returns:
        dict: Multipliers for different trading parameters
    """
    multipliers = {
        "UPTREND": {
            "confidence_boost": 1.1,
            "tp_multiplier": 1.2,
            "sl_multiplier": 0.9,
            "position_size": 1.1
        },
        "DOWNTREND": {
            "confidence_boost": 1.05,
            "tp_multiplier": 1.1,
            "sl_multiplier": 0.95,
            "position_size": 1.0
        },
        "RANGE": {
            "confidence_boost": 0.9,
            "tp_multiplier": 0.8,
            "sl_multiplier": 1.1,
            "position_size": 0.8
        },
        "UNKNOWN": {
            "confidence_boost": 0.8,
            "tp_multiplier": 1.0,
            "sl_multiplier": 1.0,
            "position_size": 0.7
        }
    }
    
    return multipliers.get(regime, multipliers["UNKNOWN"])


def detect_regime_from_indicators(close: float, sma_20: float, sma_50: float, 
                                macd: float, macd_signal: float, atr: float = None) -> str:
    """
    Direct regime detection from individual indicators (for convenience)
    
    Args:
        close: Current close price
        sma_20: 20-period simple moving average
        sma_50: 50-period simple moving average
        macd: MACD line value
        macd_signal: MACD signal line value
        atr: Average True Range (optional)
        
    Returns:
        str: Market regime
    """
    row = pd.Series({
        'close': close,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'macd': macd,
        'macd_signal': macd_signal,
        'atr': atr
    })
    
    return detect_regime(row)