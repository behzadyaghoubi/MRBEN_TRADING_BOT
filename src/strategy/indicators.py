"""
Technical Indicators for MR BEN Pro Strategy

Comprehensive collection of technical indicators used in professional trading:
- Moving Averages (SMA, EMA, WMA)
- Momentum (RSI, MACD, Stochastic)
- Volatility (ATR, Bollinger Bands)
- Volume (OBV, VWAP)
- Support/Resistance (Pivot Points)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Comprehensive technical indicators collection"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-12)  # Avoid division by zero
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(close, fast)
        ema_slow = TechnicalIndicators.ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-12))
        d_percent = TechnicalIndicators.sma(k_percent, d_period)
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return TechnicalIndicators.sma(tr, period)
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(close, period)
        std = close.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
        """Pivot Points (Standard)"""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def fibonacci_retracements(high: pd.Series, low: pd.Series, 
                              levels: list = [0.236, 0.382, 0.500, 0.618, 0.786]) -> dict:
        """Fibonacci Retracement Levels"""
        swing_high = high.max()
        swing_low = low.min()
        diff = swing_high - swing_low
        
        retracements = {}
        for level in levels:
            if swing_high > swing_low:  # Uptrend
                retracements[f'fib_{level}'] = swing_high - (diff * level)
            else:  # Downtrend
                retracements[f'fib_{level}'] = swing_low + (diff * level)
        
        return retracements
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smoothed values
        tr_smooth = TechnicalIndicators.sma(tr, period)
        dm_plus_smooth = TechnicalIndicators.sma(dm_plus, period)
        dm_minus_smooth = TechnicalIndicators.sma(dm_minus, period)
        
        # DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = TechnicalIndicators.sma(dx, period)
        
        return adx, di_plus, di_minus
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = TechnicalIndicators.sma(typical_price, period)
        mean_deviation = abs(typical_price - sma_tp).rolling(window=period).mean()
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-12))
        return williams_r
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                         volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = pd.Series(0.0, index=typical_price.index)
        negative_flow = pd.Series(0.0, index=typical_price.index)
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = TechnicalIndicators.sma(positive_flow, period)
        negative_mf = TechnicalIndicators.sma(negative_flow, period)
        
        mfi = 100 - (100 / (1 + (positive_mf / (negative_mf + 1e-12))))
        return mfi

# Convenience functions for common indicators
def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return TechnicalIndicators.sma(series, period)

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return TechnicalIndicators.ema(series, period)

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    return TechnicalIndicators.rsi(close, period)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    return TechnicalIndicators.atr(high, low, close, period)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence)"""
    return TechnicalIndicators.macd(close, fast, slow, signal)

def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    return TechnicalIndicators.bollinger_bands(close, period, std_dev)

def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """Pivot Points (Standard)"""
    return TechnicalIndicators.pivot_points(high, low, close)

def fibonacci_retracements(high: pd.Series, low: pd.Series, 
                          levels: list = [0.236, 0.382, 0.500, 0.618, 0.786]) -> dict:
    """Fibonacci Retracement Levels"""
    return TechnicalIndicators.fibonacci_retracements(high, low, levels)
