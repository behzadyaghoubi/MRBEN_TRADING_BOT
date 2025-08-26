"""
Market regime detection features module.

Provides efficient rolling computations for market analysis including:
- ATR (Average True Range)
- ADX (Average Directional Index)
- Realized Volatility
- Rolling Z-scores
- Session tagging
- Spread calculations
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Type aliases for better readability
ArrayLike = Union[np.ndarray, pd.Series]
FloatArray = Union[np.ndarray, pd.Series]


def _validate_inputs(high: ArrayLike, low: ArrayLike, close: ArrayLike, 
                     n: int, name: str = "input") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and convert inputs to numpy arrays."""
    if n <= 0:
        raise ValueError(f"{name}: n must be positive, got {n}")
    
    # Convert to numpy arrays if needed
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    
    # Check shapes
    if not (high_arr.shape == low_arr.shape == close_arr.shape):
        raise ValueError(f"{name}: All inputs must have the same shape")
    
    if len(high_arr) < n:
        raise ValueError(f"{name}: Input length {len(high_arr)} must be >= n {n}")
    
    return high_arr, low_arr, close_arr


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices  
        close: Close prices
        n: Period for ATR calculation (default: 14)
    
    Returns:
        numpy.ndarray: ATR values
    """
    high_arr, low_arr, close_arr = _validate_inputs(high, low, close, n, "ATR")
    
    # Calculate True Range
    tr1 = high_arr - low_arr
    tr2 = np.abs(high_arr - np.roll(close_arr, 1))
    tr3 = np.abs(low_arr - np.roll(close_arr, 1))
    
    # True Range is the maximum of the three
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # First value has no previous close, set to high-low
    tr[0] = high_arr[0] - low_arr[0]
    
    # Calculate ATR using exponential moving average
    atr_values = np.zeros_like(tr)
    atr_values[0] = tr[0]
    
    alpha = 2.0 / (n + 1)
    for i in range(1, len(tr)):
        atr_values[i] = alpha * tr[i] + (1 - alpha) * atr_values[i-1]
    
    return atr_values


def adx(high: ArrayLike, low: ArrayLike, close: ArrayLike, n: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        n: Period for ADX calculation (default: 14)
    
    Returns:
        numpy.ndarray: ADX values
    """
    high_arr, low_arr, close_arr = _validate_inputs(high, low, close, n, "ADX")
    
    # Calculate Directional Movement
    dm_plus = np.zeros_like(high_arr)
    dm_minus = np.zeros_like(low_arr)
    
    for i in range(1, len(high_arr)):
        high_diff = high_arr[i] - high_arr[i-1]
        low_diff = low_arr[i-1] - low_arr[i]
        
        if high_diff > low_diff and high_diff > 0:
            dm_plus[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            dm_minus[i] = low_diff
    
    # Calculate True Range
    tr = atr(high_arr, low_arr, close_arr, n)
    
    # Smooth DM+ and DM- using EMA
    alpha = 2.0 / (n + 1)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    dm_plus_smooth[0] = dm_plus[0]
    dm_minus_smooth[0] = dm_minus[0]
    
    for i in range(1, len(dm_plus)):
        dm_plus_smooth[i] = alpha * dm_plus[i] + (1 - alpha) * dm_plus_smooth[i-1]
        dm_minus_smooth[i] = alpha * dm_minus[i] + (1 - alpha) * dm_minus_smooth[i-1]
    
    # Calculate +DI and -DI
    di_plus = 100 * dm_plus_smooth / tr
    di_minus = 100 * dm_minus_smooth / tr
    
    # Calculate DX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # Calculate ADX using EMA
    adx_values = np.zeros_like(dx)
    adx_values[0] = dx[0]
    
    for i in range(1, len(dx)):
        adx_values[i] = alpha * dx[i] + (1 - alpha) * adx_values[i-1]
    
    return adx_values


def realized_vol(log_returns: ArrayLike, n: int = 60) -> np.ndarray:
    """
    Calculate realized volatility from log returns.
    
    Args:
        log_returns: Log returns
        n: Rolling window size (default: 60)
    
    Returns:
        numpy.ndarray: Realized volatility values
    """
    if n <= 0:
        raise ValueError(f"Realized volatility: n must be positive, got {n}")
    
    returns_arr = np.asarray(log_returns, dtype=np.float64)
    
    if len(returns_arr) < n:
        raise ValueError(f"Realized volatility: Input length {len(returns_arr)} must be >= n {n}")
    
    # Calculate rolling standard deviation
    vol = np.zeros_like(returns_arr)
    
    for i in range(n-1, len(returns_arr)):
        window = returns_arr[i-n+1:i+1]
        vol[i] = np.std(window) * np.sqrt(252)  # Annualized
    
    # Fill initial values with NaN or first calculated value
    vol[:n-1] = np.nan
    
    return vol


def rolling_z(x: ArrayLike, n: int = 20) -> np.ndarray:
    """
    Calculate rolling Z-score.
    
    Args:
        x: Input data
        n: Rolling window size (default: 20)
    
    Returns:
        numpy.ndarray: Z-score values
    """
    if n <= 0:
        raise ValueError(f"Rolling Z-score: n must be positive, got {n}")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    if len(x_arr) < n:
        raise ValueError(f"Rolling Z-score: Input length {len(x_arr)} must be >= n {n}")
    
    z_scores = np.zeros_like(x_arr)
    
    for i in range(n-1, len(x_arr)):
        window = x_arr[i-n+1:i+1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            z_scores[i] = (x_arr[i] - mean) / std
        else:
            z_scores[i] = 0
    
    # Fill initial values with NaN
    z_scores[:n-1] = np.nan
    
    return z_scores


def session_tag(timestamps: Union[pd.Series, pd.DatetimeIndex]) -> np.ndarray:
    """
    Tag timestamps with trading sessions.
    
    Args:
        timestamps: Pandas timestamps or DatetimeIndex
    
    Returns:
        numpy.ndarray: Session labels (0=closed, 1=asia, 2=london, 3=ny)
    """
    if isinstance(timestamps, pd.Series):
        ts = timestamps.values
    else:
        ts = timestamps
    
    # Convert to pandas DatetimeIndex if needed
    if not isinstance(ts, pd.DatetimeIndex):
        ts = pd.DatetimeIndex(ts)
    
    # Extract hour and minute
    hours = ts.hour
    minutes = ts.minute
    time_decimal = hours + minutes / 60.0
    
    # Define session boundaries (UTC)
    # Asia: 00:00-08:00 UTC
    # London: 07:00-15:00 UTC  
    # NY: 12:30-20:30 UTC
    # Closed: 20:30-00:00 UTC
    
    sessions = np.zeros(len(ts), dtype=int)
    
    # Asia session
    asia_mask = (time_decimal >= 0) & (time_decimal < 8)
    sessions[asia_mask] = 1
    
    # London session
    london_mask = (time_decimal >= 7) & (time_decimal < 15)
    sessions[london_mask] = 2
    
    # NY session
    ny_mask = (time_decimal >= 12.5) & (time_decimal < 20.5)
    sessions[ny_mask] = 3
    
    # Closed session (20:30-00:00)
    closed_mask = (time_decimal >= 20.5) | (time_decimal < 0)
    sessions[closed_mask] = 0
    
    return sessions


def spread_bp(bid: ArrayLike, ask: ArrayLike) -> np.ndarray:
    """
    Calculate spread in basis points.
    
    Args:
        bid: Bid prices
        ask: Ask prices
    
    Returns:
        numpy.ndarray: Spread in basis points
    """
    bid_arr = np.asarray(bid, dtype=np.float64)
    ask_arr = np.asarray(ask, dtype=np.float64)
    
    if bid_arr.shape != ask_arr.shape:
        raise ValueError("Spread calculation: bid and ask must have the same shape")
    
    # Calculate spread as percentage
    spread_pct = (ask_arr - bid_arr) / bid_arr
    
    # Convert to basis points (1 bp = 0.01%)
    spread_bp_values = spread_pct * 10000
    
    return spread_bp_values


def calculate_returns(close: ArrayLike) -> np.ndarray:
    """
    Calculate log returns from close prices.
    
    Args:
        close: Close prices
    
    Returns:
        numpy.ndarray: Log returns
    """
    close_arr = np.asarray(close, dtype=np.float64)
    
    if len(close_arr) < 2:
        raise ValueError("Returns calculation: Need at least 2 price points")
    
    # Calculate log returns
    log_returns = np.zeros_like(close_arr)
    log_returns[1:] = np.log(close_arr[1:] / close_arr[:-1])
    
    # First return is undefined
    log_returns[0] = np.nan
    
    return log_returns


def rolling_stats(x: ArrayLike, n: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate rolling mean, std, and percentile statistics.
    
    Args:
        x: Input data
        n: Rolling window size (default: 20)
    
    Returns:
        Tuple of (mean, std, p10) arrays
    """
    if n <= 0:
        raise ValueError(f"Rolling stats: n must be positive, got {n}")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    if len(x_arr) < n:
        raise ValueError(f"Rolling stats: Input length {len(x_arr)} must be >= n {n}")
    
    mean_vals = np.zeros_like(x_arr)
    std_vals = np.zeros_like(x_arr)
    p10_vals = np.zeros_like(x_arr)
    
    for i in range(n-1, len(x_arr)):
        window = x_arr[i-n+1:i+1]
        mean_vals[i] = np.mean(window)
        std_vals[i] = np.std(window)
        p10_vals[i] = np.percentile(window, 10)
    
    # Fill initial values with NaN
    mean_vals[:n-1] = np.nan
    std_vals[:n-1] = np.nan
    p10_vals[:n-1] = np.nan
    
    return mean_vals, std_vals, p10_vals
