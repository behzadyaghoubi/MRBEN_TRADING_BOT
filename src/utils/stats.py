"""
Statistical utilities for market analysis.

Provides rolling statistics, z-scores, exponential weighted moving averages,
and other statistical functions used in regime detection and analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = Union[np.ndarray, pd.Series]
FloatArray = Union[np.ndarray, pd.Series]


def rolling_mean(x: ArrayLike, n: int) -> np.ndarray:
    """
    Calculate rolling mean over window n.
    
    Args:
        x: Input data
        n: Window size
    
    Returns:
        Rolling mean values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    if len(x_arr) < n:
        raise ValueError(f"Input length {len(x_arr)} must be >= window size {n}")
    
    result = np.full_like(x_arr, np.nan)
    
    for i in range(n-1, len(x_arr)):
        result[i] = np.mean(x_arr[i-n+1:i+1])
    
    return result


def rolling_std(x: ArrayLike, n: int) -> np.ndarray:
    """
    Calculate rolling standard deviation over window n.
    
    Args:
        x: Input data
        n: Window size
    
    Returns:
        Rolling standard deviation values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    if len(x_arr) < n:
        raise ValueError(f"Input length {len(x_arr)} must be >= window size {n}")
    
    result = np.full_like(x_arr, np.nan)
    
    for i in range(n-1, len(x_arr)):
        result[i] = np.std(x_arr[i-n+1:i+1])
    
    return result


def rolling_percentile(x: ArrayLike, n: int, percentile: float) -> np.ndarray:
    """
    Calculate rolling percentile over window n.
    
    Args:
        x: Input data
        n: Window size
        percentile: Percentile (0-100)
    
    Returns:
        Rolling percentile values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    if len(x_arr) < n:
        raise ValueError(f"Input length {len(x_arr)} must be >= window size {n}")
    
    result = np.full_like(x_arr, np.nan)
    
    for i in range(n-1, len(x_arr)):
        result[i] = np.percentile(x_arr[i-n+1:i+1], percentile)
    
    return result


def rolling_zscore(x: ArrayLike, n: int) -> np.ndarray:
    """
    Calculate rolling Z-score over window n.
    
    Args:
        x: Input data
        n: Window size
    
    Returns:
        Rolling Z-score values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    if len(x_arr) < n:
        raise ValueError(f"Input length {len(x_arr)} must be >= window size {n}")
    
    result = np.full_like(x_arr, np.nan)
    
    for i in range(n-1, len(x_arr)):
        window = x_arr[i-n+1:i+1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            result[i] = (x_arr[i] - mean) / std
        else:
            result[i] = 0.0
    
    return result


def ewma(x: ArrayLike, alpha: float, adjust: bool = True) -> np.ndarray:
    """
    Calculate exponential weighted moving average.
    
    Args:
        x: Input data
        alpha: Smoothing factor (0 < alpha < 1)
        adjust: Whether to adjust for bias
    
    Returns:
        EWMA values
    """
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    if len(x_arr) == 0:
        return x_arr
    
    result = np.zeros_like(x_arr)
    result[0] = x_arr[0]
    
    for i in range(1, len(x_arr)):
        result[i] = alpha * x_arr[i] + (1 - alpha) * result[i-1]
    
    if adjust:
        # Adjust for bias in early periods
        weights = np.array([(1 - alpha) ** i for i in range(len(x_arr))])
        weights = weights / weights.sum()
        result = result / weights
    
    return result


def rolling_skewness(x: ArrayLike, n: int) -> np.ndarray:
    """
    Calculate rolling skewness over window n.
    
    Args:
        x: Input data
        n: Window size
    
    Returns:
        Rolling skewness values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    if len(x_arr) < n:
        raise ValueError(f"Input length {len(x_arr)} must be >= window size {n}")
    
    result = np.full_like(x_arr, np.nan)
    
    for i in range(n-1, len(x_arr)):
        window = x_arr[i-n+1:i+1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            # Calculate skewness: E[(X-μ)³] / σ³
            skew = np.mean(((window - mean) / std) ** 3)
            result[i] = skew
        else:
            result[i] = 0.0
    
    return result


def rolling_kurtosis(x: ArrayLike, n: int) -> np.ndarray:
    """
    Calculate rolling kurtosis over window n.
    
    Args:
        x: Input data
        n: Window size
    
    Returns:
        Rolling kurtosis values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    if len(x_arr) < n:
        raise ValueError(f"Input length {len(x_arr)} must be >= window size {n}")
    
    result = np.full_like(x_arr, np.nan)
    
    for i in range(n-1, len(x_arr)):
        window = x_arr[i-n+1:i+1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            # Calculate kurtosis: E[(X-μ)⁴] / σ⁴
            kurt = np.mean(((window - mean) / std) ** 4)
            result[i] = kurt
        else:
            result[i] = 0.0
    
    return result


def rolling_correlation(x: ArrayLike, y: ArrayLike, n: int) -> np.ndarray:
    """
    Calculate rolling correlation between x and y over window n.
    
    Args:
        x: First time series
        y: Second time series
        n: Window size
    
    Returns:
        Rolling correlation values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    
    if len(x_arr) < n:
        raise ValueError(f"Input length {len(x_arr)} must be >= window size {n}")
    
    result = np.full_like(x_arr, np.nan)
    
    for i in range(n-1, len(x_arr)):
        x_window = x_arr[i-n+1:i+1]
        y_window = y_arr[i-n+1:i+1]
        
        # Calculate correlation
        corr = np.corrcoef(x_window, y_window)[0, 1]
        result[i] = corr if not np.isnan(corr) else 0.0
    
    return result


def rolling_beta(x: ArrayLike, y: ArrayLike, n: int) -> np.ndarray:
    """
    Calculate rolling beta (slope) between x and y over window n.
    
    Args:
        x: Independent variable
        y: Dependent variable
        n: Window size
    
    Returns:
        Rolling beta values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    
    if len(x_arr) < n:
        raise ValueError(f"Input length {len(x_arr)} must be >= window size {n}")
    
    result = np.full_like(x_arr, np.nan)
    
    for i in range(n-1, len(x_arr)):
        x_window = x_arr[i-n+1:i+1]
        y_window = y_arr[i-n+1:i+1]
        
        # Calculate beta using linear regression
        try:
            # Add constant term for regression
            X = np.column_stack([np.ones(n), x_window])
            beta = np.linalg.lstsq(X, y_window, rcond=None)[0]
            result[i] = beta[1]  # Slope coefficient
        except np.linalg.LinAlgError:
            result[i] = 0.0
    
    return result


def rolling_sharpe(returns: ArrayLike, n: int, risk_free_rate: float = 0.0) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio over window n.
    
    Args:
        returns: Return series
        n: Window size
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Rolling Sharpe ratio values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    returns_arr = np.asarray(returns, dtype=np.float64)
    
    if len(returns_arr) < n:
        raise ValueError(f"Input length {len(returns_arr)} must be >= window size {n}")
    
    result = np.full_like(returns_arr, np.nan)
    
    for i in range(n-1, len(returns_arr)):
        window = returns_arr[i-n+1:i+1]
        mean_return = np.mean(window)
        std_return = np.std(window)
        
        if std_return > 0:
            # Annualize (assuming daily returns)
            sharpe = (mean_return * 252 - risk_free_rate) / (std_return * np.sqrt(252))
            result[i] = sharpe
        else:
            result[i] = 0.0
    
    return result


def rolling_max_drawdown(returns: ArrayLike, n: int) -> np.ndarray:
    """
    Calculate rolling maximum drawdown over window n.
    
    Args:
        returns: Return series
        n: Window size
    
    Returns:
        Rolling maximum drawdown values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    returns_arr = np.asarray(returns, dtype=np.float64)
    
    if len(returns_arr) < n:
        raise ValueError(f"Input length {len(returns_arr)} must be >= window size {n}")
    
    result = np.full_like(returns_arr, np.nan)
    
    for i in range(n-1, len(returns_arr)):
        window = returns_arr[i-n+1:i+1]
        
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + window)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        # Get maximum drawdown
        max_dd = np.min(drawdown)
        result[i] = max_dd
    
    return result


def rolling_volatility(returns: ArrayLike, n: int, annualize: bool = True) -> np.ndarray:
    """
    Calculate rolling volatility over window n.
    
    Args:
        returns: Return series
        n: Window size
        annualize: Whether to annualize the volatility
    
    Returns:
        Rolling volatility values
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    returns_arr = np.asarray(returns, dtype=np.float64)
    
    if len(returns_arr) < n:
        raise ValueError(f"Input length {len(returns_arr)} must be >= window size {n}")
    
    result = np.full_like(returns_arr, np.nan)
    
    for i in range(n-1, len(returns_arr)):
        window = returns_arr[i-n+1:i+1]
        vol = np.std(window)
        
        if annualize:
            vol *= np.sqrt(252)  # Annualize assuming daily returns
        
        result[i] = vol
    
    return result


def rolling_stats_summary(x: ArrayLike, n: int) -> Dict[str, np.ndarray]:
    """
    Calculate comprehensive rolling statistics over window n.
    
    Args:
        x: Input data
        n: Window size
    
    Returns:
        Dictionary with various rolling statistics
    """
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    if len(x_arr) < n:
        raise ValueError(f"Input length {len(x_arr)} must be >= window size {n}")
    
    return {
        "mean": rolling_mean(x_arr, n),
        "std": rolling_std(x_arr, n),
        "min": rolling_percentile(x_arr, n, 0),
        "max": rolling_percentile(x_arr, n, 100),
        "p25": rolling_percentile(x_arr, n, 25),
        "p50": rolling_percentile(x_arr, n, 50),
        "p75": rolling_percentile(x_arr, n, 75),
        "p10": rolling_percentile(x_arr, n, 10),
        "p90": rolling_percentile(x_arr, n, 90),
        "zscore": rolling_zscore(x_arr, n),
        "skewness": rolling_skewness(x_arr, n),
        "kurtosis": rolling_kurtosis(x_arr, n)
    }


def detect_outliers(x: ArrayLike, method: str = "zscore", threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers in the data.
    
    Args:
        x: Input data
        method: Detection method ("zscore", "iqr", "modified_zscore")
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean array indicating outliers
    """
    x_arr = np.asarray(x, dtype=np.float64)
    
    if method == "zscore":
        z_scores = np.abs(rolling_zscore(x_arr, min(20, len(x_arr))))
        return z_scores > threshold
    
    elif method == "iqr":
        q1 = np.percentile(x_arr, 25)
        q3 = np.percentile(x_arr, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (x_arr < lower_bound) | (x_arr > upper_bound)
    
    elif method == "modified_zscore":
        median = np.median(x_arr)
        mad = np.median(np.abs(x_arr - median))
        if mad == 0:
            return np.zeros_like(x_arr, dtype=bool)
        
        modified_z = 0.6745 * (x_arr - median) / mad
        return np.abs(modified_z) > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")


def winsorize(x: ArrayLike, limits: Tuple[float, float] = (0.05, 0.05)) -> np.ndarray:
    """
    Winsorize data by capping extreme values.
    
    Args:
        x: Input data
        limits: Tuple of (lower_limit, upper_limit) as fractions
    
    Returns:
        Winsorized data
    """
    if not (0 <= limits[0] <= 1 and 0 <= limits[1] <= 1):
        raise ValueError("Limits must be between 0 and 1")
    
    x_arr = np.asarray(x, dtype=np.float64)
    
    lower_limit = np.percentile(x_arr, limits[0] * 100)
    upper_limit = np.percentile(x_arr, (1 - limits[1]) * 100)
    
    result = x_arr.copy()
    result[result < lower_limit] = lower_limit
    result[result > upper_limit] = upper_limit
    
    return result
