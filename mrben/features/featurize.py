import numpy as np
import pandas as pd
from typing import Optional, Union


def build_features(df: Union[pd.DataFrame, np.ndarray], 
                  lookback: int = 50) -> np.ndarray:
    """
    Build feature vector for ML/LSTM models.
    
    Args:
        df: DataFrame with OHLCV data or numpy array
        lookback: Number of historical bars to include
        
    Returns:
        Feature matrix with shape [N, F] where F is feature count
    """
    if isinstance(df, np.ndarray):
        # Convert numpy array to DataFrame for easier processing
        df = pd.DataFrame(df, columns=['O', 'H', 'L', 'C', 'V'])
    
    # Ensure we have the required columns
    required_cols = ['O', 'H', 'L', 'C', 'V']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Calculate basic features
    features = []
    
    # 1. Price-based features
    close = df['C'].values
    high = df['H'].values
    low = df['L'].values
    volume = df['V'].values
    
    # Returns - calculate manually to ensure consistent lengths
    ret1 = np.zeros_like(close)
    ret3 = np.zeros_like(close)
    ret5 = np.zeros_like(close)
    
    # 1-period return
    ret1[1:] = (close[1:] - close[:-1]) / np.maximum(1e-9, close[:-1])
    
    # 3-period return
    for i in range(3, len(close)):
        ret3[i] = (close[i] - close[i-3]) / np.maximum(1e-9, close[i-3])
    
    # 5-period return
    for i in range(5, len(close)):
        ret5[i] = (close[i] - close[i-5]) / np.maximum(1e-9, close[i-5])
    
    # All arrays now have the same length as the original close array
    min_len = len(close)
    
    # High-Low range
    hl_range = (high - low) / np.maximum(1e-9, close)
    
    # Body size
    body_size = np.abs(close - df['O'].values) / np.maximum(1e-9, close)
    
    features.extend([ret1, ret3, ret5, hl_range, body_size])
    
    # 2. Technical indicators (if available)
    if 'ATR14' in df.columns:
        atr_values = df['ATR14'].values[-min_len:]
        features.append(atr_values / np.maximum(1e-9, close))
    else:
        # Calculate simple ATR
        atr = calculate_simple_atr(high, low, close)
        features.append(atr / np.maximum(1e-9, close))
    
    if 'RSI14' in df.columns:
        rsi_values = df['RSI14'].values[-min_len:]
        features.append(rsi_values / 100.0)
    else:
        # Calculate simple RSI
        rsi = calculate_simple_rsi(close)
        features.append(rsi / 100.0)
    
    if 'SMA20' in df.columns and 'SMA50' in df.columns:
        sma20_values = df['SMA20'].values[-min_len:]
        sma50_values = df['SMA50'].values[-min_len:]
        sma_diff = (sma20_values - sma50_values) / np.maximum(1e-9, sma50_values)
        features.append(sma_diff)
    else:
        # Calculate simple moving averages
        sma20 = calculate_sma(close, 20)
        sma50 = calculate_sma(close, 50)
        sma_diff = (sma20 - sma50) / np.maximum(1e-9, sma50)
        features.append(sma_diff)
    
    # 3. Volume features
    volume_ma = calculate_sma(volume, 20)
    volume_ratio = volume / np.maximum(1e-9, volume_ma)
    features.append(volume_ratio)
    
    # 4. Session features (one-hot encoding)
    if 'session' in df.columns:
        session_dummies = pd.get_dummies(df['session'], prefix='sess')
        for col in ['sess_asia', 'sess_london', 'sess_ny']:
            if col in session_dummies.columns:
                session_values = session_dummies[col].values[-min_len:]
                features.append(session_values)
            else:
                features.append(np.zeros(min_len))
    else:
        # Default session features (all zeros)
        for _ in range(3):
            features.append(np.zeros(min_len))
    
    # 5. Regime features (one-hot encoding)
    if 'regime' in df.columns:
        regime_dummies = pd.get_dummies(df['regime'], prefix='reg')
        for col in ['reg_low', 'reg_normal', 'reg_high']:
            if col in regime_dummies.columns:
                regime_values = regime_dummies[col].values[-min_len:]
                features.append(regime_values)
            else:
                features.append(np.zeros(min_len))
    else:
        # Default regime features (all zeros)
        for _ in range(3):
            features.append(np.zeros(min_len))
    
    # 6. Time-based features
    if 'hour' in df.columns:
        # Normalize hour to [0, 1]
        hour_values = df['hour'].values[-min_len:]
        hour_norm = hour_values / 24.0
        features.append(hour_norm)
    else:
        features.append(np.zeros(min_len))
    
    # 7. Volatility features
    volatility = calculate_rolling_volatility(close, 20)
    features.append(volatility)
    
    # Stack all features
    X = np.vstack([f if f.ndim == 1 else f for f in features])
    
    # Transpose to get [N, F] format
    X = X.T.astype(np.float32)
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Only limit if explicitly requested and if we have more samples than lookback
    if lookback > 0 and len(X) > lookback:
        X = X[-lookback:]
    
    return X


def calculate_simple_atr(high: np.ndarray, low: np.ndarray, 
                        close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate simple ATR."""
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = high[0] - low[0]  # First value
    
    atr = np.zeros_like(tr)
    atr[:period] = tr[:period].mean()
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr


def calculate_simple_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate simple RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    avg_gain[:period] = gain[:period].mean()
    avg_loss[:period] = loss[:period].mean()
    
    for i in range(period, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = avg_gain / np.maximum(1e-9, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate simple moving average."""
    sma = np.zeros_like(data)
    sma[:period] = data[:period].mean()
    
    for i in range(period, len(data)):
        sma[i] = (sma[i-1] * (period-1) + data[i]) / period
    
    return sma


def calculate_rolling_volatility(data: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate rolling volatility."""
    volatility = np.zeros_like(data)
    
    for i in range(period, len(data)):
        window = data[i-period:i]
        volatility[i] = np.std(window)
    
    # Normalize by mean
    volatility = volatility / np.maximum(1e-9, np.mean(data))
    
    return volatility


def prepare_ml_features(df: Union[pd.DataFrame, np.ndarray], 
                       lookback: int = 50) -> np.ndarray:
    """
    Prepare features for ML filter (single row).
    
    Args:
        df: DataFrame or numpy array with market data
        lookback: Number of historical bars
        
    Returns:
        Single feature row with shape [F]
    """
    features = build_features(df, lookback)
    
    # Return the most recent feature vector
    if len(features) > 0:
        return features[-1]
    else:
        return np.array([])


def prepare_lstm_features(df: Union[pd.DataFrame, np.ndarray], 
                         lookback: int = 50) -> np.ndarray:
    """
    Prepare features for LSTM model (sequence).
    
    Args:
        df: DataFrame or numpy array with market data
        lookback: Number of historical bars
        
    Returns:
        Feature sequence with shape [T, F]
    """
    features = build_features(df, lookback)
    
    # Ensure we have the right shape for LSTM
    if len(features) == 0:
        return np.array([])
    
    # LSTM expects [T, F] format
    return features.astype(np.float32)


def validate_features(X: np.ndarray) -> bool:
    """
    Validate feature matrix.
    
    Args:
        X: Feature matrix
        
    Returns:
        True if valid, False otherwise
    """
    if X is None or len(X) == 0:
        return False
    
    if not isinstance(X, np.ndarray):
        return False
    
    if X.ndim != 2:
        return False
    
    # Check for infinite or NaN values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        return False
    
    return True
