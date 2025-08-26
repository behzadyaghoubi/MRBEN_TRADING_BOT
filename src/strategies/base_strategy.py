"""
Base strategy class for MR BEN Trading Bot.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..core.logger import get_logger

logger = get_logger("strategies")


@dataclass
class SignalResult:
    """Result of a signal generation."""
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    features: Dict[str, Any]  # Additional features for AI
    metadata: Dict[str, Any]  # Strategy-specific metadata


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.parameters = parameters or {}
        self.logger = get_logger(f"strategy.{name}")
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals added
        """
        pass
    
    @abstractmethod
    def get_latest_signal(self, data: pd.DataFrame) -> SignalResult:
        """
        Get the latest signal from the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            SignalResult object
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns."""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        if len(data) < 50:
            self.logger.warning("Insufficient data for strategy analysis")
            return False
        
        return True
    
    def calculate_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators."""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_FAST'] = df['close'].rolling(window=20).mean()
        df['SMA_SLOW'] = df['close'].rolling(window=50).mean()
        df['SMA200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=14).mean()
        
        return df
    
    def detect_price_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect common price action patterns."""
        df = data.copy()
        
        # Pin Bar detection
        body_size = np.abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        
        # Bullish Pin Bar
        df['bullish_pin'] = (
            (body_size < total_range * 0.3) &
            (lower_shadow > total_range * 0.6) &
            (upper_shadow < total_range * 0.1)
        ).astype(int)
        
        # Bearish Pin Bar
        df['bearish_pin'] = (
            (body_size < total_range * 0.3) &
            (upper_shadow > total_range * 0.6) &
            (lower_shadow < total_range * 0.1)
        ).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)
        
        return df
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'description': self.__doc__ or "No description available"
        }
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        self.parameters.update(new_parameters)
        self.logger.info(f"Parameters updated: {new_parameters}")
    
    def get_performance_metrics(self, signals: pd.DataFrame, actual_returns: pd.Series) -> Dict[str, float]:
        """Calculate strategy performance metrics."""
        if len(signals) != len(actual_returns):
            self.logger.error("Signals and returns length mismatch")
            return {}
        
        # Filter buy signals
        buy_signals = signals[signals['signal'] == 'BUY']
        if len(buy_signals) == 0:
            return {'total_signals': 0, 'win_rate': 0.0, 'avg_return': 0.0}
        
        # Calculate returns for buy signals
        buy_returns = actual_returns[buy_signals.index]
        
        # Performance metrics
        total_signals = len(buy_signals)
        winning_trades = len(buy_returns[buy_returns > 0])
        win_rate = winning_trades / total_signals if total_signals > 0 else 0
        avg_return = buy_returns.mean()
        total_return = buy_returns.sum()
        
        return {
            'total_signals': total_signals,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return
        } 