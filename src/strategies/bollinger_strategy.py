"""
Bollinger Bands Strategy for MR BEN Trading Bot.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base_strategy import BaseStrategy, SignalResult


class BollingerStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy:
    - Mean reversion at band extremes
    - Breakout trading at band breaks
    - Squeeze detection for volatility
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'period': 20,
            'std_dev': 2.0,
            'squeeze_threshold': 0.5,
            'mean_reversion_strength': 0.7
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("BollingerStrategy", default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using Bollinger Bands."""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        df = self.calculate_basic_indicators(df)
        df = self._calculate_bollinger_indicators(df)
        
        # Generate signals
        signals = []
        for i in range(1, len(df)):
            signal = self._analyze_bollinger_signals(df, i)
            signals.append(signal)
        
        # Add signals to dataframe
        df = df.iloc[1:].copy()
        df['signal'] = signals
        
        return df
    
    def get_latest_signal(self, data: pd.DataFrame) -> SignalResult:
        """Get the latest signal from Bollinger Bands analysis."""
        if not self.validate_data(data):
            return SignalResult('HOLD', 0.0, {}, {})
        
        df = data.copy()
        df = self.calculate_basic_indicators(df)
        df = self._calculate_bollinger_indicators(df)
        
        if len(df) < 2:
            return SignalResult('HOLD', 0.0, {}, {})
        
        # Analyze the latest candle
        latest_idx = len(df) - 1
        signal = self._analyze_bollinger_signals(df, latest_idx)
        
        # Calculate confidence
        confidence = self._calculate_bollinger_confidence(df, latest_idx)
        
        # Extract features
        features = self._extract_features(df, latest_idx)
        
        # Metadata
        metadata = {
            'strategy': 'BollingerStrategy',
            'indicators': {
                'bb_position': df['bb_position'].iloc[latest_idx],
                'bb_squeeze': df['bb_squeeze'].iloc[latest_idx],
                'bb_width': df['bb_width'].iloc[latest_idx]
            }
        }
        
        return SignalResult(signal, confidence, features, metadata)
    
    def _calculate_bollinger_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands specific indicators."""
        period = self.parameters['period']
        std_dev = self.parameters['std_dev']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        
        # Band width and position
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Squeeze detection
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=period).mean() * self.parameters['squeeze_threshold']
        
        # %B indicator
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _analyze_bollinger_signals(self, df: pd.DataFrame, idx: int) -> str:
        """Analyze Bollinger Bands for trading signals."""
        if idx < 1:
            return 'HOLD'
        
        current = df.iloc[idx]
        previous = df.iloc[idx - 1]
        
        # Mean reversion signals
        mean_reversion_buy = (
            current['bb_percent_b'] < 0.1 and  # Price near lower band
            current['close'] > previous['close'] and  # Price starting to rise
            not current['bb_squeeze']  # Not in squeeze
        )
        
        mean_reversion_sell = (
            current['bb_percent_b'] > 0.9 and  # Price near upper band
            current['close'] < previous['close'] and  # Price starting to fall
            not current['bb_squeeze']  # Not in squeeze
        )
        
        # Breakout signals
        breakout_buy = (
            current['close'] > current['bb_upper'] and  # Price breaks above upper band
            current['close'] > previous['close'] and  # Strong momentum
            current['bb_width'] > previous['bb_width']  # Expanding volatility
        )
        
        breakout_sell = (
            current['close'] < current['bb_lower'] and  # Price breaks below lower band
            current['close'] < previous['close'] and  # Strong momentum
            current['bb_width'] > previous['bb_width']  # Expanding volatility
        )
        
        # Squeeze breakout
        squeeze_buy = (
            current['bb_squeeze'] and  # In squeeze
            current['close'] > current['bb_upper'] and  # Breaks out
            current['bb_width'] > previous['bb_width'] * 1.2  # Significant expansion
        )
        
        squeeze_sell = (
            current['bb_squeeze'] and  # In squeeze
            current['close'] < current['bb_lower'] and  # Breaks out
            current['bb_width'] > previous['bb_width'] * 1.2  # Significant expansion
        )
        
        # Decision logic
        if mean_reversion_buy or breakout_buy or squeeze_buy:
            return 'BUY'
        elif mean_reversion_sell or breakout_sell or squeeze_sell:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_bollinger_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate confidence based on Bollinger Bands signals."""
        if idx < 1:
            return 0.0
        
        current = df.iloc[idx]
        previous = df.iloc[idx - 1]
        
        confidence_factors = []
        
        # Band position strength
        if current['bb_percent_b'] < 0.1 or current['bb_percent_b'] > 0.9:
            confidence_factors.append(0.4)
        elif current['bb_percent_b'] < 0.2 or current['bb_percent_b'] > 0.8:
            confidence_factors.append(0.2)
        
        # Breakout strength
        if (current['close'] > current['bb_upper'] or current['close'] < current['bb_lower']):
            confidence_factors.append(0.5)
        
        # Volume confirmation (if available)
        if 'tick_volume' in df.columns:
            avg_volume = df['tick_volume'].iloc[idx-5:idx].mean()
            if current['tick_volume'] > avg_volume * 1.5:
                confidence_factors.append(0.3)
        
        # Band width expansion
        if current['bb_width'] > previous['bb_width'] * 1.1:
            confidence_factors.append(0.2)
        
        # Squeeze breakout
        if current['bb_squeeze'] and current['bb_width'] > previous['bb_width'] * 1.2:
            confidence_factors.append(0.4)
        
        return min(sum(confidence_factors), 1.0)
    
    def _extract_features(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """Extract features for AI model."""
        if idx < 1:
            return {}
        
        current = df.iloc[idx]
        
        return {
            'close': current['close'],
            'bb_middle': current['bb_middle'],
            'bb_upper': current['bb_upper'],
            'bb_lower': current['bb_lower'],
            'bb_width': current['bb_width'],
            'bb_position': current['bb_position'],
            'bb_percent_b': current['bb_percent_b'],
            'bb_squeeze': current['bb_squeeze'],
            'price_change': (current['close'] - df['close'].iloc[idx-1]) / df['close'].iloc[idx-1],
            'bb_width_change': (current['bb_width'] - df['bb_width'].iloc[idx-1]) / df['bb_width'].iloc[idx-1] if df['bb_width'].iloc[idx-1] > 0 else 0
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information."""
        info = super().get_strategy_info()
        info.update({
            'description': """
            Bollinger Bands Strategy uses volatility-based indicators:
            - Mean reversion at band extremes
            - Breakout trading at band breaks
            - Squeeze detection for low volatility periods
            - Band width analysis for volatility expansion
            
            Combines multiple signal types for robust trading decisions.
            """,
            'parameters': self.parameters,
            'signal_types': ['BUY', 'SELL', 'HOLD'],
            'timeframe': 'Any (recommended: M15, H1, H4)'
        })
        return info 