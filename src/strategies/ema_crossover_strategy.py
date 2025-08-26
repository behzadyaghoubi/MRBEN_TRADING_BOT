"""
EMA Crossover Strategy for MR BEN Trading Bot.
"""

from typing import Any

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, SignalResult


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy:
    - Fast and slow EMA crossover signals
    - Trend strength confirmation
    - Momentum analysis
    """

    def __init__(self, parameters: dict[str, Any] = None):
        default_params = {
            'fast_ema': 12,
            'slow_ema': 26,
            'signal_ema': 9,
            'trend_strength_threshold': 0.5,
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("EMACrossoverStrategy", default_params)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using EMA crossovers."""
        if not self.validate_data(data):
            return data

        df = data.copy()
        df = self.calculate_basic_indicators(df)
        df = self._calculate_ema_indicators(df)

        # Generate signals
        signals = []
        for i in range(1, len(df)):
            signal = self._analyze_ema_signals(df, i)
            signals.append(signal)

        # Add signals to dataframe
        df = df.iloc[1:].copy()
        df['signal'] = signals

        return df

    def get_latest_signal(self, data: pd.DataFrame) -> SignalResult:
        """Get the latest signal from EMA crossover analysis."""
        if not self.validate_data(data):
            return SignalResult('HOLD', 0.0, {}, {})

        df = data.copy()
        df = self.calculate_basic_indicators(df)
        df = self._calculate_ema_indicators(df)

        if len(df) < 2:
            return SignalResult('HOLD', 0.0, {}, {})

        # Analyze the latest candle
        latest_idx = len(df) - 1
        signal = self._analyze_ema_signals(df, latest_idx)

        # Calculate confidence
        confidence = self._calculate_ema_confidence(df, latest_idx)

        # Extract features
        features = self._extract_features(df, latest_idx)

        # Metadata
        metadata = {
            'strategy': 'EMACrossoverStrategy',
            'indicators': {
                'ema_cross': df['ema_cross'].iloc[latest_idx],
                'trend_strength': df['trend_strength'].iloc[latest_idx],
                'momentum': df['momentum'].iloc[latest_idx],
            },
        }

        return SignalResult(signal, confidence, features, metadata)

    def _calculate_ema_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA-specific indicators."""
        fast_ema = self.parameters['fast_ema']
        slow_ema = self.parameters['slow_ema']
        signal_ema = self.parameters['signal_ema']

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=fast_ema).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_ema).mean()
        df['ema_signal'] = df['close'].ewm(span=signal_ema).mean()

        # Crossover signals
        df['ema_cross'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        df['ema_cross_change'] = df['ema_cross'].diff()

        # Trend strength
        df['trend_strength'] = abs(df['ema_fast'] - df['ema_slow']) / df['close']

        # Momentum
        df['momentum'] = df['close'].pct_change(periods=5)

        return df

    def _analyze_ema_signals(self, df: pd.DataFrame, idx: int) -> str:
        """Analyze EMA crossovers for trading signals."""
        if idx < 1:
            return 'HOLD'

        current = df.iloc[idx]
        previous = df.iloc[idx - 1]

        # Bullish crossover
        bullish_cross = (
            current['ema_cross_change'] == 2  # Fast EMA crosses above slow EMA
            and current['trend_strength'] > self.parameters['trend_strength_threshold']
            and current['momentum'] > 0
        )

        # Bearish crossover
        bearish_cross = (
            current['ema_cross_change'] == -2  # Fast EMA crosses below slow EMA
            and current['trend_strength'] > self.parameters['trend_strength_threshold']
            and current['momentum'] < 0
        )

        # Trend continuation
        bullish_trend = (
            current['ema_cross'] == 1  # Fast EMA above slow EMA
            and current['close'] > current['ema_fast']  # Price above fast EMA
            and current['trend_strength'] > self.parameters['trend_strength_threshold'] * 0.5
        )

        bearish_trend = (
            current['ema_cross'] == -1  # Fast EMA below slow EMA
            and current['close'] < current['ema_fast']  # Price below fast EMA
            and current['trend_strength'] > self.parameters['trend_strength_threshold'] * 0.5
        )

        # Decision logic
        if bullish_cross or bullish_trend:
            return 'BUY'
        elif bearish_cross or bearish_trend:
            return 'SELL'
        else:
            return 'HOLD'

    def _calculate_ema_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate confidence based on EMA signals."""
        if idx < 1:
            return 0.0

        current = df.iloc[idx]
        previous = df.iloc[idx - 1]

        confidence_factors = []

        # Crossover strength
        if abs(current['ema_cross_change']) == 2:
            confidence_factors.append(0.5)

        # Trend strength
        trend_strength = min(current['trend_strength'] * 10, 1.0)
        confidence_factors.append(trend_strength * 0.3)

        # Momentum alignment
        if abs(current['momentum']) > 0.01:
            confidence_factors.append(0.2)

        # Price position relative to EMAs
        if (current['close'] > current['ema_fast'] > current['ema_slow']) or (
            current['close'] < current['ema_fast'] < current['ema_slow']
        ):
            confidence_factors.append(0.2)

        return min(sum(confidence_factors), 1.0)

    def _extract_features(self, df: pd.DataFrame, idx: int) -> dict[str, Any]:
        """Extract features for AI model."""
        if idx < 1:
            return {}

        current = df.iloc[idx]

        return {
            'close': current['close'],
            'ema_fast': current['ema_fast'],
            'ema_slow': current['ema_slow'],
            'ema_signal': current['ema_signal'],
            'ema_cross': current['ema_cross'],
            'trend_strength': current['trend_strength'],
            'momentum': current['momentum'],
            'price_change': (current['close'] - df['close'].iloc[idx - 1])
            / df['close'].iloc[idx - 1],
        }

    def get_strategy_info(self) -> dict[str, Any]:
        """Get detailed strategy information."""
        info = super().get_strategy_info()
        info.update(
            {
                'description': """
            EMA Crossover Strategy uses exponential moving averages:
            - Fast and slow EMA crossover signals
            - Trend strength confirmation
            - Momentum analysis
            - Trend continuation signals

            Effective for trending markets with clear directional movement.
            """,
                'parameters': self.parameters,
                'signal_types': ['BUY', 'SELL', 'HOLD'],
                'timeframe': 'Any (recommended: M15, H1, H4)',
            }
        )
        return info
