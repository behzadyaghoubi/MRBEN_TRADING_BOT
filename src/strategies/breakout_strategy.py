"""
Breakout Strategy for MR BEN Trading Bot.
"""

from typing import Any

import pandas as pd

from .base_strategy import BaseStrategy, SignalResult


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy:
    - Support and resistance level breaks
    - Volume confirmation
    - False breakout detection
    """

    def __init__(self, parameters: dict[str, Any] = None):
        default_params = {
            'support_resistance_periods': 20,
            'breakout_threshold': 0.001,
            'volume_multiplier': 1.5,
            'false_breakout_threshold': 0.0005,
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("BreakoutStrategy", default_params)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using breakout analysis."""
        if not self.validate_data(data):
            return data

        df = data.copy()
        df = self.calculate_basic_indicators(df)
        df = self._calculate_breakout_indicators(df)

        # Generate signals
        signals = []
        for i in range(1, len(df)):
            signal = self._analyze_breakout_signals(df, i)
            signals.append(signal)

        # Add signals to dataframe
        df = df.iloc[1:].copy()
        df['signal'] = signals

        return df

    def get_latest_signal(self, data: pd.DataFrame) -> SignalResult:
        """Get the latest signal from breakout analysis."""
        if not self.validate_data(data):
            return SignalResult('HOLD', 0.0, {}, {})

        df = data.copy()
        df = self.calculate_basic_indicators(df)
        df = self._calculate_breakout_indicators(df)

        if len(df) < 2:
            return SignalResult('HOLD', 0.0, {}, {})

        # Analyze the latest candle
        latest_idx = len(df) - 1
        signal = self._analyze_breakout_signals(df, latest_idx)

        # Calculate confidence
        confidence = self._calculate_breakout_confidence(df, latest_idx)

        # Extract features
        features = self._extract_features(df, latest_idx)

        # Metadata
        metadata = {
            'strategy': 'BreakoutStrategy',
            'indicators': {
                'resistance_break': df['resistance_break'].iloc[latest_idx],
                'support_break': df['support_break'].iloc[latest_idx],
                'volume_confirmation': df['volume_confirmation'].iloc[latest_idx],
            },
        }

        return SignalResult(signal, confidence, features, metadata)

    def _calculate_breakout_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout-specific indicators."""
        periods = self.parameters['support_resistance_periods']
        threshold = self.parameters['breakout_threshold']

        # Support and resistance levels
        df['resistance_level'] = df['high'].rolling(window=periods).max()
        df['support_level'] = df['low'].rolling(window=periods).min()

        # Breakout detection
        df['resistance_break'] = df['close'] > df['resistance_level'].shift(1) * (1 + threshold)
        df['support_break'] = df['close'] < df['support_level'].shift(1) * (1 - threshold)

        # Volume confirmation
        if 'tick_volume' in df.columns:
            df['volume_avg'] = df['tick_volume'].rolling(window=periods).mean()
            df['volume_confirmation'] = (
                df['tick_volume'] > df['volume_avg'] * self.parameters['volume_multiplier']
            )
        else:
            df['volume_confirmation'] = True

        # False breakout detection
        df['false_breakout_up'] = df['resistance_break'] & (
            df['close']
            < df['resistance_level'].shift(1) * (1 - self.parameters['false_breakout_threshold'])
        )

        df['false_breakout_down'] = df['support_break'] & (
            df['close']
            > df['support_level'].shift(1) * (1 + self.parameters['false_breakout_threshold'])
        )

        return df

    def _analyze_breakout_signals(self, df: pd.DataFrame, idx: int) -> str:
        """Analyze breakouts for trading signals."""
        if idx < 1:
            return 'HOLD'

        current = df.iloc[idx]
        previous = df.iloc[idx - 1]

        # Bullish breakout
        bullish_breakout = (
            current['resistance_break']
            and current['volume_confirmation']
            and not current['false_breakout_up']
            and current['close'] > previous['close']
        )

        # Bearish breakout
        bearish_breakout = (
            current['support_break']
            and current['volume_confirmation']
            and not current['false_breakout_down']
            and current['close'] < previous['close']
        )

        # Consolidation breakout
        consolidation_buy = (
            current['close'] > current['resistance_level']
            and current['close'] > previous['close'] * 1.005  # Strong move
            and current['volume_confirmation']
        )

        consolidation_sell = (
            current['close'] < current['support_level']
            and current['close'] < previous['close'] * 0.995  # Strong move
            and current['volume_confirmation']
        )

        # Decision logic
        if bullish_breakout or consolidation_buy:
            return 'BUY'
        elif bearish_breakout or consolidation_sell:
            return 'SELL'
        else:
            return 'HOLD'

    def _calculate_breakout_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate confidence based on breakout signals."""
        if idx < 1:
            return 0.0

        current = df.iloc[idx]
        previous = df.iloc[idx - 1]

        confidence_factors = []

        # Breakout strength
        if current['resistance_break'] or current['support_break']:
            confidence_factors.append(0.4)

        # Volume confirmation
        if current['volume_confirmation']:
            confidence_factors.append(0.3)

        # Price momentum
        price_change = abs(current['close'] - previous['close']) / previous['close']
        if price_change > 0.005:  # 0.5% move
            confidence_factors.append(0.2)

        # Level strength
        if current['resistance_break']:
            level_strength = (current['close'] - current['resistance_level'].shift(1)) / current[
                'resistance_level'
            ].shift(1)
            confidence_factors.append(min(level_strength * 10, 0.3))

        if current['support_break']:
            level_strength = (current['support_level'].shift(1) - current['close']) / current[
                'support_level'
            ].shift(1)
            confidence_factors.append(min(level_strength * 10, 0.3))

        return min(sum(confidence_factors), 1.0)

    def _extract_features(self, df: pd.DataFrame, idx: int) -> dict[str, Any]:
        """Extract features for AI model."""
        if idx < 1:
            return {}

        current = df.iloc[idx]

        return {
            'close': current['close'],
            'resistance_level': current['resistance_level'],
            'support_level': current['support_level'],
            'resistance_break': current['resistance_break'],
            'support_break': current['support_break'],
            'volume_confirmation': current['volume_confirmation'],
            'false_breakout_up': current['false_breakout_up'],
            'false_breakout_down': current['false_breakout_down'],
            'price_change': (current['close'] - df['close'].iloc[idx - 1])
            / df['close'].iloc[idx - 1],
        }

    def get_strategy_info(self) -> dict[str, Any]:
        """Get detailed strategy information."""
        info = super().get_strategy_info()
        info.update(
            {
                'description': """
            Breakout Strategy focuses on support and resistance levels:
            - Resistance level breakouts (bullish)
            - Support level breakouts (bearish)
            - Volume confirmation for breakouts
            - False breakout detection

            Effective for range-bound markets with clear levels.
            """,
                'parameters': self.parameters,
                'signal_types': ['BUY', 'SELL', 'HOLD'],
                'timeframe': 'Any (recommended: H1, H4, D1)',
            }
        )
        return info
