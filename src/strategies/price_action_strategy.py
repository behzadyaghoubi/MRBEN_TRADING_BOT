"""
Price Action Strategy for MR BEN Trading Bot.
"""

from typing import Any

import pandas as pd

from .base_strategy import BaseStrategy, SignalResult


class PriceActionStrategy(BaseStrategy):
    """
    Price Action Strategy based on candlestick patterns:
    - Pin Bar (Hammer/Shooting Star)
    - Engulfing patterns
    - Inside/Outside bars
    - Support/Resistance levels
    """

    def __init__(self, parameters: dict[str, Any] = None):
        default_params = {
            'pin_bar_threshold': 0.3,
            'engulfing_threshold': 0.6,
            'support_resistance_periods': 20,
            'min_pattern_strength': 0.7,
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("PriceActionStrategy", default_params)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using price action patterns."""
        if not self.validate_data(data):
            return data

        df = data.copy()
        df = self.detect_price_patterns(df)

        # Generate signals
        signals = []
        for i in range(1, len(df)):
            signal = self._analyze_price_action(df, i)
            signals.append(signal)

        # Add signals to dataframe
        df = df.iloc[1:].copy()
        df['signal'] = signals

        return df

    def get_latest_signal(self, data: pd.DataFrame) -> SignalResult:
        """Get the latest signal from price action analysis."""
        if not self.validate_data(data):
            return SignalResult('HOLD', 0.0, {}, {})

        df = data.copy()
        df = self.detect_price_patterns(df)

        if len(df) < 2:
            return SignalResult('HOLD', 0.0, {}, {})

        # Analyze the latest candle
        latest_idx = len(df) - 1
        signal = self._analyze_price_action(df, latest_idx)

        # Calculate confidence
        confidence = self._calculate_pattern_confidence(df, latest_idx)

        # Extract features
        features = self._extract_features(df, latest_idx)

        # Metadata
        metadata = {
            'strategy': 'PriceActionStrategy',
            'patterns': {
                'bullish_pin': df['bullish_pin'].iloc[latest_idx],
                'bearish_pin': df['bearish_pin'].iloc[latest_idx],
                'bullish_engulfing': df['bullish_engulfing'].iloc[latest_idx],
                'bearish_engulfing': df['bearish_engulfing'].iloc[latest_idx],
            },
        }

        return SignalResult(signal, confidence, features, metadata)

    def _analyze_price_action(self, df: pd.DataFrame, idx: int) -> str:
        """Analyze price action patterns for signals."""
        if idx < 1:
            return 'HOLD'

        current = df.iloc[idx]
        previous = df.iloc[idx - 1]

        # Bullish patterns
        bullish_patterns = [
            current['bullish_pin'] == 1,
            current['bullish_engulfing'] == 1,
            self._is_support_level(df, idx),
            self._is_breakout_up(df, idx),
        ]

        # Bearish patterns
        bearish_patterns = [
            current['bearish_pin'] == 1,
            current['bearish_engulfing'] == 1,
            self._is_resistance_level(df, idx),
            self._is_breakout_down(df, idx),
        ]

        # Count patterns
        bullish_score = sum(bullish_patterns)
        bearish_score = sum(bearish_patterns)

        # Decision logic
        if bullish_score >= 2:
            return 'BUY'
        elif bearish_score >= 2:
            return 'SELL'
        else:
            return 'HOLD'

    def _is_support_level(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if current level is support."""
        if idx < 10:
            return False

        current_low = df['low'].iloc[idx]
        recent_lows = df['low'].iloc[idx - 10 : idx]

        # Check if current low is near recent lows
        support_levels = recent_lows.nsmallest(3)
        return any(abs(current_low - level) / current_low < 0.001 for level in support_levels)

    def _is_resistance_level(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if current level is resistance."""
        if idx < 10:
            return False

        current_high = df['high'].iloc[idx]
        recent_highs = df['high'].iloc[idx - 10 : idx]

        # Check if current high is near recent highs
        resistance_levels = recent_highs.nlargest(3)
        return any(abs(current_high - level) / current_high < 0.001 for level in resistance_levels)

    def _is_breakout_up(self, df: pd.DataFrame, idx: int) -> bool:
        """Check for upward breakout."""
        if idx < 5:
            return False

        current_close = df['close'].iloc[idx]
        recent_highs = df['high'].iloc[idx - 5 : idx]

        return current_close > recent_highs.max()

    def _is_breakout_down(self, df: pd.DataFrame, idx: int) -> bool:
        """Check for downward breakout."""
        if idx < 5:
            return False

        current_close = df['close'].iloc[idx]
        recent_lows = df['low'].iloc[idx - 5 : idx]

        return current_close < recent_lows.min()

    def _calculate_pattern_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate confidence based on pattern strength."""
        if idx < 1:
            return 0.0

        current = df.iloc[idx]

        # Pattern strength factors
        factors = []

        # Pin bar strength
        if current['bullish_pin'] == 1 or current['bearish_pin'] == 1:
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            if total_range > 0:
                factors.append(1 - (body_size / total_range))

        # Engulfing strength
        if current['bullish_engulfing'] == 1 or current['bearish_engulfing'] == 1:
            factors.append(0.8)

        # Volume confirmation (if available)
        if 'tick_volume' in df.columns:
            avg_volume = df['tick_volume'].iloc[idx - 5 : idx].mean()
            if current['tick_volume'] > avg_volume * 1.5:
                factors.append(0.3)

        # Support/Resistance proximity
        if self._is_support_level(df, idx) or self._is_resistance_level(df, idx):
            factors.append(0.4)

        return min(sum(factors), 1.0) if factors else 0.0

    def _extract_features(self, df: pd.DataFrame, idx: int) -> dict[str, Any]:
        """Extract features for AI model."""
        if idx < 1:
            return {}

        current = df.iloc[idx]

        return {
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'body_size': abs(current['close'] - current['open']),
            'upper_shadow': current['high'] - max(current['open'], current['close']),
            'lower_shadow': min(current['open'], current['close']) - current['low'],
            'total_range': current['high'] - current['low'],
            'bullish_pin': current['bullish_pin'],
            'bearish_pin': current['bearish_pin'],
            'bullish_engulfing': current['bullish_engulfing'],
            'bearish_engulfing': current['bearish_engulfing'],
            'price_change': (current['close'] - df['close'].iloc[idx - 1])
            / df['close'].iloc[idx - 1],
        }

    def get_strategy_info(self) -> dict[str, Any]:
        """Get detailed strategy information."""
        info = super().get_strategy_info()
        info.update(
            {
                'description': """
            Price Action Strategy focuses on candlestick patterns and market structure:
            - Pin Bar patterns (Hammer/Shooting Star)
            - Engulfing patterns (Bullish/Bearish)
            - Support and resistance levels
            - Breakout detection

            Requires multiple pattern confirmations for signals.
            """,
                'parameters': self.parameters,
                'signal_types': ['BUY', 'SELL', 'HOLD'],
                'timeframe': 'Any (recommended: M15, H1, H4)',
            }
        )
        return info
