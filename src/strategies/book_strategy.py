"""
Book Strategy - Production-ready technical analysis strategy.
Combines multiple technical indicators with proper signal generation and risk management.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from config.settings import settings
from core.logger import get_logger

from .base_strategy import BaseStrategy, SignalResult

logger = get_logger("strategies.book_strategy")


@dataclass
class SignalDetails:
    """Detailed signal information."""

    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    reasons: list[str]
    indicators: dict[str, Any]
    timestamp: datetime


class BookStrategy(BaseStrategy):
    """
    Production-ready Book Strategy combining multiple technical indicators:
    - RSI (Relative Strength Index) for overbought/oversold conditions
    - MACD (Moving Average Convergence Divergence) for trend changes
    - Bollinger Bands for volatility and mean reversion
    - SMA/EMA crossovers for trend direction
    - Candlestick patterns for price action confirmation
    - Volume analysis for signal strength
    """

    def __init__(self, parameters: dict[str, Any] | None = None):
        """Initialize Book Strategy with parameters."""
        default_params = {
            # RSI settings
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_divergence_lookback': 10,
            # MACD settings
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            # Moving averages
            'sma_fast': 20,
            'sma_slow': 50,
            'ema_fast': 12,
            'ema_slow': 26,
            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2.0,
            # Risk management
            'risk_reward_ratio': 2.0,
            'max_risk_per_trade': 0.02,  # 2% max risk
            'min_confidence': 0.6,
            # Candlestick patterns
            'pin_bar_threshold': 0.3,
            'engulfing_threshold': 0.6,
            'doji_threshold': 0.1,
            # Volume settings
            'volume_ma_period': 20,
            'min_volume_ratio': 1.2,
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("BookStrategy", default_params)
        self.logger = get_logger(f"strategy.{self.name}")

        # Get settings from configuration
        self.symbol = settings.trading.symbol
        self.timeframe = settings.trading.timeframe
        self.base_risk = settings.trading.base_risk
        self.stop_loss_pips = settings.trading.stop_loss_pips
        self.take_profit_pips = settings.trading.take_profit_pips

        self.logger.info(f"BookStrategy initialized for {self.symbol} {self.timeframe}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        if not self.validate_data(df):
            return df

        df = df.copy()

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.parameters['rsi_period'])
        df['rsi_oversold'] = self.parameters['rsi_oversold']
        df['rsi_overbought'] = self.parameters['rsi_overbought']

        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(
            df['close'],
            self.parameters['macd_fast'],
            self.parameters['macd_slow'],
            self.parameters['macd_signal'],
        )

        # Moving Averages
        df['sma_fast'] = df['close'].rolling(self.parameters['sma_fast']).mean()
        df['sma_slow'] = df['close'].rolling(self.parameters['sma_slow']).mean()
        df['ema_fast'] = df['close'].ewm(span=self.parameters['ema_fast']).mean()
        df['ema_slow'] = df['close'].ewm(span=self.parameters['ema_slow']).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(self.parameters['bb_period']).mean()
        bb_std = df['close'].rolling(self.parameters['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.parameters['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.parameters['bb_std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Volume analysis
        df['volume_ma'] = df['tick_volume'].rolling(self.parameters['volume_ma_period']).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']

        # Price action patterns
        df = self._detect_candlestick_patterns(df)

        # Support and resistance levels
        df = self._calculate_support_resistance(df)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, prices: pd.Series, fast: int, slow: int, signal: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect candlestick patterns."""
        # Pin Bar (Hammer/Shooting Star)
        body_size = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']

        # Hammer (bullish pin bar)
        df['hammer'] = (
            (lower_shadow > body_size * self.parameters['pin_bar_threshold'])
            & (upper_shadow < body_size * 0.1)
            & (df['close'] > df['open'])  # Bullish close
        )

        # Shooting Star (bearish pin bar)
        df['shooting_star'] = (
            (upper_shadow > body_size * self.parameters['pin_bar_threshold'])
            & (lower_shadow < body_size * 0.1)
            & (df['close'] < df['open'])  # Bearish close
        )

        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['open'] < df['close'].shift(1))
            & (df['close'] > df['open'].shift(1))
            & (df['open'] < df['close'])
            & (df['close'].shift(1) < df['open'].shift(1))
        )

        df['bearish_engulfing'] = (
            (df['open'] > df['close'].shift(1))
            & (df['close'] < df['open'].shift(1))
            & (df['open'] > df['close'])
            & (df['close'].shift(1) > df['open'].shift(1))
        )

        # Doji
        df['doji'] = body_size < (df['high'] - df['low']) * self.parameters['doji_threshold']

        return df

    def _calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Calculate support and resistance levels."""
        # Simple pivot points
        df['pivot_high'] = df['high'].rolling(lookback, center=True).max()
        df['pivot_low'] = df['low'].rolling(lookback, center=True).min()

        # Dynamic support/resistance
        df['resistance'] = df['high'].rolling(lookback).max()
        df['support'] = df['low'].rolling(lookback).min()

        return df

    def _validate_required_columns(self, df: pd.DataFrame) -> bool:
        """Validate that dataframe has all required columns for signal generation."""
        required_columns = [
            'open',
            'high',
            'low',
            'close',
            'tick_volume',
            'rsi',
            'macd',
            'macd_signal',
            'macd_histogram',
            'sma_fast',
            'sma_slow',
            'ema_fast',
            'ema_slow',
            'bb_upper',
            'bb_lower',
            'bb_middle',
            'bb_width',
            'volume_ma',
            'volume_ratio',
            'hammer',
            'shooting_star',
            'bullish_engulfing',
            'bearish_engulfing',
            'doji',
            'support',
            'resistance',
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(
                f"Missing required columns for signal generation: {missing_columns}"
            )
            return False

        return True

    def generate_signal(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate trading signal with comprehensive analysis.

        Returns:
            Dict containing signal details including entry, stop loss, take profit
        """
        # Check data length
        if len(df) < 50:
            return self._create_hold_signal("Insufficient data")

        # Check if dataframe already has indicators (from calculate_indicators)
        if not self._validate_required_columns(df):
            # Try to calculate indicators
            try:
                df = self.calculate_indicators(df)
                if not self._validate_required_columns(df):
                    return self._create_hold_signal("Failed to calculate required indicators")
            except Exception as e:
                self.logger.error(f"Error calculating indicators: {e}")
                return self._create_hold_signal(f"Error calculating indicators: {str(e)}")

        # Get latest data
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current

        # Analyze signals
        buy_signals = self._analyze_buy_signals(df, current, previous)
        sell_signals = self._analyze_sell_signals(df, current, previous)

        # Determine final signal
        if (
            buy_signals['confidence'] > sell_signals['confidence']
            and buy_signals['confidence'] >= self.parameters['min_confidence']
        ):
            return self._create_buy_signal(buy_signals, current)
        elif (
            sell_signals['confidence'] > buy_signals['confidence']
            and sell_signals['confidence'] >= self.parameters['min_confidence']
        ):
            return self._create_sell_signal(sell_signals, current)
        else:
            return self._create_hold_signal("No strong signals")

    def _analyze_buy_signals(
        self, df: pd.DataFrame, current: pd.Series, previous: pd.Series
    ) -> dict[str, Any]:
        """Analyze buy signals and calculate confidence."""
        reasons = []
        confidence = 0.0

        # RSI oversold
        if current['rsi'] < self.parameters['rsi_oversold']:
            reasons.append(f"RSI oversold ({current['rsi']:.1f})")
            confidence += 0.2

        # RSI divergence (bullish)
        if self._check_rsi_bullish_divergence(df):
            reasons.append("RSI bullish divergence")
            confidence += 0.15

        # MACD bullish crossover
        if previous['macd'] < previous['macd_signal'] and current['macd'] > current['macd_signal']:
            reasons.append("MACD bullish crossover")
            confidence += 0.15

        # Moving average crossover
        if (
            previous['ema_fast'] < previous['ema_slow']
            and current['ema_fast'] > current['ema_slow']
        ):
            reasons.append("EMA bullish crossover")
            confidence += 0.1

        # Bollinger Bands bounce
        if current['close'] <= current['bb_lower'] * 1.001:
            reasons.append("Bollinger Bands support")
            confidence += 0.1

        # Candlestick patterns
        if current['hammer']:
            reasons.append("Hammer pattern")
            confidence += 0.1

        if current['bullish_engulfing']:
            reasons.append("Bullish engulfing")
            confidence += 0.15

        # Volume confirmation
        if current['volume_ratio'] > self.parameters['min_volume_ratio']:
            reasons.append("High volume confirmation")
            confidence += 0.05

        # Support level
        if current['close'] <= current['support'] * 1.002:
            reasons.append("Support level test")
            confidence += 0.1

        return {'confidence': min(confidence, 1.0), 'reasons': reasons}

    def _analyze_sell_signals(
        self, df: pd.DataFrame, current: pd.Series, previous: pd.Series
    ) -> dict[str, Any]:
        """Analyze sell signals and calculate confidence."""
        reasons = []
        confidence = 0.0

        # RSI overbought
        if current['rsi'] > self.parameters['rsi_overbought']:
            reasons.append(f"RSI overbought ({current['rsi']:.1f})")
            confidence += 0.2

        # RSI divergence (bearish)
        if self._check_rsi_bearish_divergence(df):
            reasons.append("RSI bearish divergence")
            confidence += 0.15

        # MACD bearish crossover
        if previous['macd'] > previous['macd_signal'] and current['macd'] < current['macd_signal']:
            reasons.append("MACD bearish crossover")
            confidence += 0.15

        # Moving average crossover
        if (
            previous['ema_fast'] > previous['ema_slow']
            and current['ema_fast'] < current['ema_slow']
        ):
            reasons.append("EMA bearish crossover")
            confidence += 0.1

        # Bollinger Bands rejection
        if current['close'] >= current['bb_upper'] * 0.999:
            reasons.append("Bollinger Bands resistance")
            confidence += 0.1

        # Candlestick patterns
        if current['shooting_star']:
            reasons.append("Shooting star pattern")
            confidence += 0.1

        if current['bearish_engulfing']:
            reasons.append("Bearish engulfing")
            confidence += 0.15

        # Volume confirmation
        if current['volume_ratio'] > self.parameters['min_volume_ratio']:
            reasons.append("High volume confirmation")
            confidence += 0.05

        # Resistance level
        if current['close'] >= current['resistance'] * 0.998:
            reasons.append("Resistance level test")
            confidence += 0.1

        return {'confidence': min(confidence, 1.0), 'reasons': reasons}

    def _check_rsi_bullish_divergence(self, df: pd.DataFrame) -> bool:
        """Check for RSI bullish divergence."""
        lookback = self.parameters['rsi_divergence_lookback']
        if len(df) < lookback * 2:
            return False

        # Find recent lows
        recent_lows = df['low'].tail(lookback)
        recent_rsi_lows = df['rsi'].tail(lookback)

        # Check if price made lower low but RSI made higher low
        if (
            recent_lows.iloc[-1] < recent_lows.iloc[-lookback // 2]
            and recent_rsi_lows.iloc[-1] > recent_rsi_lows.iloc[-lookback // 2]
        ):
            return True

        return False

    def _check_rsi_bearish_divergence(self, df: pd.DataFrame) -> bool:
        """Check for RSI bearish divergence."""
        lookback = self.parameters['rsi_divergence_lookback']
        if len(df) < lookback * 2:
            return False

        # Find recent highs
        recent_highs = df['high'].tail(lookback)
        recent_rsi_highs = df['rsi'].tail(lookback)

        # Check if price made higher high but RSI made lower high
        if (
            recent_highs.iloc[-1] > recent_highs.iloc[-lookback // 2]
            and recent_rsi_highs.iloc[-1] < recent_rsi_highs.iloc[-lookback // 2]
        ):
            return True

        return False

    def _create_buy_signal(
        self, signal_analysis: dict[str, Any], current: pd.Series
    ) -> dict[str, Any]:
        """Create buy signal with entry, stop loss, and take profit levels."""
        entry_price = current['close']

        # Calculate stop loss and take profit based on ATR or fixed pips
        stop_loss = entry_price - (self.stop_loss_pips * self._get_pip_value())
        take_profit = entry_price + (self.take_profit_pips * self._get_pip_value())

        # Adjust based on support/resistance levels
        if current['support'] > stop_loss:
            stop_loss = current['support'] * 0.999

        risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)

        # Log the signal
        reasons_str = ", ".join(signal_analysis['reasons'])
        self.logger.info(
            f"BUY signal for {self.symbol}: {reasons_str} | "
            f"Entry: {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}, "
            f"R:R: {risk_reward:.2f}, Confidence: {signal_analysis['confidence']:.2f}"
        )

        return {
            'signal': 'BUY',
            'confidence': signal_analysis['confidence'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward,
            'reasons': signal_analysis['reasons'],
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
        }

    def _create_sell_signal(
        self, signal_analysis: dict[str, Any], current: pd.Series
    ) -> dict[str, Any]:
        """Create sell signal with entry, stop loss, and take profit levels."""
        entry_price = current['close']

        # Calculate stop loss and take profit based on ATR or fixed pips
        stop_loss = entry_price + (self.stop_loss_pips * self._get_pip_value())
        take_profit = entry_price - (self.take_profit_pips * self._get_pip_value())

        # Adjust based on support/resistance levels
        if current['resistance'] < stop_loss:
            stop_loss = current['resistance'] * 1.001

        risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)

        # Log the signal
        reasons_str = ", ".join(signal_analysis['reasons'])
        self.logger.info(
            f"SELL signal for {self.symbol}: {reasons_str} | "
            f"Entry: {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}, "
            f"R:R: {risk_reward:.2f}, Confidence: {signal_analysis['confidence']:.2f}"
        )

        return {
            'signal': 'SELL',
            'confidence': signal_analysis['confidence'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward,
            'reasons': signal_analysis['reasons'],
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
        }

    def _create_hold_signal(self, reason: str) -> dict[str, Any]:
        """Create hold signal."""
        self.logger.debug(f"HOLD signal for {self.symbol}: {reason}")

        return {
            'signal': 'HOLD',
            'confidence': 0.0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'risk_reward_ratio': 0.0,
            'reasons': [reason],
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
        }

    def _get_pip_value(self) -> float:
        """Get pip value for the symbol."""
        # Default pip values for common symbols
        pip_values = {
            'XAUUSD': 0.1,
            'EURUSD': 0.0001,
            'GBPUSD': 0.0001,
            'USDJPY': 0.01,
            'USDCHF': 0.0001,
            'AUDUSD': 0.0001,
            'NZDUSD': 0.0001,
            'USDCAD': 0.0001,
        }

        return pip_values.get(self.symbol, 0.0001)

    def get_latest_signal(self, data: pd.DataFrame) -> SignalResult:
        """Get the latest signal from the strategy (compatibility method)."""
        signal_dict = self.generate_signal(data)

        return SignalResult(
            signal=signal_dict['signal'],
            confidence=signal_dict['confidence'],
            features={
                'entry_price': signal_dict['entry_price'],
                'stop_loss': signal_dict['stop_loss'],
                'take_profit': signal_dict['take_profit'],
                'risk_reward_ratio': signal_dict['risk_reward_ratio'],
                'reasons': signal_dict['reasons'],
            },
            metadata={
                'symbol': signal_dict['symbol'],
                'timeframe': signal_dict['timeframe'],
                'timestamp': signal_dict['timestamp'],
            },
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for the entire dataset (compatibility method)."""
        if not self.validate_data(data):
            return data

        df = self.calculate_indicators(data.copy())

        # Generate signals for each row
        signals = []
        for i in range(50, len(df)):  # Start from 50 to have enough data for indicators
            window = df.iloc[: i + 1]
            signal = self.generate_signal(window)
            signals.append(signal['signal'])

        # Pad the beginning with HOLD signals
        signals = ['HOLD'] * 50 + signals

        df['signal'] = signals[: len(df)]

        return df
