"""
Data management for MR BEN Trading System.
Handles data fetching, preprocessing, and technical indicators.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# Global MT5 availability flag
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class MT5DataManager:
    """Manages data fetching and preprocessing for trading system."""

    def __init__(self, symbol: str, timeframe_min: int):
        """
        Initialize data manager.

        Args:
            symbol: Trading symbol
            timeframe_min: Timeframe in minutes
        """
        self.symbol = symbol
        self.timeframe_min = timeframe_min
        self.mt5_connected = False
        self.current_data: pd.DataFrame | None = None

        if MT5_AVAILABLE:
            self._initialize_mt5()
        else:
            print("⚠️ MT5 not available, synthetic data mode.")

    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection for data."""
        try:
            if not mt5.initialize():
                print(f"❌ MT5 initialize failed: {mt5.last_error()}")
                return False
            print(f"✅ MT5 initialized for data: {self.symbol}")
            self.mt5_connected = True
            return True
        except Exception as e:
            print(f"❌ MT5 init error: {e}")
            return False

    def _tf_to_mt5(self, minutes: int) -> int:
        """Map minute timeframe to MT5 enum."""
        m = minutes
        if m == 1:
            return mt5.TIMEFRAME_M1
        if m == 2:
            return mt5.TIMEFRAME_M2
        if m == 3:
            return mt5.TIMEFRAME_M3
        if m == 4:
            return mt5.TIMEFRAME_M4
        if m == 5:
            return mt5.TIMEFRAME_M5
        if m == 10:
            return mt5.TIMEFRAME_M10
        if m == 15:
            return mt5.TIMEFRAME_M15
        if m == 30:
            return mt5.TIMEFRAME_M30
        if m == 60:
            return mt5.TIMEFRAME_H1
        return mt5.TIMEFRAME_M15

    def get_latest_data(self, bars: int = 500) -> pd.DataFrame:
        """
        Get latest market data.

        Args:
            bars: Number of bars to fetch

        Returns:
            DataFrame with OHLCV data and indicators
        """
        if not MT5_AVAILABLE or not self.mt5_connected:
            return self._get_synthetic_data(bars)

        try:
            tf = self._tf_to_mt5(self.timeframe_min)
            rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)
            if rates is None or len(rates) == 0:
                print("⚠️ MT5 rates empty, using synthetic data.")
                return self._get_synthetic_data(bars)

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = self._indicators(df)
            self.current_data = df
            return df
        except Exception as e:
            print(f"❌ get_latest_data error: {e}")
            return self._get_synthetic_data(bars)

    def get_current_tick(self) -> dict[str, Any] | None:
        """
        Get current tick data.

        Returns:
            Dictionary with tick information or None
        """
        if not MT5_AVAILABLE:
            return None
        try:
            t = mt5.symbol_info_tick(self.symbol)
            if not t:
                return None

            # Make tick time timezone-aware using trader config timezone
            try:
                import pytz

                with open('config.json', encoding='utf-8') as f:
                    import json

                    cfg = json.load(f)
                tzname = cfg.get("session", {}).get("timezone", "Etc/UTC")
                tz = pytz.timezone(tzname)
            except Exception:
                tz = None

            tick_dt_utc = datetime.fromtimestamp(t.time, tz=UTC)  # epoch is UTC
            tick_time = tick_dt_utc.astimezone(tz) if tz else tick_dt_utc

            # Staleness check in the same tz basis (seconds don't depend on tz)
            if (datetime.now(UTC) - tick_dt_utc).total_seconds() > 300:
                return None

            return {'bid': t.bid, 'ask': t.ask, 'time': tick_time, 'volume': t.volume}
        except Exception:
            return None

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # RSI
        delta = df['close'].diff().fillna(0)
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(14).mean()

        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        return df.ffill().bfill()

    def _get_synthetic_data(self, bars: int) -> pd.DataFrame:
        """Generate synthetic data for testing/demo mode."""
        base = 3300.0
        data = []
        now = datetime.now()
        for i in range(bars):
            op = base + np.random.uniform(-10, 10)
            cl = op + np.random.uniform(-5, 5)
            hi = max(op, cl) + np.random.uniform(0, 3)
            lo = min(op, cl) - np.random.uniform(0, 3)
            vol = np.random.randint(100, 1000)
            data.append(
                {
                    'time': now - timedelta(minutes=i * self.timeframe_min),
                    'open': op,
                    'high': hi,
                    'low': lo,
                    'close': cl,
                    'tick_volume': vol,
                }
            )
        df = pd.DataFrame(data)[::-1].reset_index(drop=True)
        return self._indicators(df)

    def refresh_symbol(self, symbol: str) -> bool:
        """
        Refresh symbol data and reinitialize if needed.

        Args:
            symbol: Symbol to refresh

        Returns:
            True if successful
        """
        try:
            if not MT5_AVAILABLE:
                return False
            if not mt5.symbol_select(symbol, True):
                print(f"❌ Cannot select symbol {symbol}")
                return False
            print(f"✅ Symbol {symbol} refreshed")
            return True
        except Exception as e:
            print(f"❌ Symbol refresh error: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        if MT5_AVAILABLE:
            mt5.shutdown()
