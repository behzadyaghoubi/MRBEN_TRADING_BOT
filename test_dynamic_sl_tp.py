#!/usr/bin/env python3
"""
Test for Dynamic SL/TP System
Tests the new advanced SL/TP calculation with confidence, volatility, spread, and structure
"""

import json
import os
import sys
import unittest
from unittest.mock import Mock, patch

import pandas as pd

# Add the current directory to Python path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock MT5 before importing our modules
sys.modules['MetaTrader5'] = Mock()
sys.modules['MetaTrader5'].TRADE_RETCODE_DONE = 10009
sys.modules['MetaTrader5'].ORDER_TYPE_BUY = 0
sys.modules['MetaTrader5'].ORDER_TYPE_SELL = 1
sys.modules['MetaTrader5'].TIMEFRAME_M15 = 15
sys.modules['MetaTrader5'].TIMEFRAME_M5 = 5
sys.modules['MetaTrader5'].TRADE_ACTION_DEAL = 1
sys.modules['MetaTrader5'].ORDER_TIME_GTC = 0
sys.modules['MetaTrader5'].ORDER_FILLING_IOC = 1

# Now import our modules
from live_trader_clean import MT5Config, MT5LiveTrader, _linmap, _rolling_atr, _swing_extrema


class TestDynamicSLTP(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock config for testing
        self.mock_config_data = {
            "credentials": {"login": 12345, "password": "test", "server": "test-server"},
            "trading": {
                "symbol": "XAUUSD.PRO",
                "timeframe": 15,
                "bars": 500,
                "magic_number": 20250721,
                "sessions": ["24h"],
                "max_spread_points": 200,
                "use_risk_based_volume": False,
                "fixed_volume": 0.1,
                "sleep_seconds": 12,
                "retry_delay": 5,
                "consecutive_signals_required": 1,
                "lstm_timesteps": 50,
                "cooldown_seconds": 180,
            },
            "risk": {
                "base_risk": 0.01,
                "min_lot": 0.01,
                "max_lot": 2.0,
                "max_open_trades": 3,
                "max_daily_loss": 0.02,
                "max_trades_per_day": 10,
                "sl_atr_multiplier": 1.6,
                "tp_atr_multiplier": 3.0,
            },
            "advanced": {"dynamic_spread_atr_frac": 0.15, "swing_lookback": 12},
            "flags": {"demo_mode": True},
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "logs/trading_bot.log",
                "trade_log_path": "data/trade_log_gold.csv",
            },
            "models": {
                "use_rl": True,
                "use_lstm": True,
                "use_technical": True,
                "ml_filter_threshold": 0.5,
            },
            "notifications": {
                "telegram_enabled": False,
                "email_enabled": False,
                "daily_summary": True,
            },
        }

        # Mock the config file
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                self.mock_config_data
            )
            self.config = MT5Config()

    def test_linmap_function(self):
        """Test the linear mapping function."""
        print("\n🧪 Testing Linear Mapping Function")

        # Test normal mapping
        result = _linmap(0.5, 0, 1, 10, 20)
        self.assertEqual(result, 15.0)

        # Test clipping
        result = _linmap(2.0, 0, 1, 10, 20)
        self.assertEqual(result, 20.0)

        result = _linmap(-1.0, 0, 1, 10, 20)
        self.assertEqual(result, 10.0)

        # Test edge case
        result = _linmap(0.5, 0, 0, 10, 20)
        self.assertEqual(result, 15.0)

        print("✅ Linear mapping function works correctly")

    def test_rolling_atr_function(self):
        """Test the rolling ATR calculation."""
        print("\n🧪 Testing Rolling ATR Function")

        # Create test data
        df = pd.DataFrame(
            {
                'high': [2005.0, 2008.0, 2010.0, 2007.0, 2009.0],
                'low': [1998.0, 2000.0, 2003.0, 2001.0, 2004.0],
                'close': [2002.0, 2005.0, 2008.0, 2004.0, 2006.0],
            }
        )

        atr = _rolling_atr(df, period=3)
        self.assertIsInstance(atr, float)
        self.assertGreater(atr, 0)

        print(f"✅ Rolling ATR calculated: {atr:.4f}")

    def test_swing_extrema_function(self):
        """Test the swing extrema calculation."""
        print("\n🧪 Testing Swing Extrema Function")

        # Create test data
        df = pd.DataFrame(
            {
                'high': [2005.0, 2008.0, 2010.0, 2007.0, 2009.0],
                'low': [1998.0, 2000.0, 2003.0, 2001.0, 2004.0],
            }
        )

        lo, hi = _swing_extrema(df, bars=3)
        self.assertEqual(lo, 2001.0)  # min of last 3 lows
        self.assertEqual(hi, 2010.0)  # max of last 3 highs

        print(f"✅ Swing extrema: low={lo}, high={hi}")

    @patch('live_trader_clean.mt5')
    def test_dynamic_sl_tp_calculation(self, mock_mt5):
        """Test the dynamic SL/TP calculation."""
        print("\n🧪 Testing Dynamic SL/TP Calculation")

        # Create trader instance
        trader = MT5LiveTrader()

        # Mock MT5 symbol info
        mock_symbol_info = Mock()
        mock_symbol_info.point = 0.01
        mock_mt5.symbol_info.return_value = mock_symbol_info

        # Mock symbol tick
        mock_tick = Mock()
        mock_tick.ask = 2005.0
        mock_tick.bid = 2004.0
        mock_mt5.symbol_info_tick.return_value = mock_tick

        # Create test market data
        df = pd.DataFrame(
            {
                'high': [2005.0, 2008.0, 2010.0, 2007.0, 2009.0, 2012.0, 2015.0],
                'low': [1998.0, 2000.0, 2003.0, 2001.0, 2004.0, 2007.0, 2010.0],
                'close': [2002.0, 2005.0, 2008.0, 2004.0, 2006.0, 2009.0, 2012.0],
                'atr': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],  # Add ATR column
            }
        )

        # Test BUY signal with high confidence
        entry = 2010.0
        signal = 1  # BUY
        confidence = 0.8  # High confidence

        sl_price, tp_price, meta = trader._calculate_dynamic_sl_tp(
            df, entry, signal, confidence, "XAUUSD.PRO"
        )

        print("📊 BUY Signal Results:")
        print(f"   Entry: {entry}")
        print(f"   SL: {sl_price:.4f}")
        print(f"   TP: {tp_price:.4f}")
        print(f"   ATR: {meta.get('atr', 0):.4f}")
        print(f"   Vol Ratio: {meta.get('vol_ratio', 0):.4f}")
        print(f"   SL Mult: {meta.get('sl_mult', 0):.4f}")
        print(f"   TP Mult: {meta.get('tp_mult', 0):.4f}")
        print(f"   R: {meta.get('R', 0):.4f}")

        # Verify results
        self.assertLess(sl_price, entry)  # SL below entry for BUY
        self.assertGreater(tp_price, entry)  # TP above entry for BUY
        self.assertGreater(meta.get('R', 0), 1.0)  # R should be > 1
        self.assertIsInstance(meta.get('sl_mult'), float)
        self.assertIsInstance(meta.get('tp_mult'), float)

        # Test SELL signal with low confidence
        signal = -1  # SELL
        confidence = 0.4  # Low confidence

        sl_price, tp_price, meta = trader._calculate_dynamic_sl_tp(
            df, entry, signal, confidence, "XAUUSD.PRO"
        )

        print("📊 SELL Signal Results:")
        print(f"   Entry: {entry}")
        print(f"   SL: {sl_price:.4f}")
        print(f"   TP: {tp_price:.4f}")
        print(f"   ATR: {meta.get('atr', 0):.4f}")
        print(f"   Vol Ratio: {meta.get('vol_ratio', 0):.4f}")
        print(f"   SL Mult: {meta.get('sl_mult', 0):.4f}")
        print(f"   TP Mult: {meta.get('tp_mult', 0):.4f}")
        print(f"   R: {meta.get('R', 0):.4f}")

        # Verify results
        self.assertGreater(sl_price, entry)  # SL above entry for SELL
        self.assertLess(tp_price, entry)  # TP below entry for SELL
        self.assertGreater(meta.get('R', 0), 1.0)  # R should be > 1

        print("✅ Dynamic SL/TP calculation works correctly")

    @patch('live_trader_clean.mt5')
    def test_confidence_impact(self, mock_mt5):
        """Test how confidence affects SL/TP multipliers."""
        print("\n🧪 Testing Confidence Impact on SL/TP")

        # Create trader instance
        trader = MT5LiveTrader()

        # Mock MT5
        mock_symbol_info = Mock()
        mock_symbol_info.point = 0.01
        mock_mt5.symbol_info.return_value = mock_symbol_info

        mock_tick = Mock()
        mock_tick.ask = 2005.0
        mock_tick.bid = 2004.0
        mock_mt5.symbol_info_tick.return_value = mock_tick

        # Create test data
        df = pd.DataFrame(
            {
                'high': [2005.0, 2008.0, 2010.0, 2007.0, 2009.0],
                'low': [1998.0, 2000.0, 2003.0, 2001.0, 2004.0],
                'close': [2002.0, 2005.0, 2008.0, 2004.0, 2006.0],
                'atr': [2.0, 2.1, 2.2, 2.3, 2.4],
            }
        )

        entry = 2010.0
        signal = 1  # BUY

        # Test with low confidence
        sl_low, tp_low, meta_low = trader._calculate_dynamic_sl_tp(
            df, entry, signal, 0.4, "XAUUSD.PRO"
        )

        # Test with high confidence
        sl_high, tp_high, meta_high = trader._calculate_dynamic_sl_tp(
            df, entry, signal, 0.8, "XAUUSD.PRO"
        )

        print("📊 Low Confidence (0.4):")
        print(f"   SL Mult: {meta_low.get('sl_mult', 0):.4f}")
        print(f"   TP Mult: {meta_low.get('tp_mult', 0):.4f}")

        print("📊 High Confidence (0.8):")
        print(f"   SL Mult: {meta_high.get('sl_mult', 0):.4f}")
        print(f"   TP Mult: {meta_high.get('tp_mult', 0):.4f}")

        # High confidence should have tighter SL and wider TP
        self.assertLess(meta_high.get('sl_mult', 0), meta_low.get('sl_mult', 0))
        self.assertGreater(meta_high.get('tp_mult', 0), meta_low.get('tp_mult', 0))

        print("✅ Confidence correctly affects SL/TP multipliers")

    @patch('live_trader_clean.mt5')
    def test_minimum_r_ratio(self, mock_mt5):
        """Test that minimum R ratio of 1.5 is enforced."""
        print("\n🧪 Testing Minimum R Ratio Enforcement")

        # Create trader instance
        trader = MT5LiveTrader()

        # Mock MT5
        mock_symbol_info = Mock()
        mock_symbol_info.point = 0.01
        mock_mt5.symbol_info.return_value = mock_symbol_info

        mock_tick = Mock()
        mock_tick.ask = 2005.0
        mock_tick.bid = 2004.0
        mock_mt5.symbol_info_tick.return_value = mock_tick

        # Create test data with very small ATR to test R enforcement
        df = pd.DataFrame(
            {
                'high': [2000.1, 2000.2, 2000.3, 2000.4, 2000.5],
                'low': [1999.9, 2000.0, 2000.1, 2000.2, 2000.3],
                'close': [2000.0, 2000.1, 2000.2, 2000.3, 2000.4],
                'atr': [0.1, 0.1, 0.1, 0.1, 0.1],  # Very small ATR
            }
        )

        entry = 2000.0
        signal = 1  # BUY
        confidence = 0.5

        sl_price, tp_price, meta = trader._calculate_dynamic_sl_tp(
            df, entry, signal, confidence, "XAUUSD.PRO"
        )

        r_ratio = meta.get('R', 0)
        print(f"📊 R Ratio: {r_ratio:.4f}")

        # R should be at least 1.5
        self.assertGreaterEqual(r_ratio, 1.5)

        print("✅ Minimum R ratio of 1.5 is enforced")


if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    print("🚀 Starting Dynamic SL/TP System Tests")
    print("=" * 60)

    # Run the tests
    unittest.main(verbosity=2)
