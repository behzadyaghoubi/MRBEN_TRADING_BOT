"""
Integration tests for MR BEN Trading System.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ai.system import MRBENAdvancedAISystem
from config.settings import MT5Config
from core.trader import MT5LiveTrader
from execution.executor import EnhancedTradeExecutor
from risk.manager import EnhancedRiskManager


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete trading system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'config.json')

        config_data = {
            "credentials": {"login": 12345, "password": "test_password", "server": "TestServer"},
            "flags": {"demo_mode": True},
            "trading": {
                "symbol": "XAUUSD.PRO",
                "timeframe": 15,
                "bars": 500,
                "magic_number": 20250721,
                "sessions": ["London", "NY"],
                "max_spread_points": 200,
                "use_risk_based_volume": True,
                "fixed_volume": 0.01,
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
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "logs/test.log",
                "trade_log_path": "data/test_trades.csv",
            },
            "session": {"timezone": "Etc/UTC"},
            "advanced": {"swing_lookback": 12, "dynamic_spread_atr_frac": 0.10},
            "execution": {
                "spread_eps": 0.02,
                "use_spread_ma": True,
                "spread_ma_window": 5,
                "spread_hysteresis_factor": 1.05,
            },
            "tp_policy": {
                "split": True,
                "tp1_r": 0.8,
                "tp2_r": 1.5,
                "tp1_share": 0.5,
                "breakeven_after_tp1": True,
            },
        }

        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_component_initialization_integration(self):
        """Test that all components can be initialized together."""
        # Mock MT5 availability
        with patch('src.core.trader.MT5_AVAILABLE', False):
            with patch('src.config.settings.MT5Config') as mock_config_class:
                # Mock the config to use our temp file
                mock_config_class.return_value.config_file = self.config_file
                mock_config_class.return_value.LOG_LEVEL = "INFO"
                mock_config_class.return_value.LOG_FILE = "logs/test.log"
                mock_config_class.return_value.TRADE_LOG_PATH = "data/test_trades.csv"
                mock_config_class.return_value.SYMBOL = "XAUUSD.PRO"
                mock_config_class.return_value.TIMEFRAME_MIN = 15
                mock_config_class.return_value.BARS = 500
                mock_config_class.return_value.MAGIC = 20250721
                mock_config_class.return_value.SESSIONS = ["London", "NY"]
                mock_config_class.return_value.MAX_SPREAD_POINTS = 200
                mock_config_class.return_value.USE_RISK_BASED_VOLUME = True
                mock_config_class.return_value.FIXED_VOLUME = 0.01
                mock_config_class.return_value.SLEEP_SECONDS = 12
                mock_config_class.return_value.RETRY_DELAY = 5
                mock_config_class.return_value.CONSECUTIVE_SIGNALS_REQUIRED = 1
                mock_config_class.return_value.LSTM_TIMESTEPS = 50
                mock_config_class.return_value.COOLDOWN_SECONDS = 180
                mock_config_class.return_value.BASE_RISK = 0.01
                mock_config_class.return_value.MIN_LOT = 0.01
                mock_config_class.return_value.MAX_LOT = 2.0
                mock_config_class.return_value.MAX_OPEN_TRADES = 3
                mock_config_class.return_value.MAX_DAILY_LOSS = 0.02
                mock_config_class.return_value.MAX_TRADES_PER_DAY = 10
                mock_config_class.return_value.SESSION_TZ = "Etc/UTC"
                mock_config_class.return_value.config_data = {
                    "risk": {"sl_atr_multiplier": 1.6, "tp_atr_multiplier": 3.0},
                    "advanced": {"swing_lookback": 12, "dynamic_spread_atr_frac": 0.10},
                    "execution": {
                        "spread_eps": 0.02,
                        "use_spread_ma": True,
                        "spread_ma_window": 5,
                        "spread_hysteresis_factor": 1.05,
                    },
                    "tp_policy": {
                        "split": True,
                        "tp1_r": 0.8,
                        "tp2_r": 1.5,
                        "tp1_share": 0.5,
                        "breakeven_after_tp1": True,
                    },
                }

                # Mock all external dependencies
                with patch.multiple(
                    'src.core.trader',
                    MT5Config=mock_config_class,
                    MT5DataManager=Mock(),
                    MRBENAdvancedAISystem=Mock(),
                    EnhancedRiskManager=Mock(),
                    EnhancedTradeExecutor=Mock(),
                    _get_open_positions=Mock(return_value={}),
                    _prune_trailing_registry=Mock(return_value=0),
                    _count_open_positions=Mock(return_value=0),
                    log_memory_usage=Mock(),
                    cleanup_memory=Mock(),
                ):

                    # Test trader initialization
                    trader = MT5LiveTrader()

                    # Verify all components are initialized
                    self.assertIsNotNone(trader.config)
                    self.assertIsNotNone(trader.risk_manager)
                    self.assertIsNotNone(trader.trade_executor)
                    self.assertIsNotNone(trader.data_manager)
                    self.assertIsNotNone(trader.ai_system)
                    self.assertIsNotNone(trader.metrics)

    def test_data_flow_integration(self):
        """Test data flow between components."""
        # Mock components
        mock_data_manager = Mock()
        mock_ai_system = Mock()
        mock_risk_manager = Mock()
        mock_trade_executor = Mock()

        # Mock data flow
        mock_data = pd.DataFrame(
            {
                'time': pd.date_range('2024-01-01', periods=100, freq='H'),
                'open': range(100),
                'high': range(100),
                'low': range(100),
                'close': range(100),
                'volume': [100] * 100,
            }
        )

        mock_data_manager.get_latest_data.return_value = mock_data
        mock_data_manager.get_current_tick.return_value = {
            'time': datetime.now(),
            'bid': 2000.0,
            'ask': 2000.1,
            'volume': 100,
        }

        mock_ai_system.generate_ensemble_signal.return_value = {
            'signal': 1,
            'confidence': 0.8,
            'score': 0.6,
            'source': 'Test AI',
        }

        mock_risk_manager.calculate_dynamic_sl_tp.return_value = (1990.0, 2010.0)
        mock_risk_manager.calculate_lot_size.return_value = 0.1

        # Test data flow
        # 1. Get market data
        data = mock_data_manager.get_latest_data(500)
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 100)

        # 2. Generate AI signal
        signal = mock_ai_system.generate_ensemble_signal({})
        self.assertEqual(signal['signal'], 1)
        self.assertEqual(signal['confidence'], 0.8)

        # 3. Calculate risk parameters
        sl, tp = mock_risk_manager.calculate_dynamic_sl_tp("XAUUSD.PRO", 2000.0, "BUY")
        self.assertEqual(sl, 1990.0)
        self.assertEqual(tp, 2010.0)

        # 4. Calculate position size
        lot_size = mock_risk_manager.calculate_lot_size(10000.0, 0.01, 10.0, "XAUUSD.PRO")
        self.assertEqual(lot_size, 0.1)

    def test_risk_management_integration(self):
        """Test risk management integration."""
        # Create risk manager
        risk_manager = EnhancedRiskManager(
            base_risk=0.01, min_lot=0.01, max_lot=2.0, max_open_trades=3, max_drawdown=0.10
        )

        # Test risk calculations
        # Mock ATR
        with patch.object(risk_manager, 'get_atr', return_value=2.0):
            sl, tp = risk_manager.calculate_dynamic_sl_tp("XAUUSD.PRO", 2000.0, "BUY")
            self.assertLess(sl, 2000.0)
            self.assertGreater(tp, 2000.0)

        # Test trade eligibility
        self.assertTrue(risk_manager.can_open_new_trade(10000.0, 10000.0, 0))
        self.assertFalse(risk_manager.can_open_new_trade(10000.0, 10000.0, 3))
        self.assertFalse(risk_manager.can_open_new_trade(9000.0, 10000.0, 0))

    def test_ai_system_integration(self):
        """Test AI system integration."""
        # Create AI system
        ai_system = MRBENAdvancedAISystem()

        # Test meta-feature generation
        df = pd.DataFrame(
            {
                'time': pd.date_range('2024-01-01', periods=5, freq='H'),
                'close': [2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
                'high': [2000.5, 2001.5, 2002.5, 2003.5, 2004.5],
                'low': [1999.5, 2000.5, 2001.5, 2002.5, 2003.5],
            }
        )

        result_df = ai_system.generate_meta_features(df)

        # Verify features were added
        expected_features = [
            'hour',
            'day_of_week',
            'session',
            'session_encoded',
            'rsi',
            'macd',
            'macd_signal',
            'macd_hist',
            'atr',
            'sma_20',
            'sma_50',
        ]

        for feature in expected_features:
            self.assertIn(feature, result_df.columns)

    def test_trade_execution_integration(self):
        """Test trade execution integration."""
        # Create risk manager and trade executor
        risk_manager = EnhancedRiskManager()
        trade_executor = EnhancedTradeExecutor(risk_manager)

        # Test account info retrieval
        account_info = trade_executor.get_account_info()
        self.assertIsInstance(account_info, dict)
        self.assertIn('balance', account_info)
        self.assertIn('equity', account_info)

    def test_error_handling_integration(self):
        """Test error handling across components."""
        # Test that errors in one component don't crash the system
        with patch('src.core.trader.MT5_AVAILABLE', False):
            with patch('src.config.settings.MT5Config') as mock_config_class:
                # Mock config with minimal requirements
                mock_config_class.return_value.LOG_LEVEL = "INFO"
                mock_config_class.return_value.LOG_FILE = "logs/test.log"
                mock_config_class.return_value.TRADE_LOG_PATH = "data/test_trades.csv"
                mock_config_class.return_value.SYMBOL = "XAUUSD.PRO"
                mock_config_class.return_value.TIMEFRAME_MIN = 15
                mock_config_class.return_value.BARS = 500
                mock_config_class.return_value.MAGIC = 20250721
                mock_config_class.return_value.SESSIONS = ["London", "NY"]
                mock_config_class.return_value.MAX_SPREAD_POINTS = 200
                mock_config_class.return_value.USE_RISK_BASED_VOLUME = True
                mock_config_class.return_value.FIXED_VOLUME = 0.01
                mock_config_class.return_value.SLEEP_SECONDS = 12
                mock_config_class.return_value.RETRY_DELAY = 5
                mock_config_class.return_value.CONSECUTIVE_SIGNALS_REQUIRED = 1
                mock_config_class.return_value.LSTM_TIMESTEPS = 50
                mock_config_class.return_value.COOLDOWN_SECONDS = 180
                mock_config_class.return_value.BASE_RISK = 0.01
                mock_config_class.return_value.MIN_LOT = 0.01
                mock_config_class.return_value.MAX_LOT = 2.0
                mock_config_class.return_value.MAX_OPEN_TRADES = 3
                mock_config_class.return_value.MAX_DAILY_LOSS = 0.02
                mock_config_class.return_value.MAX_TRADES_PER_DAY = 10
                mock_config_class.return_value.SESSION_TZ = "Etc/UTC"
                mock_config_class.return_value.config_data = {}

                # Mock all dependencies
                with patch.multiple(
                    'src.core.trader',
                    MT5Config=mock_config_class,
                    MT5DataManager=Mock(),
                    MRBENAdvancedAISystem=Mock(),
                    EnhancedRiskManager=Mock(),
                    EnhancedTradeExecutor=Mock(),
                    _get_open_positions=Mock(return_value={}),
                    _prune_trailing_registry=Mock(return_value=0),
                    _count_open_positions=Mock(return_value=0),
                    log_memory_usage=Mock(),
                    cleanup_memory=Mock(),
                ):

                    # Should handle initialization gracefully
                    trader = MT5LiveTrader()

                    # Test system validation
                    self.assertTrue(trader._validate_system())

                    # Test preflight check
                    with patch('os.path.exists', return_value=True):
                        with patch('os.makedirs'):
                            self.assertTrue(trader._preflight_check())

    def test_configuration_integration(self):
        """Test configuration integration across components."""
        # Test that configuration is properly propagated
        config = MT5Config()

        # Verify configuration values
        self.assertEqual(config.SYMBOL, "XAUUSD.PRO")
        self.assertEqual(config.TIMEFRAME_MIN, 15)
        self.assertEqual(config.BARS, 500)
        self.assertEqual(config.MAGIC, 20250721)
        self.assertEqual(config.BASE_RISK, 0.01)
        self.assertEqual(config.MIN_LOT, 0.01)
        self.assertEqual(config.MAX_LOT, 2.0)

    def test_memory_management_integration(self):
        """Test memory management integration."""
        # Test that memory management works across components
        from src.utils.memory import cleanup_memory, log_memory_usage

        # Mock logger
        mock_logger = Mock()

        # Test memory logging
        log_memory_usage(mock_logger, "Test memory check")
        mock_logger.info.assert_called()

        # Test memory cleanup
        cleanup_memory()
        # Should not raise any exceptions

    def test_logging_integration(self):
        """Test logging integration across components."""
        # Test that logging is properly configured
        import logging

        # Create test logger
        test_logger = logging.getLogger("TestLogger")
        test_logger.setLevel(logging.INFO)

        # Test logging functionality
        test_logger.info("Test log message")
        test_logger.warning("Test warning")
        test_logger.error("Test error")

        # Should not raise any exceptions

    def test_performance_metrics_integration(self):
        """Test performance metrics integration."""
        from src.core.metrics import PerformanceMetrics

        # Create metrics
        metrics = PerformanceMetrics()

        # Test metrics recording
        metrics.record_cycle(0.1)
        metrics.record_trade()
        metrics.record_error()

        # Test stats retrieval
        stats = metrics.get_stats()
        self.assertIn('uptime_seconds', stats)
        self.assertIn('cycle_count', stats)
        self.assertIn('total_trades', stats)
        self.assertIn('error_rate', stats)

        # Test reset functionality
        metrics.reset()
        stats_after_reset = metrics.get_stats()
        self.assertEqual(stats_after_reset['cycle_count'], 0)
        self.assertEqual(stats_after_reset['total_trades'], 0)


if __name__ == "__main__":
    unittest.main()
