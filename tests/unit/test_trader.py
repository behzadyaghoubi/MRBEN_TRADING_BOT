"""
Unit tests for MT5LiveTrader class.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.trader import MT5LiveTrader


class TestMT5LiveTrader(unittest.TestCase):
    """Test cases for MT5LiveTrader class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the config to avoid file dependencies
        with patch('src.config.settings.MT5Config') as mock_config:
            mock_config.return_value.LOG_LEVEL = "INFO"
            mock_config.return_value.LOG_FILE = "logs/test.log"
            mock_config.return_value.TRADE_LOG_PATH = "data/test_trades.csv"
            mock_config.return_value.SYMBOL = "XAUUSD.PRO"
            mock_config.return_value.TIMEFRAME_MIN = 15
            mock_config.return_value.BARS = 500
            mock_config.return_value.MAGIC = 20250721
            mock_config.return_value.SESSIONS = ["London", "NY"]
            mock_config.return_value.MAX_SPREAD_POINTS = 200
            mock_config.return_value.USE_RISK_BASED_VOLUME = True
            mock_config.return_value.FIXED_VOLUME = 0.01
            mock_config.return_value.SLEEP_SECONDS = 12
            mock_config.return_value.RETRY_DELAY = 5
            mock_config.return_value.CONSECUTIVE_SIGNALS_REQUIRED = 1
            mock_config.return_value.LSTM_TIMESTEPS = 50
            mock_config.return_value.COOLDOWN_SECONDS = 180
            mock_config.return_value.BASE_RISK = 0.01
            mock_config.return_value.MIN_LOT = 0.01
            mock_config.return_value.MAX_LOT = 2.0
            mock_config.return_value.MAX_OPEN_TRADES = 3
            mock_config.return_value.MAX_DAILY_LOSS = 0.02
            mock_config.return_value.MAX_TRADES_PER_DAY = 10
            mock_config.return_value.SESSION_TZ = "Etc/UTC"
            mock_config.return_value.config_data = {
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
                MT5Config=mock_config,
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

                # Mock MT5 availability
                with patch('src.core.trader.MT5_AVAILABLE', False):
                    self.trader = MT5LiveTrader()

    def test_initialization(self):
        """Test trader initialization."""
        self.assertIsNotNone(self.trader)
        self.assertIsNotNone(self.trader.config)
        self.assertIsNotNone(self.trader.metrics)
        self.assertFalse(self.trader.running)
        self.assertIsNotNone(self.trader.run_id)

    def test_setup_logging(self):
        """Test logging setup."""
        self.assertIsNotNone(self.trader.logger)
        self.assertEqual(self.trader.logger.level, 20)  # INFO level

    def test_initialize_components(self):
        """Test component initialization."""
        # This should not raise any exceptions
        self.assertIsNotNone(self.trader.risk_manager)
        self.assertIsNotNone(self.trader.trade_executor)
        self.assertIsNotNone(self.trader.data_manager)
        self.assertIsNotNone(self.trader.ai_system)

    def test_validate_system(self):
        """Test system validation."""
        self.assertTrue(self.trader._validate_system())

    def test_preflight_check(self):
        """Test preflight checks."""
        # Mock file system operations
        with patch('os.path.exists', return_value=True):
            with patch('os.makedirs'):
                self.assertTrue(self.trader._preflight_check())

    def test_should_continue_trading(self):
        """Test trading continuation logic."""
        # Test basic conditions
        self.assertTrue(self.trader._should_continue_trading())

        # Test when system is stopped
        self.trader.running = False
        self.assertFalse(self.trader._should_continue_trading())

    def test_is_trading_session(self):
        """Test trading session logic."""
        # Test with no session restrictions
        self.trader.config.SESSIONS = []
        self.assertTrue(self.trader._is_trading_session())

        # Test with session restrictions
        self.trader.config.SESSIONS = ["London", "NY"]
        # Mock timezone to return a specific hour
        with patch('pytz.timezone') as mock_tz:
            mock_now = Mock()
            mock_now.hour = 10  # London session
            mock_tz.return_value.localize.return_value = mock_now
            self.assertTrue(self.trader._is_trading_session())

    def test_can_open_new_trade(self):
        """Test trade opening eligibility."""
        # Test basic conditions
        self.assertTrue(self.trader._can_open_new_trade(0))

        # Test max open trades limit
        self.assertFalse(self.trader._can_open_new_trade(3))

        # Test cooldown period
        self.trader.cooldown_until = datetime.now() + timedelta(seconds=60)
        self.assertFalse(self.trader._can_open_new_trade(0))

    def test_is_new_bar(self):
        """Test new bar detection."""
        # Test with no previous bar
        self.trader.last_bar_time = None
        mock_df = Mock()
        mock_df.__getitem__.return_value.iloc = [-1]
        mock_df.__getitem__.return_value.iloc[-1] = datetime.now()
        self.assertTrue(self.trader._is_new_bar(mock_df))

        # Test with same bar time
        current_time = datetime.now()
        self.trader.last_bar_time = current_time
        mock_df.__getitem__.return_value.iloc[-1] = current_time
        self.assertFalse(self.trader._is_new_bar(mock_df))

    def test_get_status(self):
        """Test status retrieval."""
        status = self.trader.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn('running', status)
        self.assertIn('run_id', status)
        self.assertIn('symbol', status)

    def test_start_stop(self):
        """Test start and stop functionality."""
        # Mock validation and preflight
        with patch.object(self.trader, '_validate_system', return_value=True):
            with patch.object(self.trader, '_preflight_check', return_value=True):
                with patch.object(self.trader, '_bootstrap_trailing'):
                    with patch('threading.Thread'):
                        # Test start
                        self.trader.start()
                        self.assertTrue(self.trader.running)

                        # Test stop
                        self.trader.stop()
                        self.assertFalse(self.trader.running)

    def test_error_handling(self):
        """Test error handling in trading loop."""
        # Mock the trading loop to test error handling
        with patch.object(
            self.trader, '_should_continue_trading', side_effect=Exception("Test error")
        ):
            with patch.object(self.trader, 'metrics') as mock_metrics:
                # This should not crash and should record the error
                self.trader._trading_loop()
                mock_metrics.record_error.assert_called()

    def test_memory_management(self):
        """Test memory management functionality."""
        # Test memory check interval
        self.trader.last_memory_check = 0
        self.trader.memory_check_interval = 1

        with patch('time.time', return_value=2):
            with patch.object(self.trader, 'logger') as mock_logger:
                # This should trigger memory check
                self.trader._trading_loop()
                # Verify memory check was called
                mock_logger.info.assert_called()

    def test_signal_generation(self):
        """Test signal generation process."""
        # Mock market data
        mock_df = Mock()
        mock_df.__len__ = lambda x: 100

        mock_tick = {'time': datetime.now(), 'bid': 2000.0, 'ask': 2000.1, 'volume': 100}

        # Mock AI system
        self.trader.ai_system.generate_ensemble_signal.return_value = {
            'signal': 1,
            'confidence': 0.8,
            'score': 0.6,
            'source': 'Test AI',
        }

        # Test signal generation
        signal = self.trader._generate_trading_signal(mock_df, mock_tick)
        self.assertIsNotNone(signal)
        self.assertEqual(signal['signal'], 1)
        self.assertEqual(signal['confidence'], 0.8)

    def test_trade_execution_conditions(self):
        """Test trade execution conditions."""
        signal_data = {'signal': 1, 'confidence': 0.8, 'score': 0.6}

        # Test basic conditions
        self.assertTrue(self.trader._should_execute_trade(signal_data, 0))

        # Test confidence threshold
        signal_data['confidence'] = 0.1
        self.assertFalse(self.trader._should_execute_trade(signal_data, 0))

        # Test consecutive signals requirement
        signal_data['confidence'] = 0.8
        self.trader.config.CONSECUTIVE_SIGNALS_REQUIRED = 2
        self.assertFalse(self.trader._should_execute_trade(signal_data, 0))

    def test_cleanup_resources(self):
        """Test resource cleanup."""
        # Mock trailing registry
        self.trader.trailing_registry = {1: {'test': 'data'}}

        # Test cleanup
        self.trader._cleanup_resources()
        self.assertEqual(len(self.trader.trailing_registry), 0)

    def test_bootstrap_trailing(self):
        """Test trailing stop bootstrap."""
        # Mock open positions
        mock_position = Mock()
        mock_position.type = 0  # Buy position
        mock_position.price_open = 2000.0
        mock_position.sl = 1990.0
        mock_position.tp = 2010.0

        with patch('src.core.trader._get_open_positions', return_value={1: mock_position}):
            with patch('src.core.trader.mt5.symbol_info_tick') as mock_tick:
                mock_tick.return_value.bid = 2000.0
                mock_tick.return_value.ask = 2000.1

                self.trader._bootstrap_trailing()
                self.assertIn(1, self.trader.trailing_registry)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'trader'):
            try:
                self.trader.stop()
            except:
                pass


if __name__ == '__main__':
    unittest.main()
