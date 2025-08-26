"""
Smoke tests for MR BEN Trading System.
These tests verify basic functionality without requiring external dependencies.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSmokeTests:
    """Smoke tests for basic system functionality."""

    def test_imports_work(self):
        """Test that all modules can be imported."""
        try:
            from src.config.settings import MT5Config
            from src.core.exceptions import TradingSystemError
            from src.core.metrics import PerformanceMetrics
            from src.utils.error_handler import error_handler
            from src.utils.helpers import enforce_min_distance_and_round, round_price

            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_config_structure(self):
        """Test configuration structure."""
        from src.config.settings import MT5Config

        # Mock config file
        mock_config = {
            "credentials": {"login": "12345", "password": "test", "server": "test"},
            "flags": {"demo_mode": True},
            "trading": {
                "symbol": "XAUUSD.PRO",
                "timeframe": 15,
                "bars": 500,
                "magic_number": 20250721,
                "sessions": ["London"],
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
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "logs/trading_bot.log",
                "trade_log_path": "data/trade_log_gold.csv",
            },
            "models": {},
            "session": {"timezone": "Etc/UTC"},
        }

        with patch("builtins.open", MagicMock()) as mock_file:
            mock_file.return_value.__enter__.return_value.read.return_value = str(mock_config)
            mock_file.return_value.__enter__.return_value.read.return_value = '{}'

            # Mock json.load to return our config
            with patch("json.load", return_value=mock_config):
                with patch("os.getenv", return_value="test_password"):
                    config = MT5Config()

                    # Verify basic structure
                    assert hasattr(config, 'SYMBOL')
                    assert hasattr(config, 'TIMEFRAME_MIN')
                    assert hasattr(config, 'BASE_RISK')
                    assert hasattr(config, 'LOG_LEVEL')

    def test_exceptions_work(self):
        """Test that custom exceptions work."""
        from src.core.exceptions import DataError, MT5ConnectionError, RiskError, TradingSystemError

        # Test exception inheritance
        assert issubclass(MT5ConnectionError, TradingSystemError)
        assert issubclass(DataError, TradingSystemError)
        assert issubclass(RiskError, TradingSystemError)

        # Test exception instantiation
        try:
            raise MT5ConnectionError("Test connection error")
        except MT5ConnectionError as e:
            assert str(e) == "Test connection error"

    def test_metrics_work(self):
        """Test that performance metrics work."""
        from src.core.metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        # Test basic functionality
        metrics.record_cycle(0.1)
        metrics.record_trade()
        metrics.record_error()

        stats = metrics.get_stats()

        assert 'uptime_seconds' in stats
        assert 'cycle_count' in stats
        assert 'total_trades' in stats
        assert 'error_rate' in stats
        assert stats['cycle_count'] == 1
        assert stats['total_trades'] == 1
        assert stats['error_count'] == 1

    def test_helpers_work(self):
        """Test that helper functions work."""
        from src.utils.helpers import enforce_min_distance_and_round, round_price

        # Test round_price (should work without MT5)
        result = round_price("XAUUSD", 1234.5678)
        assert isinstance(result, float)
        assert result > 0

        # Test enforce_min_distance_and_round
        sl, tp = enforce_min_distance_and_round("XAUUSD", 1000.0, 990.0, 1010.0, True)
        assert isinstance(sl, float)
        assert isinstance(tp, float)
        assert sl < 1000.0  # SL should be below entry
        assert tp > 1000.0  # TP should be above entry

    def test_error_handler_works(self):
        """Test that error handler works."""
        import logging

        from src.utils.error_handler import error_handler

        logger = logging.getLogger("test")

        # Test successful operation
        with error_handler(logger, "test_operation", None):
            result = 42
            assert result == 42

        # Test error handling
        with error_handler(logger, "test_operation", "fallback"):
            raise ValueError("Test error")
            # Should not reach here

    @patch('src.utils.helpers.MT5_AVAILABLE', False)
    def test_helpers_without_mt5(self):
        """Test helper functions work without MT5."""
        from src.utils.helpers import is_spread_ok

        # Should return default values when MT5 not available
        ok, spread, threshold = is_spread_ok("XAUUSD", 200)
        assert ok is True
        assert spread == 0.0
        assert threshold == 200.0


if __name__ == "__main__":
    # Run smoke tests
    pytest.main([__file__, "-v"])
