"""
Unit tests for configuration module.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open
from src.config.settings import MT5Config


class TestMT5Config:
    """Test configuration management."""
    
    def test_config_initialization_success(self):
        """Test successful configuration initialization."""
        config_data = {
            "credentials": {
                "login": "12345",
                "password": "test_password",
                "server": "test_server"
            },
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
                "cooldown_seconds": 180
            },
            "risk": {
                "base_risk": 0.01,
                "min_lot": 0.01,
                "max_lot": 2.0,
                "max_open_trades": 3,
                "max_daily_loss": 0.02,
                "max_trades_per_day": 10
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "logs/trading_bot.log",
                "trade_log_path": "data/trade_log_gold.csv"
            },
            "models": {},
            "session": {"timezone": "Etc/UTC"}
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            with patch("os.getenv", return_value="env_password"):
                config = MT5Config()
                
                assert config.LOGIN == "12345"
                assert config.PASSWORD == "env_password"  # Should use env var
                assert config.SERVER == "test_server"
                assert config.DEMO_MODE is True
                assert config.SYMBOL == "XAUUSD.PRO"
                assert config.TIMEFRAME_MIN == 15
                assert config.BARS == 500
                assert config.MAGIC == 20250721
                assert config.SESSIONS == ["London", "NY"]
                assert config.MAX_SPREAD_POINTS == 200
                assert config.USE_RISK_BASED_VOLUME is True
                assert config.FIXED_VOLUME == 0.01
                assert config.SLEEP_SECONDS == 12
                assert config.RETRY_DELAY == 5
                assert config.CONSECUTIVE_SIGNALS_REQUIRED == 1
                assert config.LSTM_TIMESTEPS == 50
                assert config.COOLDOWN_SECONDS == 180
                assert config.BASE_RISK == 0.01
                assert config.MIN_LOT == 0.01
                assert config.MAX_LOT == 2.0
                assert config.MAX_OPEN_TRADES == 3
                assert config.MAX_DAILY_LOSS == 0.02
                assert config.MAX_TRADES_PER_DAY == 10
                assert config.LOG_ENABLED is True
                assert config.LOG_LEVEL == "INFO"
                assert config.LOG_FILE == "logs/trading_bot.log"
                assert config.TRADE_LOG_PATH == "data/trade_log_gold.csv"
                assert config.MODELS == {}
                assert config.SESSION_TZ == "Etc/UTC"
    
    def test_config_initialization_failure(self):
        """Test configuration initialization failure."""
        with patch("builtins.open", side_effect=FileNotFoundError("Config file not found")):
            with pytest.raises(Exception):
                MT5Config()
    
    def test_demo_mode_credentials_bypass(self):
        """Test that demo mode bypasses credential requirements."""
        config_data = {
            "credentials": {},
            "flags": {"demo_mode": True},
            "trading": {"symbol": "XAUUSD.PRO", "timeframe": 15, "bars": 500, "magic_number": 20250721, "sessions": ["London"], "max_spread_points": 200, "use_risk_based_volume": True, "fixed_volume": 0.01, "sleep_seconds": 12, "retry_delay": 5, "consecutive_signals_required": 1, "lstm_timesteps": 50, "cooldown_seconds": 180},
            "risk": {"base_risk": 0.01, "min_lot": 0.01, "max_lot": 2.0, "max_open_trades": 3, "max_daily_loss": 0.02, "max_trades_per_day": 10},
            "logging": {"enabled": True, "level": "INFO", "log_file": "logs/trading_bot.log", "trade_log_path": "data/trade_log_gold.csv"},
            "models": {},
            "session": {"timezone": "Etc/UTC"}
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            with patch("os.getenv", return_value=None):
                # Should not raise exception in demo mode
                config = MT5Config()
                assert config.DEMO_MODE is True
    
    def test_live_mode_credentials_required(self):
        """Test that live mode requires credentials."""
        config_data = {
            "credentials": {},
            "flags": {"demo_mode": False},
            "trading": {"symbol": "XAUUSD.PRO", "timeframe": 15, "bars": 500, "magic_number": 20250721, "sessions": ["London"], "max_spread_points": 200, "use_risk_based_volume": True, "fixed_volume": 0.01, "sleep_seconds": 12, "retry_delay": 5, "consecutive_signals_required": 1, "lstm_timesteps": 50, "cooldown_seconds": 180},
            "risk": {"base_risk": 0.01, "min_lot": 0.01, "max_lot": 2.0, "max_open_trades": 3, "max_daily_loss": 0.02, "max_trades_per_day": 10},
            "logging": {"enabled": True, "level": "INFO", "log_file": "logs/trading_bot.log", "trade_log_path": "data/trade_log_gold.csv"},
            "models": {},
            "session": {"timezone": "Etc/UTC"}
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            with patch("os.getenv", return_value=None):
                with pytest.raises(RuntimeError, match="MT5 credentials missing"):
                    MT5Config()
    
    def test_get_method(self):
        """Test get method for configuration values."""
        config_data = {
            "credentials": {"login": "12345"},
            "flags": {"demo_mode": True},
            "trading": {"symbol": "XAUUSD.PRO", "timeframe": 15, "bars": 500, "magic_number": 20250721, "sessions": ["London"], "max_spread_points": 200, "use_risk_based_volume": True, "fixed_volume": 0.01, "sleep_seconds": 12, "retry_delay": 5, "consecutive_signals_required": 1, "lstm_timesteps": 50, "cooldown_seconds": 180},
            "risk": {"base_risk": 0.01, "min_lot": 0.01, "max_lot": 2.0, "max_open_trades": 3, "max_daily_loss": 0.02, "max_trades_per_day": 10},
            "logging": {"enabled": True, "level": "INFO", "log_file": "logs/trading_bot.log", "trade_log_path": "data/trade_log_gold.csv"},
            "models": {},
            "session": {"timezone": "Etc/UTC"}
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            with patch("os.getenv", return_value="env_password"):
                config = MT5Config()
                
                assert config.get("credentials") == {"login": "12345"}
                assert config.get("nonexistent", "default") == "default"
    
    def test_get_nested_method(self):
        """Test get_nested method for nested configuration values."""
        config_data = {
            "credentials": {"login": "12345"},
            "flags": {"demo_mode": True},
            "trading": {"symbol": "XAUUSD.PRO", "timeframe": 15, "bars": 500, "magic_number": 20250721, "sessions": ["London"], "max_spread_points": 200, "use_risk_based_volume": True, "fixed_volume": 0.01, "sleep_seconds": 12, "retry_delay": 5, "consecutive_signals_required": 1, "lstm_timesteps": 50, "cooldown_seconds": 180},
            "risk": {"base_risk": 0.01, "min_lot": 0.01, "max_lot": 2.0, "max_open_trades": 3, "max_daily_loss": 0.02, "max_trades_per_day": 10},
            "logging": {"enabled": True, "level": "INFO", "log_file": "logs/trading_bot.log", "trade_log_path": "data/trade_log_gold.csv"},
            "models": {},
            "session": {"timezone": "Etc/UTC"}
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            with patch("os.getenv", return_value="env_password"):
                config = MT5Config()
                
                assert config.get_nested("credentials", "login") == "12345"
                assert config.get_nested("trading", "symbol") == "XAUUSD.PRO"
                assert config.get_nested("nonexistent", "key") is None
                assert config.get_nested("nonexistent", "key", "default") == "default"
