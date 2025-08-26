"""
Configuration loader for MR BEN Trading Bot.
Loads configuration from environment variables with validation and type conversion.
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and validates configuration from environment variables."""

    @staticmethod
    def get_str(key: str, default: str = "") -> str:
        """Get string value from environment variable."""
        return os.getenv(key, default)

    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """Get integer value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}, using default: {default}")
            return default

    @staticmethod
    def get_float(key: str, default: float = 0.0) -> float:
        """Get float value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float value for {key}: {value}, using default: {default}")
            return default

    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')

    @staticmethod
    def get_path(key: str, default: str = "") -> Path:
        """Get Path object from environment variable."""
        value = os.getenv(key, default)
        return Path(value) if value else Path()

    @staticmethod
    def load_trading_config() -> dict[str, Any]:
        """Load trading configuration from environment variables."""
        return {
            'symbol': ConfigLoader.get_str('TRADING_SYMBOL', 'XAUUSD'),
            'timeframe': ConfigLoader.get_str('TRADING_TIMEFRAME', 'M15'),
            'enabled': ConfigLoader.get_bool('TRADING_ENABLED', True),
            'base_risk': ConfigLoader.get_float('TRADING_BASE_RISK', 0.01),
            'min_lot': ConfigLoader.get_float('TRADING_MIN_LOT', 0.01),
            'max_lot': ConfigLoader.get_float('TRADING_MAX_LOT', 2.0),
            'max_open_trades': ConfigLoader.get_int('TRADING_MAX_OPEN_TRADES', 3),
            'dynamic_sensitivity': ConfigLoader.get_float('TRADING_DYNAMIC_SENSITIVITY', 0.5),
            'start_balance': ConfigLoader.get_float('TRADING_START_BALANCE', 100000),
            'stop_loss_pips': ConfigLoader.get_int('TRADING_STOP_LOSS_PIPS', 30),
            'take_profit_pips': ConfigLoader.get_int('TRADING_TAKE_PROFIT_PIPS', 60),
            'magic_number': ConfigLoader.get_int('TRADING_MAGIC_NUMBER', 20250627),
            'deviation': ConfigLoader.get_int('TRADING_DEVIATION', 10),
        }

    @staticmethod
    def load_mt5_config() -> dict[str, Any]:
        """Load MT5 configuration from environment variables."""
        return {
            'login': ConfigLoader.get_int('MT5_LOGIN', 0),
            'password': ConfigLoader.get_str('MT5_PASSWORD', ''),
            'server': ConfigLoader.get_str('MT5_SERVER', ''),
            'timeout': ConfigLoader.get_int('MT5_TIMEOUT', 60000),
            'enable_real_trading': ConfigLoader.get_bool('MT5_ENABLE_REAL_TRADING', False),
        }

    @staticmethod
    def load_ai_config() -> dict[str, Any]:
        """Load AI configuration from environment variables."""
        return {
            'model_path': ConfigLoader.get_str(
                'AI_MODEL_PATH', 'models/mrben_ai_signal_filter_xgb.joblib'
            ),
            'threshold': ConfigLoader.get_float('AI_THRESHOLD', 0.5),
            'fallback_enabled': ConfigLoader.get_bool('AI_FALLBACK_ENABLED', True),
        }

    @staticmethod
    def load_logging_config() -> dict[str, Any]:
        """Load logging configuration from environment variables."""
        return {
            'level': ConfigLoader.get_str('LOGGING_LEVEL', 'INFO'),
            'file_path': ConfigLoader.get_str('LOGGING_FILE_PATH', 'logs/trading_bot.log'),
            'max_file_size': ConfigLoader.get_int(
                'LOGGING_MAX_FILE_SIZE', 10 * 1024 * 1024
            ),  # 10MB
            'backup_count': ConfigLoader.get_int('LOGGING_BACKUP_COUNT', 5),
            'console_enabled': ConfigLoader.get_bool('LOGGING_CONSOLE_ENABLED', True),
            'file_enabled': ConfigLoader.get_bool('LOGGING_FILE_ENABLED', True),
        }

    @staticmethod
    def load_database_config() -> dict[str, Any]:
        """Load database configuration from environment variables."""
        return {
            'trades_db': ConfigLoader.get_str('DB_TRADES_PATH', 'data/trades.db'),
            'signals_db': ConfigLoader.get_str('DB_SIGNALS_PATH', 'data/signals.db'),
            'backup_enabled': ConfigLoader.get_bool('DB_BACKUP_ENABLED', True),
            'backup_interval_hours': ConfigLoader.get_int('DB_BACKUP_INTERVAL_HOURS', 24),
        }

    @staticmethod
    def validate_config() -> bool:
        """Validate critical configuration values."""
        errors = []

        # Validate MT5 credentials
        if not ConfigLoader.get_int('MT5_LOGIN', 0):
            errors.append("MT5_LOGIN is required")

        if not ConfigLoader.get_str('MT5_PASSWORD'):
            errors.append("MT5_PASSWORD is required")

        if not ConfigLoader.get_str('MT5_SERVER'):
            errors.append("MT5_SERVER is required")

        # Validate trading parameters
        if ConfigLoader.get_float('TRADING_BASE_RISK', 0.01) <= 0:
            errors.append("TRADING_BASE_RISK must be greater than 0")

        if ConfigLoader.get_float('TRADING_MIN_LOT', 0.01) <= 0:
            errors.append("TRADING_MIN_LOT must be greater than 0")

        if ConfigLoader.get_int('TRADING_MAX_OPEN_TRADES', 3) <= 0:
            errors.append("TRADING_MAX_OPEN_TRADES must be greater than 0")

        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False

        return True

    @staticmethod
    def get_all_config() -> dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            'trading': ConfigLoader.load_trading_config(),
            'mt5': ConfigLoader.load_mt5_config(),
            'ai': ConfigLoader.load_ai_config(),
            'logging': ConfigLoader.load_logging_config(),
            'database': ConfigLoader.load_database_config(),
        }
