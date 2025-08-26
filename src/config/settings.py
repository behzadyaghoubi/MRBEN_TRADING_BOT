"""
Configuration management for MR BEN Trading System.
Handles loading and validation of configuration from JSON files.
"""

import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


class MT5Config:
    """Reads config.json and exposes fields used by the system."""
    
    def __init__(self):
        config_path = 'config.json'
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            
            # Validate config with Pydantic schema if available
            try:
                from utils.config_schema import RootConfig
                validated = RootConfig(**raw)
                self.config_data = validated.model_dump()
            except ImportError:
                try:
                    # Try src path
                    from src.utils.config_schema import RootConfig
                    validated = RootConfig(**raw)
                    self.config_data = validated.model_dump()
                except ImportError:
                    # Fallback to direct assignment if schema not available
                    self.config_data = raw
            
        except Exception as e:
            print(f"❌ Config validation error: {e}")
            raise

        # Credentials
        creds = self.config_data.get("credentials", {})
        self.LOGIN = creds.get("login")
        self.PASSWORD = os.getenv("MT5_PASSWORD", creds.get("password"))
        self.SERVER = creds.get("server")

        # Flags
        flags = self.config_data.get("flags", {})
        self.DEMO_MODE = bool(flags.get("demo_mode", True))

        # Trading
        trading = self.config_data.get("trading", {})
        self.SYMBOL = trading.get("symbol", "XAUUSD.PRO")
        self.TIMEFRAME_MIN = int(trading.get("timeframe", 15))
        self.BARS = int(trading.get("bars", 500))
        self.MAGIC = int(trading.get("magic_number", 20250721))
        self.SESSIONS: List[str] = trading.get("sessions", ["London", "NY"])
        self.MAX_SPREAD_POINTS = int(trading.get("max_spread_points", 200))
        self.USE_RISK_BASED_VOLUME = bool(trading.get("use_risk_based_volume", True))
        self.FIXED_VOLUME = float(trading.get("fixed_volume", 0.01))
        self.SLEEP_SECONDS = int(trading.get("sleep_seconds", 12))
        self.RETRY_DELAY = int(trading.get("retry_delay", 5))
        self.CONSECUTIVE_SIGNALS_REQUIRED = int(trading.get("consecutive_signals_required", 1))
        self.LSTM_TIMESTEPS = int(trading.get("lstm_timesteps", 50))
        self.COOLDOWN_SECONDS = int(trading.get("cooldown_seconds", 180))

        # Risk
        risk = self.config_data.get("risk", {})
        self.BASE_RISK = float(risk.get("base_risk", 0.01))
        self.MIN_LOT = float(risk.get("min_lot", 0.01))
        self.MAX_LOT = float(risk.get("max_lot", 2.0))
        self.MAX_OPEN_TRADES = int(risk.get("max_open_trades", 3))
        self.MAX_DAILY_LOSS = float(risk.get("max_daily_loss", 0.02))
        self.MAX_TRADES_PER_DAY = int(risk.get("max_trades_per_day", 10))

        # Logging
        logging_cfg = self.config_data.get("logging", {})
        self.LOG_ENABLED = bool(logging_cfg.get("enabled", True))
        self.LOG_LEVEL = logging_cfg.get("level", "INFO")
        self.LOG_FILE = logging_cfg.get("log_file", "logs/trading_bot.log")
        self.TRADE_LOG_PATH = logging_cfg.get("trade_log_path", "data/trade_log_gold.csv")

        # Models (optional flags)
        self.MODELS = self.config_data.get("models", {})

        # Session TZ config
        session_cfg = self.config_data.get("session", {})
        self.SESSION_TZ = session_cfg.get("timezone", "Etc/UTC")

        # Strict credential check only if DEMO_MODE is False
        if not self.DEMO_MODE and not (self.LOGIN and self.PASSWORD and self.SERVER):
            raise RuntimeError("❌ MT5 credentials missing. Provide via config.json under 'credentials'.")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        return self.config_data.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value."""
        value = self.config_data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value 