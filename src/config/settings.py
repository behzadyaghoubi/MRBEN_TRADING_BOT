from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass
class AppSettings:
    # Logging
    LOG_LEVEL: str = "INFO"

    # MT5 / broker (placeholders; can be None for tests)
    MT5_LOGIN: int | None = None
    MT5_PASSWORD: str | None = None
    MT5_SERVER: str | None = None

    # Paths / storage
    ENABLE_CSV_LOG: bool = False
    ENABLE_SQLITE_LOG: bool = False
    SQLITE_PATH: str = "mrben.db"


@dataclass
class MT5Config:
    # Trading parameters
    SYMBOL: str = "XAUUSD.PRO"
    TIMEFRAME_MIN: int = 15
    BARS: int = 500
    MAGIC: int = 20250721
    DEMO_MODE: bool = True

    # Credentials
    LOGIN: int = 12345
    PASSWORD: str = "test_password"
    SERVER: str = "test_server"

    # Trading limits
    MAX_SPREAD_POINTS: int = 200
    USE_RISK_BASED_VOLUME: bool = True
    FIXED_VOLUME: float = 0.01
    SLEEP_SECONDS: int = 12
    RETRY_DELAY: int = 5
    CONSECUTIVE_SIGNALS_REQUIRED: int = 1
    LSTM_TIMESTEPS: int = 50
    COOLDOWN_SECONDS: int = 180

    # Risk management
    BASE_RISK: float = 0.01
    MIN_LOT: float = 0.01
    MAX_LOT: float = 2.0
    MAX_OPEN_TRADES: int = 3
    MAX_DAILY_LOSS: float = 0.02
    MAX_TRADES_PER_DAY: int = 10

    # Sessions
    SESSIONS: list[str] = None
    SESSION_TZ: str = "Etc/UTC"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/trading_bot.log"
    TRADE_LOG_PATH: str = "data/trade_log_gold.csv"

    # Configuration data
    config_data: dict[str, Any] = None
    config_file: str = "config/pro_config.json"

    def __post_init__(self):
        if self.SESSIONS is None:
            self.SESSIONS = ["London", "NY"]
        if self.config_data is None:
            self.config_data = {
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

    def get_config_summary(self) -> dict[str, Any]:
        return {
            "symbol": self.SYMBOL,
            "bars": self.BARS,
            "magic": self.MAGIC,
            "demo_mode": self.DEMO_MODE,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return getattr(self, key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value."""
        value = self.config_data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


settings: AppSettings = get_settings()
