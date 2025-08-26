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
    # Minimal test-friendly config; adjust fields if tests require more.
    SYMBOL: str = "XAUUSD.PRO"
    BARS: int = 500
    MAGIC: int = 20250721
    DEMO_MODE: bool = True

    def get_config_summary(self) -> dict[str, Any]:
        return {
            "symbol": self.SYMBOL,
            "bars": self.BARS,
            "magic": self.MAGIC,
            "demo_mode": self.DEMO_MODE,
        }


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


settings: AppSettings = get_settings()
