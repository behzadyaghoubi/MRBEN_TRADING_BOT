from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


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


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


settings: AppSettings = get_settings()
