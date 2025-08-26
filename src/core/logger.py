"""
Logging configuration and utilities for MR BEN Trading Bot.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

try:
    from config.settings import settings
except Exception:

    class _Tmp:
        LOG_LEVEL = "INFO"

    settings = _Tmp()  # fail-safe for tests


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    level = getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler (optional, safe defaults)
    try:
        fh = RotatingFileHandler("logs/trading_bot.log", maxBytes=1_000_000, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass  # logs dir may not exist in tests; ignore

    logger.propagate = False
    return logger
