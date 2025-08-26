from __future__ import annotations

import sys

from loguru import logger

_DEF_FMT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<7}</level> | {message} | {extra}"


def setup_logging(level: str = "INFO"):
    """Setup structured logging with loguru"""
    logger.remove()
    logger.add(sys.stdout, level=level.upper(), format=_DEF_FMT, backtrace=False, diagnose=False)
    return logger


def log_cfg(cfg_json: str):
    """Log configuration with structured event"""
    logger.bind(evt="CONFIG").info("config_loaded", cfg=cfg_json)
