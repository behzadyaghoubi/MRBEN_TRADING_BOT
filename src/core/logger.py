"""
Logging configuration and utilities for MR BEN Trading Bot.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import os

from ..config.settings import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class TradingLogger:
    """Centralized logging system for the trading bot."""
    
    def __init__(self, name: str = "mrben_trading_bot"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers."""
        # Create logs directory
        log_dir = Path(settings.logging.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.logging.level))
        
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            settings.logging.file_path,
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Error file handler
        error_log_path = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # Trade-specific handler
        trade_log_path = log_dir / "trades.log"
        trade_handler = logging.handlers.RotatingFileHandler(
            trade_log_path,
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count
        )
        trade_handler.setLevel(logging.INFO)
        trade_formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        trade_handler.setFormatter(trade_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(trade_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def trade(self, message: str):
        """Log trade-specific message."""
        self.logger.info(f"[TRADE] {message}")
    
    def signal(self, message: str):
        """Log signal-specific message."""
        self.logger.info(f"[SIGNAL] {message}")
    
    def ai(self, message: str):
        """Log AI-specific message."""
        self.logger.info(f"[AI] {message}")


# Global logger instance
logger = TradingLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f"mrben_trading_bot.{name}")


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self):
        self.logger = get_logger("performance")
        self.metrics = {}
    
    def log_trade_performance(self, trade_data: dict):
        """Log trade performance metrics."""
        self.logger.info(f"Trade Performance: {trade_data}")
    
    def log_signal_accuracy(self, signal_data: dict):
        """Log signal accuracy metrics."""
        self.logger.info(f"Signal Accuracy: {signal_data}")
    
    def log_ai_performance(self, ai_data: dict):
        """Log AI model performance metrics."""
        self.logger.info(f"AI Performance: {ai_data}")


# Global performance logger
performance_logger = PerformanceLogger() 