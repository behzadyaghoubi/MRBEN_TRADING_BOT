"""
Performance monitoring for MR BEN Trading System.
"""

import logging
import time
from typing import Any

import numpy as np
import psutil


class PerformanceMetrics:
    """Performance monitoring and metrics collection."""

    def __init__(self):
        self.start_time = time.time()
        self.cycle_count = 0
        self.trade_count = 0
        self.error_count = 0
        self.memory_usage: list[float] = []
        self.response_times: list[float] = []

    def record_cycle(self, response_time: float) -> None:
        """Record a trading cycle with response time."""
        self.cycle_count += 1
        self.response_times.append(response_time)
        if len(self.response_times) > 1000:  # Keep last 1000
            self.response_times.pop(0)

    def record_trade(self) -> None:
        """Record a completed trade."""
        self.trade_count += 1

    def record_error(self) -> None:
        """Record an error occurrence."""
        self.error_count += 1

    def get_stats(self) -> dict[str, Any]:
        """Get current performance statistics."""
        uptime = time.time() - self.start_time
        avg_response = np.mean(self.response_times) if self.response_times else 0

        try:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            memory_mb = 0.0

        return {
            "uptime_seconds": uptime,
            "cycle_count": self.cycle_count,
            "cycles_per_second": self.cycle_count / uptime if uptime > 0 else 0,
            "avg_response_time": avg_response,
            "total_trades": self.trade_count,
            "error_rate": self.error_count / max(self.cycle_count, 1),
            "memory_mb": memory_mb,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.start_time = time.time()
        self.cycle_count = 0
        self.trade_count = 0
        self.error_count = 0
        self.memory_usage.clear()
        self.response_times.clear()

    def log_summary(self, logger: logging.Logger) -> None:
        """Log current performance summary."""
        stats = self.get_stats()
        logger.info("📊 Performance Summary:")
        logger.info(f"   Uptime: {stats['uptime_seconds']:.0f}s")
        logger.info(f"   Cycles/sec: {stats['cycles_per_second']:.2f}")
        logger.info(f"   Avg Response: {stats['avg_response_time']:.3f}s")
        logger.info(f"   Total Trades: {stats['total_trades']}")
        logger.info(f"   Error Rate: {stats['error_rate']:.3f}")
        logger.info(f"   Memory: {stats['memory_mb']:.1f} MB")
