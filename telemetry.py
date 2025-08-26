#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Telemetry Module
"""

import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EventLogger:
    """Event logging for trading activities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def log_trade_attempt(self, symbol: str, signal: int, confidence: float, **kwargs):
        """Log trade attempt"""
        self.logger.info(f"Trade attempt: {symbol} signal={signal} conf={confidence:.3f}")
        
    def log_trade_result(self, symbol: str, success: bool, **kwargs):
        """Log trade result"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Trade result: {symbol} {status}")
        
    def close(self):
        """Close the event logger"""
        pass

class MFELogger:
    """Multi-Factor Event Logger"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Performance metrics tracking"""
    
    def __init__(self):
        self.start_time = time.time()
        self.cycle_count = 0
        self.trade_count = 0
        self.error_count = 0
        self.memory_usage = []
        
    def record_cycle(self, duration: float):
        """Record a trading cycle"""
        self.cycle_count += 1
        
    def record_trade(self, success: bool):
        """Record a trade execution"""
        self.trade_count += 1
        
    def record_error(self, error: str):
        """Record an error"""
        self.error_count += 1
        
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage"""
        self.memory_usage.append(memory_mb)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        uptime = time.time() - self.start_time
        cycles_per_second = self.cycle_count / uptime if uptime > 0 else 0
        
        return {
            "uptime_seconds": uptime,
            "cycle_count": self.cycle_count,
            "cycles_per_second": cycles_per_second,
            "trade_count": self.trade_count,
            "error_count": self.error_count,
            "avg_memory_mb": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        }

class MemoryMonitor:
    """Memory usage monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
