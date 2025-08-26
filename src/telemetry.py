#!/usr/bin/env python3
"""
MR BEN Telemetry Module
Event logging and MFE (Maximum Favorable Excursion) logging
"""

import json
import logging
import os
from datetime import datetime
from typing import Any


class EventLogger:
    """Event logging system for trading operations"""

    def __init__(self, path: str, run_id: str, symbol: str):
        self.path = path
        self.run_id = run_id
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Log startup event
        self.emit("logger_started", run_id=run_id, symbol=symbol)

    def emit(self, event_type: str, **kwargs):
        """Emit an event with timestamp and metadata"""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "run_id": self.run_id,
                "symbol": self.symbol,
                **kwargs,
            }

            # Write to JSONL file
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')

            self.logger.debug(f"Event logged: {event_type}")

        except Exception as e:
            self.logger.error(f"Failed to log event {event_type}: {e}")

    def log_trade_attempt(
        self,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        volume: float,
        confidence: float,
        source: str,
    ):
        """Log trade attempt"""
        self.emit(
            "trade_attempt",
            side=side,
            entry=entry,
            sl=sl,
            tp=tp,
            volume=volume,
            confidence=confidence,
            source=source,
        )

    def log_trade_execution(
        self,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        volume: float,
        order_id: int,
        execution_time: float,
    ):
        """Log successful trade execution"""
        self.emit(
            "trade_execution",
            side=side,
            entry=entry,
            sl=sl,
            tp=tp,
            volume=volume,
            order_id=order_id,
            execution_time=execution_time,
        )

    def log_trade_rejection(self, side: str, reason: str, confidence: float):
        """Log trade rejection"""
        self.emit("trade_rejection", side=side, reason=reason, confidence=confidence)

    def log_risk_event(self, event_type: str, details: dict[str, Any]):
        """Log risk management events"""
        self.emit("risk_event", risk_type=event_type, details=details)

    def log_system_event(self, event_type: str, details: dict[str, Any]):
        """Log system-level events"""
        self.emit("system_event", system_type=event_type, details=details)

    def close(self):
        """Close the event logger"""
        try:
            self.emit("logger_shutdown", timestamp=datetime.now().isoformat())
            self.logger.info("Event logger closed")
        except Exception as e:
            self.logger.warning(f"Error during logger shutdown: {e}")


class MFELogger:
    """Maximum Favorable Excursion logger for position analysis"""

    def __init__(self, path: str):
        self.path = path
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Initialize MFE tracking
        self.mfe_data = {}

    def log_tick(
        self,
        run_id: str,
        ticket: int,
        current_price: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
    ):
        """Log tick data for MFE analysis"""
        try:
            # Calculate current MFE
            if ticket not in self.mfe_data:
                self.mfe_data[ticket] = {
                    "run_id": run_id,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "max_favorable": 0.0,
                    "max_adverse": 0.0,
                    "ticks": [],
                }

            # Calculate price movement from entry
            price_move = current_price - entry_price

            # Update MFE tracking
            if price_move > self.mfe_data[ticket]["max_favorable"]:
                self.mfe_data[ticket]["max_favorable"] = price_move
            if price_move < self.mfe_data[ticket]["max_adverse"]:
                self.mfe_data[ticket]["max_adverse"] = price_move

            # Log tick data
            tick_data = {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "ticket": ticket,
                "current_price": current_price,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "price_move": price_move,
                "max_favorable": self.mfe_data[ticket]["max_favorable"],
                "max_adverse": self.mfe_data[ticket]["max_adverse"],
            }

            # Write to JSONL file
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(tick_data) + '\n')

        except Exception as e:
            self.logger.error(f"Failed to log MFE tick for ticket {ticket}: {e}")

    def get_mfe_summary(self, ticket: int) -> dict[str, Any] | None:
        """Get MFE summary for a specific ticket"""
        return self.mfe_data.get(ticket)

    def get_all_mfe_summaries(self) -> dict[int, dict[str, Any]]:
        """Get MFE summaries for all tracked tickets"""
        return self.mfe_data.copy()

    def clear_ticket(self, ticket: int):
        """Clear MFE data for a specific ticket (e.g., when position closes)"""
        if ticket in self.mfe_data:
            del self.mfe_data[ticket]
            self.logger.debug(f"Cleared MFE data for ticket {ticket}")

    def close(self):
        """Close the MFE logger"""
        try:
            # Log final MFE summaries
            for ticket, data in self.mfe_data.items():
                final_data = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "mfe_final_summary",
                    "ticket": ticket,
                    **data,
                }

                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(final_data) + '\n')

            self.logger.info("MFE logger closed")
        except Exception as e:
            self.logger.warning(f"Error during MFE logger shutdown: {e}")


class PerformanceMetrics:
    """Performance monitoring and metrics collection"""

    def __init__(self):
        self.start_time = datetime.now()
        self.cycle_count = 0
        self.trade_count = 0
        self.error_count = 0
        self.response_times = []
        self.memory_usage = []

        self.logger = logging.getLogger(self.__class__.__name__)

    def record_cycle(self, response_time: float):
        """Record a trading cycle"""
        self.cycle_count += 1
        self.response_times.append(response_time)

        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times.pop(0)

    def record_trade(self):
        """Record a trade execution"""
        self.trade_count += 1

    def record_error(self):
        """Record an error occurrence"""
        self.error_count += 1

    def record_memory_usage(self, memory_mb: float):
        """Record memory usage"""
        self.memory_usage.append({"timestamp": datetime.now().isoformat(), "memory_mb": memory_mb})

        # Keep only last 100 memory readings
        if len(self.memory_usage) > 100:
            self.memory_usage.pop(0)

    def get_stats(self) -> dict[str, Any]:
        """Get current performance statistics"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()

            # Calculate response time statistics
            avg_response = 0.0
            if self.response_times:
                avg_response = sum(self.response_times) / len(self.response_times)

            # Calculate memory statistics
            current_memory = 0.0
            if self.memory_usage:
                current_memory = self.memory_usage[-1]["memory_mb"]

            return {
                "uptime_seconds": uptime,
                "cycle_count": self.cycle_count,
                "cycles_per_second": self.cycle_count / uptime if uptime > 0 else 0,
                "avg_response_time": avg_response,
                "total_trades": self.trade_count,
                "error_rate": self.error_count / max(self.cycle_count, 1),
                "memory_mb": current_memory,
                "start_time": self.start_time.isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error calculating performance stats: {e}")
            return {
                "uptime_seconds": 0,
                "cycle_count": 0,
                "cycles_per_second": 0,
                "avg_response_time": 0,
                "total_trades": 0,
                "error_rate": 0,
                "memory_mb": 0,
                "start_time": self.start_time.isoformat(),
            }

    def reset(self):
        """Reset all metrics"""
        self.start_time = datetime.now()
        self.cycle_count = 0
        self.trade_count = 0
        self.error_count = 0
        self.response_times.clear()
        self.memory_usage.clear()
        self.logger.info("Performance metrics reset")


class MemoryMonitor:
    """Memory usage monitoring and cleanup"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_check = datetime.now()
        self.check_interval = 300  # 5 minutes

    def check_memory(self, force: bool = False) -> float | None:
        """Check current memory usage"""
        try:
            now = datetime.now()
            if not force and (now - self.last_check).total_seconds() < self.check_interval:
                return None

            import gc

            import psutil

            # Force garbage collection
            collected = gc.collect()

            # Get process memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            # Log memory usage
            self.logger.info(f"💾 Memory check: {memory_mb:.1f} MB (collected {collected} objects)")

            # Log system memory if available
            try:
                system_memory = psutil.virtual_memory()
                system_used_mb = system_memory.used / 1024 / 1024
                system_total_mb = system_memory.total / 1024 / 1024
                system_percent = system_memory.percent

                self.logger.info(
                    f"💻 System Memory: {system_used_mb:.1f}/{system_total_mb:.1f} MB ({system_percent:.1f}%)"
                )
            except Exception:
                pass

            self.last_check = now
            return memory_mb

        except ImportError:
            self.logger.warning("psutil not available - memory monitoring disabled")
            return None
        except Exception as e:
            self.logger.warning(f"Memory monitoring failed: {e}")
            return None

    def cleanup_memory(self):
        """Force memory cleanup"""
        try:
            import gc

            # Force garbage collection
            collected = gc.collect()

            self.logger.info(f"🧹 Memory cleanup: collected {collected} objects")
            return collected

        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
            return 0

    def should_cleanup(self, memory_mb: float, threshold_mb: float = 1000) -> bool:
        """Determine if memory cleanup is needed"""
        return memory_mb > threshold_mb
