#!/usr/bin/env python3
"""
MR BEN Trading Loop Manager
Manages the main trading loop and decision logic
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .telemetry import MemoryMonitor, PerformanceMetrics
from .trading_system import TradingSystem


class TradingLoopManager:
    """Manages the main trading loop and decision logic"""

    def __init__(self, trading_system: TradingSystem):
        self.trading_system = trading_system
        self.config = trading_system.config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.memory_monitor = MemoryMonitor()

        # Loop control
        self.running = False
        self.cycle = 0
        self.last_trade_ts = time.time()

        # State tracking
        self.last_bar_time = None
        self._last_inbar_eval = None

        # Logging intervals
        self.SESSION_LOG_EVERY = 20
        self.PERFORMANCE_LOG_EVERY = 100
        self.MEMORY_CHECK_INTERVAL = 300

    def start(self):
        """Start the trading loop in a separate thread"""
        if self.running:
            self.logger.warning("Trading loop already running")
            return

        self.running = True
        self.logger.info("🔄 Starting trading loop...")

        # Start loop in background thread
        self.loop_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.loop_thread.start()

        self.logger.info("✅ Trading loop started successfully")

    def stop(self):
        """Stop the trading loop"""
        self.logger.info("🛑 Stopping trading loop...")
        self.running = False

        # Wait for loop to finish
        if hasattr(self, 'loop_thread') and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=5.0)

        self.logger.info("✅ Trading loop stopped")

    def _trading_loop(self):
        """Main trading loop implementation"""
        self.logger.info("🔄 Trading loop started")

        while self.running:
            cycle_start = time.time()

            try:
                # Performance monitoring
                if self.cycle % self.PERFORMANCE_LOG_EVERY == 0:
                    self._log_performance_metrics()

                # Memory management
                memory_mb = self.memory_monitor.check_memory()
                if memory_mb is not None:
                    self.metrics.record_memory_usage(memory_mb)

                    # Force cleanup if memory usage is high
                    if self.memory_monitor.should_cleanup(memory_mb):
                        self.logger.warning(
                            f"🧹 High memory usage ({memory_mb:.1f} MB), forcing cleanup"
                        )
                        self.memory_monitor.cleanup_memory()

                # Check if trading is allowed
                can_trade, reason = self.trading_system.can_trade()
                if not can_trade:
                    self.logger.info(f"⏸️ Skip: {reason}")
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # Update trailing stops
                self.trading_system.update_trailing_stops()

                # Get market data
                df = self.trading_system.data_manager.get_latest_data(self.config.BARS)
                if df is None or len(df) < 50:
                    self.logger.warning("Insufficient data; retrying...")
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # Bar-gate logic
                if not self._check_bar_gate(df):
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # Generate trading signal
                signal_data = self.trading_system.generate_signal(df)

                # Validate signal
                is_valid, validation_reason = self.trading_system.validate_signal(signal_data)
                if not is_valid:
                    self.logger.info(f"⏸️ Skip: {validation_reason}")
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # Execute trade if conditions are met
                if self._should_execute_trade(signal_data):
                    self.logger.info("🚀 Executing trade...")
                    success = self.trading_system.execute_trade(signal_data, df)

                    if success:
                        self.last_trade_ts = time.time()
                        self.metrics.record_trade()
                        self.logger.info("✅ Trade executed successfully")
                    else:
                        self.logger.error("❌ Trade execution failed")
                else:
                    self.logger.info("⏸️ Trade execution skipped - conditions not met")

                # Cycle summary logging
                if self.cycle % 10 == 0:
                    self._log_cycle_summary(signal_data)

                # Record cycle performance
                cycle_time = time.time() - cycle_start
                self.metrics.record_cycle(cycle_time)

                self.cycle += 1
                time.sleep(self.config.SLEEP_SECONDS)

            except Exception as e:
                self.metrics.record_error()
                self.logger.error(f"Loop error: {e}")
                time.sleep(self.config.RETRY_DELAY)

    def _check_bar_gate(self, df: pd.DataFrame) -> bool:
        """Check if we should process this bar"""
        try:
            tf_min = self.config.TIMEFRAME_MIN
            bar_ts = pd.to_datetime(df['time'].iloc[-1]).to_pydatetime()

            # Floor to timeframe: 06:37 → 06:30 for TF=15
            bar_ts_floor = bar_ts.replace(
                minute=(bar_ts.minute // tf_min) * tf_min, second=0, microsecond=0
            )

            # Check for stale bars
            stale_lim = timedelta(minutes=1.5 * tf_min)
            if datetime.now() - bar_ts_floor > stale_lim:
                self.logger.warning("⚠️ Stale bars detected (no new candle)")
                return False

            # Check if this is a new bar
            if self.last_bar_time is None:
                self.last_bar_time = bar_ts_floor
                return True

            same_bar = bar_ts_floor == self.last_bar_time
            if not same_bar:
                self.logger.info(f"🆕 New bar: {self.last_bar_time:%H:%M} → {bar_ts_floor:%H:%M}")
                self.last_bar_time = bar_ts_floor

            return True

        except Exception as e:
            self.logger.error(f"Error in bar gate check: {e}")
            return False

    def _should_execute_trade(self, signal_data: dict[str, Any]) -> bool:
        """Determine if trade should be executed based on signal and conditions"""
        try:
            # Check signal validity
            if signal_data['signal'] == 0:
                return False

            # Check confidence threshold
            thr = self.trading_system.risk_manager.get_current_confidence_threshold()
            if signal_data['confidence'] < thr:
                return False

            # Check consecutive signals requirement
            req_consec = self.config.CONSECUTIVE_SIGNALS_REQUIRED
            if self.trading_system.state.consecutive_signals < req_consec:
                return False

            # Check open positions limit
            open_count = self._get_open_trades_count()
            if open_count >= self.config.MAX_OPEN_TRADES:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking trade execution conditions: {e}")
            return False

    def _get_open_trades_count(self) -> int:
        """Get count of open trades"""
        try:
            # This would typically call the risk manager or trade executor
            # Simplified for modularity
            return 0
        except Exception as e:
            self.logger.error(f"Error getting open trades count: {e}")
            return 0

    def _log_performance_metrics(self):
        """Log performance metrics and system health"""
        try:
            stats = self.metrics.get_stats()
            self.logger.info("📊 Performance Metrics:")
            self.logger.info(f"   Uptime: {stats['uptime_seconds']:.0f}s")
            self.logger.info(f"   Cycles/sec: {stats['cycles_per_second']:.2f}")
            self.logger.info(f"   Avg Response: {stats['avg_response_time']:.3f}s")
            self.logger.info(f"   Total Trades: {stats['total_trades']}")
            self.logger.info(f"   Error Rate: {stats['error_rate']:.3f}")
            self.logger.info(f"   Memory: {stats['memory_mb']:.1f} MB")
        except Exception as e:
            self.logger.warning(f"Failed to log performance metrics: {e}")

    def _log_cycle_summary(self, signal_data: dict[str, Any]):
        """Log summary for current cycle"""
        try:
            open_count = self._get_open_trades_count()

            self.logger.info(f"📊 Cycle {self.cycle} summary:")
            self.logger.info(
                f"   Last signal: {signal_data['signal']} (confidence: {signal_data['confidence']:.3f})"
            )
            self.logger.info(f"   Consecutive: {self.trading_system.state.consecutive_signals}")
            self.logger.info(f"   Open trades: {open_count}")

        except Exception as e:
            self.logger.warning(f"Failed to log cycle summary: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current trading loop status"""
        try:
            stats = self.metrics.get_stats()
            return {
                "running": self.running,
                "cycle": self.cycle,
                "uptime_seconds": stats['uptime_seconds'],
                "total_trades": stats['total_trades'],
                "error_rate": stats['error_rate'],
                "memory_mb": stats['memory_mb'],
                "last_trade_ts": self.last_trade_ts,
                "last_bar_time": self.last_bar_time.isoformat() if self.last_bar_time else None,
            }
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"error": str(e)}

    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics.reset()
        self.logger.info("Performance metrics reset")
