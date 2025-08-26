#!/usr/bin/env python3
"""
MR BEN - Emergency Stop System
File-based kill switch for immediate trading cessation
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .loggingx import logger


@dataclass
class EmergencyState:
    """Emergency stop state information"""

    is_active: bool
    triggered_at: datetime | None
    trigger_reason: str
    trigger_source: str  # 'file', 'api', 'manual'
    last_check: datetime
    checks_count: int


class EmergencyStop:
    """Emergency stop system with file-based kill switch"""

    def __init__(
        self,
        halt_file_path: str = "halt.flag",
        check_interval: float = 1.0,
        auto_recovery: bool = False,
        recovery_delay: float = 300.0,
    ):  # 5 minutes
        self.halt_file_path = Path(halt_file_path)
        self.check_interval = check_interval
        self.auto_recovery = auto_recovery
        self.recovery_delay = recovery_delay

        # State tracking
        self.state = EmergencyState(
            is_active=False,
            triggered_at=None,
            trigger_reason="",
            trigger_source="",
            last_check=datetime.now(UTC),
            checks_count=0,
        )

        # Callbacks
        self.on_emergency_callbacks: list[Callable] = []
        self.on_recovery_callbacks: list[Callable] = []

        # Monitoring thread
        self._monitor_thread: threading.Thread | None = None
        self._stop_monitoring = threading.Event()

        # Initialize
        self._check_emergency_state()

        logger.bind(evt="EMERGENCY").info(
            "emergency_stop_initialized",
            halt_file=str(self.halt_file_path),
            check_interval=self.check_interval,
            auto_recovery=self.auto_recovery,
        )

    def start_monitoring(self) -> None:
        """Start continuous monitoring for emergency stop"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.bind(evt="EMERGENCY").warning("monitoring_already_active")
            return

        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="EmergencyStopMonitor", daemon=True
        )
        self._monitor_thread.start()

        logger.bind(evt="EMERGENCY").info("emergency_monitoring_started")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_monitoring.set()
            self._monitor_thread.join(timeout=2.0)

            logger.bind(evt="EMERGENCY").info("emergency_monitoring_stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                self._check_emergency_state()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.bind(evt="EMERGENCY").error("monitoring_loop_error", error=str(e))
                time.sleep(self.check_interval)

    def _check_emergency_state(self) -> None:
        """Check current emergency state"""
        try:
            previous_state = self.state.is_active
            current_time = datetime.now(UTC)

            # Check if halt file exists
            halt_file_exists = self.halt_file_path.exists()

            if halt_file_exists and not self.state.is_active:
                # Emergency stop triggered
                self._trigger_emergency_stop("file", current_time)

            elif not halt_file_exists and self.state.is_active:
                # Emergency stop cleared
                if self.auto_recovery:
                    self._trigger_recovery("file", current_time)
                else:
                    logger.bind(evt="EMERGENCY").info("emergency_cleared_manual_recovery_required")

            # Update state
            self.state.last_check = current_time
            self.state.checks_count += 1

            # Log state changes
            if previous_state != self.state.is_active:
                if self.state.is_active:
                    logger.bind(evt="EMERGENCY").warning(
                        "emergency_stop_activated",
                        reason=self.state.trigger_reason,
                        source=self.state.trigger_source,
                    )
                else:
                    logger.bind(evt="EMERGENCY").info(
                        "emergency_stop_deactivated", recovery_source=self.state.trigger_source
                    )

        except Exception as e:
            logger.bind(evt="EMERGENCY").error("emergency_state_check_error", error=str(e))

    def _trigger_emergency_stop(self, source: str, trigger_time: datetime) -> None:
        """Trigger emergency stop"""
        self.state.is_active = True
        self.state.triggered_at = trigger_time
        self.state.trigger_reason = "Emergency stop activated"
        self.state.trigger_source = source

        # Execute callbacks
        self._execute_emergency_callbacks()

        logger.bind(evt="EMERGENCY").critical(
            "emergency_stop_triggered", source=source, timestamp=trigger_time.isoformat()
        )

    def _trigger_recovery(self, source: str, recovery_time: datetime) -> None:
        """Trigger recovery from emergency stop"""
        if not self.state.is_active:
            return

        # Check if enough time has passed for auto-recovery
        if (
            self.auto_recovery
            and self.state.triggered_at
            and (recovery_time - self.state.triggered_at).total_seconds() < self.recovery_delay
        ):
            logger.bind(evt="EMERGENCY").info(
                "recovery_delayed",
                remaining_seconds=self.recovery_delay
                - (recovery_time - self.state.triggered_at).total_seconds(),
            )
            return

        self.state.is_active = False
        self.state.triggered_at = None
        self.state.trigger_reason = ""
        self.state.trigger_source = ""

        # Execute recovery callbacks
        self._execute_recovery_callbacks()

        logger.bind(evt="EMERGENCY").info(
            "emergency_recovery_triggered", source=source, timestamp=recovery_time.isoformat()
        )

    def _execute_emergency_callbacks(self) -> None:
        """Execute all emergency stop callbacks"""
        for callback in self.on_emergency_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.bind(evt="EMERGENCY").error(
                    "emergency_callback_error", callback=str(callback), error=str(e)
                )

    def _execute_recovery_callbacks(self) -> None:
        """Execute all recovery callbacks"""
        for callback in self.on_recovery_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.bind(evt="EMERGENCY").error(
                    "recovery_callback_error", callback=str(callback), error=str(e)
                )

    def manual_emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Manually trigger emergency stop"""
        if self.state.is_active:
            logger.bind(evt="EMERGENCY").warning("emergency_already_active")
            return

        # Create halt file
        try:
            with open(self.halt_file_path, 'w') as f:
                f.write(f"Emergency Stop: {reason}\n")
                f.write(f"Triggered at: {datetime.now(UTC).isoformat()}\n")
                f.write("Source: manual\n")

            self._trigger_emergency_stop("manual", datetime.now(UTC))

        except Exception as e:
            logger.bind(evt="EMERGENCY").error("manual_emergency_stop_failed", error=str(e))

    def manual_recovery(self) -> None:
        """Manually trigger recovery"""
        if not self.state.is_active:
            logger.bind(evt="EMERGENCY").warning("emergency_not_active")
            return

        # Remove halt file
        try:
            if self.halt_file_path.exists():
                self.halt_file_path.unlink()

            self._trigger_recovery("manual", datetime.now(UTC))

        except Exception as e:
            logger.bind(evt="EMERGENCY").error("manual_recovery_failed", error=str(e))

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        return not self.state.is_active

    def get_state(self) -> EmergencyState:
        """Get current emergency state"""
        return self.state

    def add_emergency_callback(self, callback: Callable[[EmergencyState], None]) -> None:
        """Add callback for emergency stop events"""
        self.on_emergency_callbacks.append(callback)
        logger.bind(evt="EMERGENCY").debug("emergency_callback_added")

    def add_recovery_callback(self, callback: Callable[[EmergencyState], None]) -> None:
        """Add callback for recovery events"""
        self.on_recovery_callbacks.append(callback)
        logger.bind(evt="EMERGENCY").debug("recovery_callback_added")

    def get_halt_file_info(self) -> dict | None:
        """Get information about the halt file if it exists"""
        if not self.halt_file_path.exists():
            return None

        try:
            stat = self.halt_file_path.stat()
            with open(self.halt_file_path) as f:
                content = f.read()

            return {
                "path": str(self.halt_file_path),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime, tz=UTC),
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                "content": content,
            }
        except Exception as e:
            logger.bind(evt="EMERGENCY").error("halt_file_info_error", error=str(e))
            return None

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self.stop_monitoring()

            # Remove halt file if it exists
            if self.halt_file_path.exists():
                self.halt_file_path.unlink()
                logger.bind(evt="EMERGENCY").info("halt_file_cleaned_up")

            logger.bind(evt="EMERGENCY").info("emergency_stop_cleanup_complete")

        except Exception as e:
            logger.bind(evt="EMERGENCY").error("cleanup_error", error=str(e))
