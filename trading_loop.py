#!/usr/bin/env python3
"""
MR BEN Trading Loop Manager Module
"""

import logging
import time
from collections.abc import Callable
from threading import Event, Thread
from typing import Any

logger = logging.getLogger(__name__)


class TradingLoopManager:
    """Manages the main trading loop execution"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.is_running = False
        self.stop_event = Event()
        self.loop_thread = None
        self.logger = logging.getLogger(__name__)
        self.cycle_count = 0
        self.last_cycle_time = 0

    def start_loop(self, loop_function: Callable, *args, **kwargs) -> bool:
        """Start the trading loop in a separate thread"""
        if self.is_running:
            self.logger.warning("Trading loop already running")
            return False

        try:
            self.logger.info("Starting trading loop...")
            self.is_running = True
            self.stop_event.clear()

            # Start loop in separate thread
            self.loop_thread = Thread(
                target=self._run_loop, args=(loop_function, args, kwargs), daemon=True
            )
            self.loop_thread.start()

            self.logger.info("✅ Trading loop started successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start trading loop: {e}")
            self.is_running = False
            return False

    def _run_loop(self, loop_function: Callable, args: tuple, kwargs: dict):
        """Internal loop execution method"""
        try:
            while not self.stop_event.is_set():
                start_time = time.time()

                # Execute the loop function
                try:
                    loop_function(*args, **kwargs)
                    self.cycle_count += 1
                except Exception as e:
                    self.logger.error(f"Error in trading loop cycle: {e}")

                # Calculate cycle time
                cycle_time = time.time() - start_time
                self.last_cycle_time = cycle_time

                # Sleep if cycle was too fast
                min_cycle_time = self.config.get('min_cycle_time', 0.1)
                if cycle_time < min_cycle_time:
                    time.sleep(min_cycle_time - cycle_time)

        except Exception as e:
            self.logger.error(f"Fatal error in trading loop: {e}")
        finally:
            self.is_running = False
            self.logger.info("Trading loop stopped")

    def stop_loop(self) -> bool:
        """Stop the trading loop"""
        if not self.is_running:
            self.logger.warning("Trading loop not running")
            return False

        try:
            self.logger.info("Stopping trading loop...")
            self.stop_event.set()

            # Wait for thread to finish
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=5.0)

            self.is_running = False
            self.logger.info("✅ Trading loop stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to stop trading loop: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get trading loop status"""
        return {
            "running": self.is_running,
            "cycle_count": self.cycle_count,
            "last_cycle_time": self.last_cycle_time,
            "thread_alive": self.loop_thread.is_alive() if self.loop_thread else False,
        }

    def is_alive(self) -> bool:
        """Check if the trading loop is alive"""
        return self.is_running and (self.loop_thread and self.loop_thread.is_alive())
