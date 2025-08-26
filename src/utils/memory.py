"""
Memory management utilities for MR BEN Trading System.
"""

import gc
import logging


def log_memory_usage(logger: logging.Logger, context: str = "Memory check") -> None:
    """
    Log current memory usage for monitoring.

    Args:
        logger: Logger instance
        context: Context description for the memory check
    """
    try:
        import psutil

        # Force garbage collection
        gc.collect()

        # Get process memory info
        process = psutil.Process()
        memory_info = process.memory_info()

        # Convert to MB
        memory_mb = memory_info.rss / 1024 / 1024

        logger.info(f"💾 {context}: {memory_mb:.1f} MB")

        # Log system memory if available
        try:
            system_memory = psutil.virtual_memory()
            system_used_mb = system_memory.used / 1024 / 1024
            system_total_mb = system_memory.total / 1024 / 1024
            system_percent = system_memory.percent

            logger.info(
                f"💻 System Memory: {system_used_mb:.1f}/{system_total_mb:.1f} MB ({system_percent:.1f}%)"
            )
        except Exception:
            pass

    except ImportError:
        logger.warning("psutil not available - memory monitoring disabled")
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {e}")


def cleanup_memory() -> None:
    """
    Clean up memory and force garbage collection.
    """
    try:
        # Force garbage collection
        collected = gc.collect()

        # Log cleanup results
        logging.getLogger("MemoryMgmt").info(f"🧹 Memory cleanup: collected {collected} objects")

    except Exception as e:
        logging.getLogger("MemoryMgmt").warning(f"Memory cleanup failed: {e}")


def get_memory_usage() -> float | None:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in MB or None if unavailable
    """
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024
    except Exception:
        return None


def check_memory_threshold(threshold_mb: float = 1000.0) -> bool:
    """
    Check if memory usage exceeds threshold.

    Args:
        threshold_mb: Memory threshold in MB

    Returns:
        True if memory usage exceeds threshold
    """
    current_memory = get_memory_usage()
    if current_memory is None:
        return False
    return current_memory > threshold_mb


def force_cleanup_if_needed(
    threshold_mb: float = 1000.0, logger: logging.Logger | None = None
) -> bool:
    """
    Force memory cleanup if usage exceeds threshold.

    Args:
        threshold_mb: Memory threshold in MB
        logger: Optional logger instance

    Returns:
        True if cleanup was performed
    """
    if check_memory_threshold(threshold_mb):
        if logger:
            logger.warning("🧹 High memory usage detected, forcing cleanup")
        cleanup_memory()
        return True
    return False
