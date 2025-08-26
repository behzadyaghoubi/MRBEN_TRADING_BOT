"""
Error handling utilities for MR BEN Trading System.
"""

import logging
from contextlib import contextmanager
from typing import Any


@contextmanager
def error_handler(logger: logging.Logger, operation: str, fallback_value: Any | None = None):
    """
    Context manager for consistent error handling.

    Args:
        logger: Logger instance for error reporting
        operation: Description of the operation being performed
        fallback_value: Value to yield if an error occurs

    Yields:
        The fallback value if an error occurs, otherwise nothing
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation}: {e}")
        if fallback_value is not None:
            yield fallback_value
        else:
            raise
