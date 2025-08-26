from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager

try:
    from core.metrics import PerformanceMetrics
except Exception:
    PerformanceMetrics = None  # type: ignore


def _get_logger(logger: logging.Logger | None) -> logging.Logger:
    if logger is not None:
        return logger
    import logging as _logging

    lg = _logging.getLogger("test:error_handler")
    if not lg.handlers:
        h = _logging.StreamHandler()
        fmt = _logging.Formatter("%(levelname)s %(name)s:%(lineno)d %(message)s")
        h.setFormatter(fmt)
        lg.addHandler(h)
        lg.setLevel(_logging.INFO)
    return lg


@contextmanager
def error_handler(
    op_name_or_logger,
    logger_or_op: logging.Logger | None = None,
    metrics_or_fallback: PerformanceMetrics | None = None,
) -> Generator[None, None, None]:
    """
    Backward compatible error handler supporting both signatures:

    Old signature (for tests):
        with error_handler(logger, "operation", fallback_value):
            dangerous_call()

    New signature:
        with error_handler("operation", logger, metrics):
            dangerous_call()

    On exception: logs with stack, increments metrics.error_count, and re-raises.
    """
    # Detect signature based on first argument type
    if isinstance(op_name_or_logger, logging.Logger):
        # Old signature: error_handler(logger, op_name, fallback)
        lg = op_name_or_logger
        op_name = str(logger_or_op) if logger_or_op else "unknown_operation"
        fallback_value = metrics_or_fallback
    else:
        # New signature: error_handler(op_name, logger, metrics)
        op_name = str(op_name_or_logger)
        lg = _get_logger(logger_or_op)
        fallback_value = None

    try:
        yield
    except Exception as e:  # no bare except
        lg.error("Error in %s: %s", op_name, e, exc_info=True)

        # Handle metrics if available (new signature)
        if not isinstance(op_name_or_logger, logging.Logger) and metrics_or_fallback is not None:
            if hasattr(metrics_or_fallback, "inc_error"):
                try:
                    metrics_or_fallback.inc_error(e)
                except Exception:
                    # metrics failure must not swallow original error
                    pass

        # IMPORTANT: re-raise so that contextmanager "stops after throw"
        raise
