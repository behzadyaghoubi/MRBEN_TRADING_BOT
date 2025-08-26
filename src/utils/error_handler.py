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
    op_name: str, logger: logging.Logger | None = None, metrics: PerformanceMetrics | None = None
) -> Generator[None, None, None]:
    """
    Usage:
        with error_handler("fetch-prices", logger, metrics):
            dangerous_call()
    On exception: logs with stack, increments metrics.error_count, and re-raises.
    """
    lg = _get_logger(logger)
    try:
        yield
    except Exception as e:  # no bare except
        lg.error("Error in %s: %s", op_name, e, exc_info=True)
        if metrics is not None and hasattr(metrics, "inc_error"):
            try:
                metrics.inc_error(e)
            except Exception:
                # metrics failure must not swallow original error
                pass
        # IMPORTANT: re-raise so that contextmanager "stops after throw"
        raise
