"""
Utility functions and helpers for MR BEN Trading System.
"""

from .error_handler import error_handler
from .helpers import (
    _apply_soft_gate,
    _rolling_atr,
    _swing_extrema,
    enforce_min_distance_and_round,
    is_spread_ok,
    round_price,
)
from .memory import cleanup_memory, log_memory_usage
from .performance_monitor import (
    get_performance_summary,
    record_operation_latency,
    record_trade_latency,
    start_performance_monitoring,
    stop_performance_monitoring,
)
from .position_management import (
    _count_open_positions,
    _get_open_positions,
    _modify_position_sltp,
    _prune_trailing_registry,
    get_position_summary,
    validate_position_data,
)

__all__ = [
    "round_price",
    "enforce_min_distance_and_round",
    "is_spread_ok",
    "_rolling_atr",
    "_swing_extrema",
    "_apply_soft_gate",
    "_get_open_positions",
    "_modify_position_sltp",
    "_prune_trailing_registry",
    "_count_open_positions",
    "validate_position_data",
    "get_position_summary",
    "log_memory_usage",
    "cleanup_memory",
    "error_handler",
    "start_performance_monitoring",
    "stop_performance_monitoring",
    "record_operation_latency",
    "record_trade_latency",
    "get_performance_summary",
]
