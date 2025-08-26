"""
Utility functions and helpers for MR BEN Trading System.
"""

from .helpers import (
    round_price,
    enforce_min_distance_and_round,
    is_spread_ok,
    _rolling_atr,
    _swing_extrema,
    _apply_soft_gate
)
from .position_management import (
    _get_open_positions,
    _modify_position_sltp,
    _prune_trailing_registry,
    _count_open_positions,
    validate_position_data,
    get_position_summary
)
from .memory import log_memory_usage, cleanup_memory
from .error_handler import error_handler

from .performance_monitor import (
    start_performance_monitoring,
    stop_performance_monitoring,
    record_operation_latency,
    record_trade_latency,
    get_performance_summary
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
    "get_performance_summary"
]
