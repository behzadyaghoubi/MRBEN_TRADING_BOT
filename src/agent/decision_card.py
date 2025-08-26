#!/usr/bin/env python3
"""
MR BEN Decision Card Module
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DecisionCard:
    """Trading decision card with all relevant information"""

    timestamp: datetime
    symbol: str
    signal: int
    confidence: float
    consecutive_signals: int
    price: float
    sma20: float
    sma50: float
    regime: str
    adj_conf: float
    threshold: float
    allow_trade: bool
    spread: float
    open_positions: int
    atr: float | None = None
    session: str | None = None
    reasons: dict[str, Any] | None = None


@dataclass
class HealthEvent:
    """System health event"""

    timestamp: datetime
    event_type: str
    severity: str
    message: str
    details: dict[str, Any] | None = None


@dataclass
class AgentAction:
    """AI agent action"""

    timestamp: datetime
    action_type: str
    reason: str
    confidence: float
    details: dict[str, Any] | None = None
