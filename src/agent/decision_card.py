#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Decision Card Module
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

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
    atr: Optional[float] = None
    session: Optional[str] = None
    reasons: Optional[Dict[str, Any]] = None

@dataclass
class HealthEvent:
    """System health event"""
    timestamp: datetime
    event_type: str
    severity: str
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class AgentAction:
    """AI agent action"""
    timestamp: datetime
    action_type: str
    reason: str
    confidence: float
    details: Optional[Dict[str, Any]] = None
