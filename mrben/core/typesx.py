#!/usr/bin/env python3
"""
MR BEN - Shared Types for A/B Testing
Common decision contract between Control and Pro deciders
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Levels:
    """Price levels for SL/TP"""
    sl: float
    tp1: float
    tp2: float


@dataclass
class DecisionCard:
    """Decision card with all necessary information"""
    action: str          # ENTER | HOLD
    dir: int             # +1 buy | -1 sell | 0 flat
    reason: str          # Explanation for pass/block
    score: float         # [0..1]
    dyn_conf: float      # [0..1]
    lot: float = 0.0
    levels: Optional[Levels] = None
    track: str = "pro"   # 'pro' or 'control'
    
    def __post_init__(self):
        """Validate decision card"""
        assert self.action in ["ENTER", "HOLD"], f"Invalid action: {self.action}"
        assert self.dir in [-1, 0, 1], f"Invalid direction: {self.dir}"
        assert 0.0 <= self.score <= 1.0, f"Invalid score: {self.score}"
        assert 0.0 <= self.dyn_conf <= 1.0, f"Invalid confidence: {self.dyn_conf}"
        assert self.track in ["pro", "control"], f"Invalid track: {self.track}"
        
        # Validate levels if action is ENTER
        if self.action == "ENTER":
            assert self.dir != 0, "Direction must be +1 or -1 for ENTER"
            assert self.lot > 0, "Lot size must be positive for ENTER"
            assert self.levels is not None, "Levels required for ENTER"
        else:  # HOLD
            assert self.dir == 0, "Direction must be 0 for HOLD"
            assert self.lot == 0, "Lot size must be 0 for HOLD"
            assert self.levels is None, "Levels not allowed for HOLD"


@dataclass
class MarketContext:
    """Market context for decision making"""
    price: float
    bid: float
    ask: float
    atr_pts: float
    sma20: float
    sma50: float
    session: str
    regime: str
    equity: float
    balance: float
    spread_pts: float
    open_positions: int
    
    @property
    def spread(self) -> float:
        """Calculate current spread"""
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2.0
    
    @property
    def trend_direction(self) -> int:
        """Determine trend direction from SMA"""
        if self.sma20 > self.sma50:
            return 1
        elif self.sma20 < self.sma50:
            return -1
        else:
            return 0
