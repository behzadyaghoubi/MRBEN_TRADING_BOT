#!/usr/bin/env python3
"""
MR BEN - Paper Broker for Control Track
Simulates trading for the control decider without real execution
"""

from __future__ import annotations

from dataclasses import dataclass

from .loggingx import logger
from .metricsx import observe_trade_close, observe_trade_open
from .typesx import DecisionCard


@dataclass
class PaperPos:
    """Paper position for tracking"""

    symbol: str
    dir: int
    entry: float
    sl: float
    tp1: float
    tp2: float
    lot: float
    tp1_hit: bool = False
    entry_time: str | None = None


class PaperBroker:
    """Paper broker for control track simulation"""

    def __init__(self, symbol: str, track: str = "control"):
        self.symbol = symbol
        self.track = track
        self.pos: PaperPos | None = None
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.breakevens = 0

        logger.bind(evt="PAPER").info("paper_broker_initialized", symbol=symbol, track=track)

    def open(self, dc: DecisionCard) -> None:
        """Open a paper position based on decision card"""
        if self.pos is not None:
            logger.bind(evt="PAPER").warning(
                "position_already_open", symbol=self.symbol, track=self.track
            )
            return

        if dc.action != "ENTER":
            logger.bind(evt="PAPER").warning(
                "invalid_action_for_open", action=dc.action, track=self.track
            )
            return

        # Create paper position
        self.pos = PaperPos(
            symbol=self.symbol,
            dir=dc.dir,
            entry=dc.levels.sl + (dc.levels.tp1 - dc.levels.sl) * 0.5,  # Mid-point entry
            sl=dc.levels.sl,
            tp1=dc.levels.tp1,
            tp2=dc.levels.tp2,
            lot=dc.lot,
            entry_time="now",  # Simplified for demo
        )

        # Log position opening
        logger.bind(evt="PAPER").info(
            "position_opened",
            symbol=self.symbol,
            track=self.track,
            direction="buy" if dc.dir > 0 else "sell",
            lot=dc.lot,
            entry=self.pos.entry,
            sl=dc.levels.sl,
            tp1=dc.levels.tp1,
            tp2=dc.levels.tp2,
        )

        # Observe trade opening in metrics
        observe_trade_open(self.symbol, dc.dir, self.track)

        self.total_trades += 1

    def on_tick(self, bid: float, ask: float, atr_pts: float) -> None:
        """Process tick data for position management"""
        if self.pos is None:
            return

        p = self.pos
        price = bid if p.dir < 0 else ask  # Use bid for sell, ask for buy

        # Check Stop Loss
        if self._check_stop_loss(price):
            r_multiple = self._calculate_r_multiple(price)
            self._close_position("stop_loss", r_multiple)
            return

        # Check TP1
        if not p.tp1_hit and self._check_tp1(price):
            p.tp1_hit = True
            # Move SL to breakeven
            old_sl = p.sl
            p.sl = p.entry

            logger.bind(evt="PAPER").info(
                "tp1_hit_breakeven",
                symbol=self.symbol,
                track=self.track,
                old_sl=old_sl,
                new_sl=p.entry,
            )
            return

        # Check TP2
        if p.tp1_hit and self._check_tp2(price):
            r_multiple = self._calculate_r_multiple(price)
            self._close_position("tp2", r_multiple)
            return

    def _check_stop_loss(self, price: float) -> bool:
        """Check if stop loss is hit"""
        p = self.pos
        if p.dir > 0:  # Buy position
            return price <= p.sl
        else:  # Sell position
            return price >= p.sl

    def _check_tp1(self, price: float) -> bool:
        """Check if TP1 is hit"""
        p = self.pos
        if p.dir > 0:  # Buy position
            return price >= p.tp1
        else:  # Sell position
            return price <= p.tp1

    def _check_tp2(self, price: float) -> bool:
        """Check if TP2 is hit"""
        p = self.pos
        if p.dir > 0:  # Buy position
            return price >= p.tp2
        else:  # Sell position
            return price <= p.tp2

    def _calculate_r_multiple(self, exit_price: float) -> float:
        """Calculate R-multiple for the trade"""
        p = self.pos
        entry = p.entry
        sl = p.sl

        # Calculate risk in price units
        risk = abs(entry - sl)
        if risk == 0:
            return 0.0

        # Calculate reward in price units
        if p.tp1_hit:
            # TP2 hit - calculate based on TP2
            if p.dir > 0:  # Buy
                reward = p.tp2 - entry
            else:  # Sell
                reward = entry - p.tp2
        else:
            # SL hit - calculate based on exit
            if p.dir > 0:  # Buy
                reward = exit_price - entry
            else:  # Sell
                reward = entry - exit_price

        # Calculate R-multiple
        r_multiple = reward / risk

        # Apply direction
        if p.dir > 0:  # Buy
            if exit_price < entry:  # Loss
                r_multiple = -r_multiple
        else:  # Sell
            if exit_price > entry:  # Loss
                r_multiple = -r_multiple

        return round(r_multiple, 2)

    def _close_position(self, reason: str, r_multiple: float) -> None:
        """Close the paper position"""
        if self.pos is None:
            return

        p = self.pos

        # Update statistics
        if abs(r_multiple) < 0.01:
            self.breakevens += 1
        elif r_multiple > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Log position closing
        logger.bind(evt="PAPER").info(
            "position_closed",
            symbol=self.symbol,
            track=self.track,
            reason=reason,
            r_multiple=r_multiple,
            direction="buy" if p.dir > 0 else "sell",
        )

        # Observe trade closing in metrics
        observe_trade_close(self.symbol, p.dir, self.track, r_multiple)

        # Clear position
        self.pos = None

    def close_all(self) -> None:
        """Close all positions (for cleanup)"""
        if self.pos:
            # Mark-to-market close
            r_multiple = 0.0  # Neutral close
            self._close_position("cleanup", r_multiple)

    def get_statistics(self) -> dict:
        """Get trading statistics"""
        win_rate = (self.wins / max(self.total_trades, 1)) * 100

        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "breakevens": self.breakevens,
            "win_rate_pct": round(win_rate, 2),
            "current_position": "open" if self.pos else "closed",
        }

    def get_position_summary(self) -> dict | None:
        """Get current position summary"""
        if self.pos is None:
            return None

        p = self.pos
        return {
            "symbol": p.symbol,
            "direction": "buy" if p.dir > 0 else "sell",
            "entry": p.entry,
            "stop_loss": p.sl,
            "tp1": p.tp1,
            "tp2": p.tp2,
            "lot": p.lot,
            "tp1_hit": p.tp1_hit,
            "entry_time": p.entry_time,
        }
