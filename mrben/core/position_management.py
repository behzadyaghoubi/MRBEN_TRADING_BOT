#!/usr/bin/env python3
"""
MR BEN - Position Management System
Advanced position management with TP-Split, Breakeven, and Trailing Stop Loss
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import NamedTuple

from .advanced_portfolio import AdvancedPortfolioManager
from .advanced_position import AdvancedPositionManager
from .loggingx import logger


class PositionStatus(Enum):
    """Position lifecycle status"""

    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"
    CLOSED = "closed"


class TPLevel(NamedTuple):
    """Take Profit level definition"""

    price: float
    size_percent: float  # Percentage of position to close
    description: str


@dataclass
class PositionInfo:
    """Complete position information"""

    symbol: str
    ticket: int
    type: str  # "buy" or "sell"
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    status: PositionStatus = PositionStatus.OPEN

    # TP-Split tracking
    tp_levels: list[TPLevel] = field(default_factory=list)
    closed_size: float = 0.0
    remaining_size: float = 0.0

    # Breakeven tracking
    breakeven_triggered: bool = False
    breakeven_price: float = 0.0
    breakeven_time: datetime | None = None

    # Trailing stop tracking
    trailing_enabled: bool = False
    trailing_start_price: float = 0.0
    trailing_distance: float = 0.0
    current_trailing_stop: float = 0.0

    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self):
        if self.remaining_size == 0.0:
            self.remaining_size = self.size


class TPManager:
    """
    Take Profit Manager: Handles TP-Split functionality
    Allows partial position closure at multiple price levels
    """

    def __init__(self, config):
        self.enabled = getattr(config.position_management, 'tp_split_enabled', True)
        self.default_levels = getattr(config.position_management, 'default_tp_levels', [])
        self.min_size_percent = getattr(config.position_management, 'min_tp_size_percent', 10.0)

    def create_tp_levels(
        self, entry_price: float, direction: int, atr_value: float, confidence: float
    ) -> list[TPLevel]:
        """
        Create optimal TP levels based on market conditions

        Args:
            entry_price: Position entry price
            direction: Trade direction (+1 for buy, -1 for sell)
            atr_value: Current ATR value
            confidence: Signal confidence

        Returns:
            List of TPLevel objects
        """
        if not self.enabled:
            return []

        # Base TP distance using ATR
        base_tp_distance = atr_value * 2.0  # 2x ATR for first TP

        # Adjust based on confidence
        if confidence > 0.8:
            base_tp_distance *= 1.2  # Higher confidence = further TP
        elif confidence < 0.6:
            base_tp_distance *= 0.8  # Lower confidence = closer TP

        # Create TP levels
        tp_levels = []

        # First TP: 30% at 2x ATR
        tp1_distance = base_tp_distance
        tp1_price = entry_price + (direction * tp1_distance)
        tp_levels.append(
            TPLevel(price=tp1_price, size_percent=30.0, description="First TP - 30% at 2x ATR")
        )

        # Second TP: 40% at 3x ATR
        tp2_distance = base_tp_distance * 1.5
        tp2_price = entry_price + (direction * tp2_distance)
        tp_levels.append(
            TPLevel(price=tp2_price, size_percent=40.0, description="Second TP - 40% at 3x ATR")
        )

        # Third TP: 30% at 4x ATR (let it run)
        tp3_distance = base_tp_distance * 2.0
        tp3_price = entry_price + (direction * tp3_distance)
        tp_levels.append(
            TPLevel(price=tp3_price, size_percent=30.0, description="Final TP - 30% at 4x ATR")
        )

        return tp_levels

    def check_tp_triggers(
        self, position: PositionInfo, current_price: float
    ) -> list[tuple[TPLevel, float]]:
        """
        Check which TP levels have been triggered

        Returns:
            List of (TPLevel, size_to_close) tuples
        """
        if not self.tp_levels:
            return []

        triggered_levels = []

        for tp_level in self.tp_levels:
            if self._is_tp_triggered(position, tp_level, current_price):
                size_to_close = self._calculate_tp_size(position, tp_level)
                if size_to_close > 0:
                    triggered_levels.append((tp_level, size_to_close))

        return triggered_levels

    def _is_tp_triggered(
        self, position: PositionInfo, tp_level: TPLevel, current_price: float
    ) -> bool:
        """Check if a TP level has been triggered"""
        if position.type == "buy":
            return current_price >= tp_level.price
        else:  # sell
            return current_price <= tp_level.price

    def _calculate_tp_size(self, position: PositionInfo, tp_level: TPLevel) -> float:
        """Calculate how much size to close at this TP level"""
        # Check if we've already closed at this level
        if tp_level in [tp for tp, _ in position.tp_levels]:
            return 0.0

        # Calculate size based on percentage
        size_to_close = (position.remaining_size * tp_level.size_percent) / 100.0

        # Ensure minimum size
        min_size = (position.size * self.min_size_percent) / 100.0
        if size_to_close < min_size:
            size_to_close = min_size

        # Don't close more than remaining
        return min(size_to_close, position.remaining_size)


class BreakevenManager:
    """
    Breakeven Manager: Moves stop loss to entry price after certain profit
    Protects profits and reduces risk
    """

    def __init__(self, config):
        self.enabled = getattr(config.position_management, 'breakeven_enabled', True)
        self.trigger_distance = getattr(
            config.position_management, 'breakeven_trigger_distance', 0.5
        )
        self.breakeven_distance = getattr(config.position_management, 'breakeven_distance', 0.1)

    def check_breakeven_trigger(self, position: PositionInfo, current_price: float) -> bool:
        """Check if breakeven should be triggered"""
        if not self.enabled or position.breakeven_triggered:
            return False

        # Calculate profit distance
        if position.type == "buy":
            profit_distance = current_price - position.entry_price
        else:  # sell
            profit_distance = position.entry_price - current_price

        # Check if we've reached trigger distance
        if profit_distance >= self.trigger_distance:
            return True

        return False

    def execute_breakeven(self, position: PositionInfo) -> float:
        """Execute breakeven by moving stop loss"""
        if position.breakeven_triggered:
            return position.stop_loss

        # Calculate new stop loss (entry price + small buffer)
        if position.type == "buy":
            new_stop_loss = position.entry_price + self.breakeven_distance
        else:  # sell
            new_stop_loss = position.entry_price - self.breakeven_distance

        # Update position
        position.breakeven_triggered = True
        position.breakeven_price = new_stop_loss
        position.breakeven_time = datetime.now(UTC)
        position.stop_loss = new_stop_loss
        position.status = PositionStatus.BREAKEVEN

        logger.bind(evt="POSITION").info(
            "breakeven_triggered",
            symbol=position.symbol,
            ticket=position.ticket,
            new_stop_loss=new_stop_loss,
        )

        return new_stop_loss


class TrailingStopManager:
    """
    Trailing Stop Manager: Dynamic stop loss that follows price
    Locks in profits while allowing upside
    """

    def __init__(self, config):
        self.enabled = getattr(config.position_management, 'trailing_enabled', True)
        self.activation_distance = getattr(
            config.position_management, 'trailing_activation_distance', 1.0
        )
        self.trailing_distance = getattr(config.position_management, 'trailing_distance', 0.5)
        self.trailing_multiplier = getattr(config.position_management, 'trailing_multiplier', 1.0)

    def check_trailing_activation(self, position: PositionInfo, current_price: float) -> bool:
        """Check if trailing stop should be activated"""
        if not self.enabled or position.trailing_enabled:
            return False

        # Calculate profit distance
        if position.type == "buy":
            profit_distance = current_price - position.entry_price
        else:  # sell
            profit_distance = position.entry_price - current_price

        # Check if we've reached activation distance
        if profit_distance >= self.activation_distance:
            return True

        return False

    def activate_trailing(self, position: PositionInfo, current_price: float) -> float:
        """Activate trailing stop"""
        if position.trailing_enabled:
            return position.current_trailing_stop

        # Calculate initial trailing stop
        if position.type == "buy":
            initial_stop = current_price - self.trailing_distance
        else:  # sell
            initial_stop = current_price + self.trailing_distance

        # Update position
        position.trailing_enabled = True
        position.trailing_start_price = current_price
        position.trailing_distance = self.trailing_distance
        position.current_trailing_stop = initial_stop
        position.status = PositionStatus.TRAILING

        logger.bind(evt="POSITION").info(
            "trailing_activated",
            symbol=position.symbol,
            ticket=position.ticket,
            initial_stop=initial_stop,
        )

        return initial_stop

    def update_trailing_stop(self, position: PositionInfo, current_price: float) -> float:
        """Update trailing stop based on current price"""
        if not position.trailing_enabled:
            return position.stop_loss

        # Calculate new trailing stop
        if position.type == "buy":
            new_stop = current_price - self.trailing_distance
            # Only move stop up (never down)
            if new_stop > position.current_trailing_stop:
                position.current_trailing_stop = new_stop
                position.stop_loss = new_stop
        else:  # sell
            new_stop = current_price + self.trailing_distance
            # Only move stop down (never up)
            if new_stop < position.current_trailing_stop:
                position.current_trailing_stop = new_stop
                position.stop_loss = new_stop

        return position.current_trailing_stop


class PositionManager:
    """
    Main Position Manager: Coordinates all position management features
    """

    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config.position_management, 'enabled', True)

        # Initialize managers
        self.tp_manager = TPManager(config)
        self.breakeven_manager = BreakevenManager(config)
        self.trailing_manager = TrailingStopManager(config)

        # Initialize advanced position manager
        try:
            self.advanced_manager = AdvancedPositionManager(
                config_path="advanced_position_config.json",
                enable_ml=True,
                enable_portfolio_optimization=True,
                enable_dynamic_adjustment=True,
            )
            self.logger.bind(evt="POSITION").info("advanced_position_manager_initialized")
        except Exception as e:
            self.logger.bind(evt="POSITION").warning(
                "advanced_position_manager_initialization_failed", error=str(e)
            )
            self.advanced_manager = None

        # Initialize advanced portfolio manager
        try:
            self.portfolio_manager = AdvancedPortfolioManager(
                config_path="advanced_portfolio_config.json",
                enable_ml=True,
                enable_correlation=True,
                enable_optimization=True,
            )
            self.logger.bind(evt="POSITION").info("advanced_portfolio_manager_initialized")
        except Exception as e:
            self.logger.bind(evt="POSITION").warning(
                "advanced_portfolio_manager_initialization_failed", error=str(e)
            )
            self.portfolio_manager = None

        # Active positions tracking
        self.active_positions: dict[int, PositionInfo] = {}

        self.logger = logger

    def open_position(
        self,
        symbol: str,
        ticket: int,
        position_type: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr_value: float,
        confidence: float,
    ) -> PositionInfo:
        """Open a new position with full management"""
        if not self.enabled:
            return None

        # Create position info
        position = PositionInfo(
            symbol=symbol,
            ticket=ticket,
            type=position_type,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(UTC),
            remaining_size=size,
        )

        # Create TP levels
        direction = 1 if position_type == "buy" else -1
        position.tp_levels = self.tp_manager.create_tp_levels(
            entry_price, direction, atr_value, confidence
        )

        # Store position
        self.active_positions[ticket] = position

        self.logger.bind(evt="POSITION").info(
            "position_opened",
            symbol=symbol,
            ticket=ticket,
            type=position_type,
            size=size,
            entry_price=entry_price,
            tp_levels=len(position.tp_levels),
        )

        return position

    def update_position(self, ticket: int, current_price: float, unrealized_pnl: float) -> dict:
        """Update position and check for management actions"""
        if not self.enabled:
            return {}

        position = self.active_positions.get(ticket)
        if not position:
            return {}

        # Update P&L
        position.unrealized_pnl = unrealized_pnl

        actions = {}

        # Check TP triggers
        tp_triggers = self.tp_manager.check_tp_triggers(position, current_price)
        if tp_triggers:
            actions['tp_triggers'] = tp_triggers

        # Check breakeven trigger
        if self.breakeven_manager.check_breakeven_trigger(position, current_price):
            new_stop = self.breakeven_manager.execute_breakeven(position)
            actions['breakeven'] = new_stop

        # Check trailing stop activation
        if self.trailing_manager.check_trailing_activation(position, current_price):
            initial_stop = self.trailing_manager.activate_trailing(position, current_price)
            actions['trailing_activated'] = initial_stop

        # Update trailing stop if active
        if position.trailing_enabled:
            new_trailing_stop = self.trailing_manager.update_trailing_stop(position, current_price)
            if new_trailing_stop != position.stop_loss:
                actions['trailing_updated'] = new_trailing_stop

        return actions

    def close_position_partial(self, ticket: int, tp_level: TPLevel, size_to_close: float) -> bool:
        """Partially close position at TP level"""
        position = self.active_positions.get(ticket)
        if not position:
            return False

        # Update position
        position.closed_size += size_to_close
        position.remaining_size -= size_to_close

        # Record TP level execution
        position.tp_levels.append(tp_level)

        # Update status
        if position.remaining_size > 0:
            position.status = PositionStatus.PARTIALLY_CLOSED
        else:
            position.status = PositionStatus.CLOSED

        self.logger.bind(evt="POSITION").info(
            "position_partially_closed",
            symbol=position.symbol,
            ticket=ticket,
            size_closed=size_to_close,
            remaining_size=position.remaining_size,
            tp_level=tp_level.description,
        )

        return True

    def close_position_full(self, ticket: int, realized_pnl: float) -> bool:
        """Fully close position"""
        position = self.active_positions.get(ticket)
        if not position:
            return False

        # Update position
        position.realized_pnl = realized_pnl
        position.status = PositionStatus.CLOSED
        position.remaining_size = 0.0

        # Remove from active positions
        del self.active_positions[ticket]

        self.logger.bind(evt="POSITION").info(
            "position_closed", symbol=position.symbol, ticket=ticket, realized_pnl=realized_pnl
        )

        return True

    def get_position_status(self, ticket: int) -> PositionInfo | None:
        """Get current position status"""
        return self.active_positions.get(ticket)

    def get_all_positions(self) -> list[PositionInfo]:
        """Get all active positions"""
        return list(self.active_positions.values())

    def get_position_summary(self) -> dict:
        """Get summary of all positions"""
        if not self.active_positions:
            return {"total_positions": 0}

        total_size = sum(pos.remaining_size for pos in self.active_positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())

        status_counts = {}
        for pos in self.active_positions.values():
            status = pos.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        summary = {
            "total_positions": len(self.active_positions),
            "total_size": total_size,
            "total_unrealized_pnl": total_unrealized_pnl,
            "status_distribution": status_counts,
        }

        # Add advanced position management status if available
        if self.advanced_manager:
            try:
                portfolio_summary = self.advanced_manager.get_portfolio_summary()
                summary["advanced_position_management"] = portfolio_summary
            except Exception as e:
                self.logger.warning(f"Failed to get advanced position management status: {e}")
                summary["advanced_position_management"] = {"error": str(e)}

        # Add advanced portfolio management status if available
        if self.portfolio_manager:
            try:
                portfolio_status = self.portfolio_manager.get_portfolio_summary()
                summary["advanced_portfolio_management"] = portfolio_status
            except Exception as e:
                self.logger.warning(f"Failed to get advanced portfolio management status: {e}")
                summary["advanced_portfolio_management"] = {"error": str(e)}

        return summary
