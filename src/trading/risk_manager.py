"""
Risk Management for MR BEN Trading Bot.
"""

from dataclasses import dataclass

from config.settings import settings
from core.logger import get_logger

logger = get_logger("trading.risk_manager")


@dataclass
class RiskParameters:
    """Risk management parameters."""

    base_risk: float = 0.01  # 1% base risk per trade
    max_risk: float = 0.03  # 3% maximum risk per trade
    min_risk: float = 0.005  # 0.5% minimum risk per trade
    max_open_trades: int = 3
    max_daily_loss: float = 0.05  # 5% maximum daily loss
    max_drawdown: float = 0.20  # 20% maximum drawdown
    correlation_threshold: float = 0.7  # Maximum correlation between positions


class RiskManager:
    """
    Professional risk manager for trading bots:
    - Fixed fractional risk (percent per trade)
    - Dynamic risk based on account growth/decline
    - Min/max lot control
    - Max simultaneous open trades
    - Trailing stop utility
    - Drawdown protection
    - Correlation analysis
    """

    def __init__(self, parameters: RiskParameters | None = None):
        """
        Initialize Risk Manager.

        Args:
            parameters: Risk management parameters
        """
        self.params = parameters or RiskParameters()
        self.logger = get_logger("risk_manager")

        # Load settings
        self.min_lot = settings.trading.min_lot
        self.max_lot = settings.trading.max_lot
        self.dynamic_sensitivity = settings.trading.dynamic_sensitivity

        # Performance tracking
        self.daily_pnl = 0.0
        self.max_balance = 0.0
        self.current_drawdown = 0.0

        self.logger.info("Risk Manager initialized")

    def calc_dynamic_risk(self, balance: float, start_balance: float) -> float:
        """
        Adjusts risk dynamically according to account growth.

        Args:
            balance: Current account balance
            start_balance: Initial account balance

        Returns:
            float: Adjusted risk percentage
        """
        if start_balance <= 0:
            return self.params.base_risk

        # Calculate growth percentage
        growth = (balance - start_balance) / start_balance

        # Adjust risk based on growth
        risk = self.params.base_risk
        risk += self.dynamic_sensitivity * growth * self.params.base_risk

        # Clamp to min/max bounds
        risk = max(self.params.min_risk, min(risk, self.params.max_risk))

        self.logger.debug(f"Dynamic risk: {risk:.4f} (growth: {growth:.2%})")
        return round(risk, 4)

    def calc_lot_size(
        self,
        balance: float,
        stop_loss_pips: float,
        pip_value: float,
        open_trades: int,
        start_balance: float | None = None,
        symbol: str = "",
    ) -> float:
        """
        Calculate trade lot size based on advanced risk management.

        Args:
            balance: Current account balance
            stop_loss_pips: Stop loss in pips
            pip_value: Value of one pip
            open_trades: Number of currently open trades
            start_balance: Initial account balance for dynamic risk
            symbol: Trading symbol for logging

        Returns:
            float: Calculated lot size
        """
        # Check maximum open trades
        if open_trades >= self.params.max_open_trades:
            self.logger.warning(f"Maximum open trades reached ({open_trades})")
            return 0.0

        # Check daily loss limit
        if self.daily_pnl < -(balance * self.params.max_daily_loss):
            self.logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            return 0.0

        # Check drawdown limit
        if self.current_drawdown > self.params.max_drawdown:
            self.logger.warning(f"Maximum drawdown reached: {self.current_drawdown:.2%}")
            return 0.0

        # Calculate risk percentage
        risk_perc = self.params.base_risk
        if start_balance:
            risk_perc = self.calc_dynamic_risk(balance, start_balance)

        # Calculate risk amount
        risk_amount = balance * risk_perc

        # Calculate lot size
        if stop_loss_pips * pip_value == 0:
            self.logger.warning("Invalid stop loss or pip value")
            return self.min_lot

        lot_size = risk_amount / (stop_loss_pips * pip_value)

        # Apply lot size constraints
        lot_size = max(self.min_lot, min(lot_size, self.max_lot))

        # Round to 2 decimal places
        lot_size = round(lot_size, 2)

        self.logger.info(
            f"Lot size calculated for {symbol}: {lot_size} "
            f"(risk: {risk_perc:.2%}, amount: {risk_amount:.2f})"
        )

        return lot_size

    def calculate_stop_loss(
        self, entry_price: float, direction: str, atr: float, atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate dynamic stop loss based on ATR.

        Args:
            entry_price: Entry price
            direction: 'BUY' or 'SELL'
            atr: Average True Range
            atr_multiplier: ATR multiplier for stop loss

        Returns:
            float: Stop loss price
        """
        stop_distance = atr * atr_multiplier

        if direction.upper() == 'BUY':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        self.logger.debug(f"Stop loss calculated: {stop_loss:.5f} (ATR: {atr:.5f})")
        return round(stop_loss, 5)

    def calculate_take_profit(
        self, entry_price: float, direction: str, stop_loss: float, risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take profit based on risk-reward ratio.

        Args:
            entry_price: Entry price
            direction: 'BUY' or 'SELL'
            stop_loss: Stop loss price
            risk_reward_ratio: Risk to reward ratio

        Returns:
            float: Take profit price
        """
        risk_distance = abs(entry_price - stop_loss)
        reward_distance = risk_distance * risk_reward_ratio

        if direction.upper() == 'BUY':
            take_profit = entry_price + reward_distance
        else:
            take_profit = entry_price - reward_distance

        self.logger.debug(f"Take profit calculated: {take_profit:.5f} (R:R: {risk_reward_ratio})")
        return round(take_profit, 5)

    def trailing_stop(
        self, entry_price: float, current_price: float, trailing_distance: float, is_buy: bool
    ) -> float:
        """
        Calculate trailing stop price.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            trailing_distance: Trailing distance in pips
            is_buy: True for buy position, False for sell

        Returns:
            float: New trailing stop price
        """
        if is_buy:
            new_sl = max(entry_price, current_price - trailing_distance)
        else:
            new_sl = min(entry_price, current_price + trailing_distance)

        return round(new_sl, 5)

    def update_performance(self, pnl: float, balance: float) -> None:
        """
        Update performance tracking.

        Args:
            pnl: Profit/Loss for the period
            balance: Current account balance
        """
        self.daily_pnl += pnl

        # Update maximum balance and drawdown
        if balance > self.max_balance:
            self.max_balance = balance

        if self.max_balance > 0:
            self.current_drawdown = (self.max_balance - balance) / self.max_balance

        self.logger.debug(
            f"Performance updated: PnL={pnl:.2f}, Drawdown={self.current_drawdown:.2%}"
        )

    def reset_daily_tracking(self) -> None:
        """Reset daily performance tracking."""
        self.daily_pnl = 0.0
        self.logger.info("Daily performance tracking reset")

    def check_correlation(self, positions: list) -> bool:
        """
        Check correlation between open positions.

        Args:
            positions: List of position dictionaries

        Returns:
            bool: True if correlation is acceptable
        """
        if len(positions) < 2:
            return True

        # Simple correlation check based on direction
        buy_positions = sum(1 for pos in positions if pos.get('direction') == 'BUY')
        sell_positions = len(positions) - buy_positions

        # If all positions are in the same direction, correlation is high
        if buy_positions == 0 or sell_positions == 0:
            self.logger.warning("High correlation detected: all positions in same direction")
            return False

        return True

    def get_risk_summary(self) -> dict[str, float]:
        """Get current risk management summary."""
        return {
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.params.max_drawdown,
            'daily_loss_limit': self.params.max_daily_loss,
            'base_risk': self.params.base_risk,
            'max_risk': self.params.max_risk,
            'min_risk': self.params.min_risk,
        }

    def validate_trade_parameters(
        self, symbol: str, lot_size: float, stop_loss_pips: float, balance: float
    ) -> tuple[bool, str]:
        """
        Validate trade parameters before execution.

        Args:
            symbol: Trading symbol
            lot_size: Lot size
            stop_loss_pips: Stop loss in pips
            balance: Account balance

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check lot size
        if lot_size < self.min_lot:
            return False, f"Lot size {lot_size} below minimum {self.min_lot}"

        if lot_size > self.max_lot:
            return False, f"Lot size {lot_size} above maximum {self.max_lot}"

        # Check stop loss
        if stop_loss_pips <= 0:
            return False, "Stop loss must be positive"

        # Check balance
        if balance <= 0:
            return False, "Invalid account balance"

        # Check daily loss limit
        if self.daily_pnl < -(balance * self.params.max_daily_loss):
            return False, "Daily loss limit exceeded"

        # Check drawdown limit
        if self.current_drawdown > self.params.max_drawdown:
            return False, "Maximum drawdown exceeded"

        return True, "Trade parameters valid"


# Global risk manager instance
risk_manager = RiskManager()
