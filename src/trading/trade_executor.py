"""
Trade Execution for MR BEN Trading Bot.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import MetaTrader5 as mt5

from config.settings import settings
from core.logger import get_logger

logger = get_logger("trading.executor")


@dataclass
class TradeResult:
    """Result of a trade execution."""

    success: bool
    retcode: int
    comment: str
    order_id: int | None = None
    price: float | None = None


class TradeExecutor:
    """Handles trade execution through MetaTrader 5."""

    def __init__(self):
        """Initialize the trade executor."""
        self.logger = get_logger("trade_executor")
        self.magic_number = settings.trading.magic_number
        self.deviation = settings.trading.deviation

        self.logger.info("Trade Executor initialized")

    def execute_trade(self, trade_signal) -> TradeResult:
        """
        Execute a trade based on the signal.

        Args:
            trade_signal: TradeSignal object containing trade parameters

        Returns:
            TradeResult: Result of the trade execution
        """
        try:
            # Validate trade parameters
            if not self._validate_trade_signal(trade_signal):
                return TradeResult(False, -1, "Invalid trade signal")

            # Get symbol info
            symbol_info = mt5.symbol_info(trade_signal.symbol)
            if symbol_info is None:
                return TradeResult(False, -1, f"Symbol {trade_signal.symbol} not found")

            # Prepare trade request
            request = self._prepare_trade_request(trade_signal, symbol_info)

            # Send order
            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(
                    f"Trade executed successfully: {trade_signal.action} {trade_signal.symbol} "
                    f"Lot: {trade_signal.lot_size}, Price: {result.price}"
                )
                return TradeResult(
                    success=True,
                    retcode=result.retcode,
                    comment=result.comment,
                    order_id=getattr(result, 'order', None),
                    price=result.price,
                )
            else:
                self.logger.error(f"Trade execution failed: {result.retcode} - {result.comment}")
                return TradeResult(success=False, retcode=result.retcode, comment=result.comment)

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return TradeResult(False, -1, str(e))

    def _validate_trade_signal(self, trade_signal) -> bool:
        """Validate trade signal parameters."""
        if not trade_signal.symbol:
            self.logger.error("Symbol is required")
            return False

        if trade_signal.action not in ['BUY', 'SELL']:
            self.logger.error(f"Invalid action: {trade_signal.action}")
            return False

        if trade_signal.lot_size <= 0:
            self.logger.error(f"Invalid lot size: {trade_signal.lot_size}")
            return False

        if trade_signal.entry_price <= 0:
            self.logger.error(f"Invalid entry price: {trade_signal.entry_price}")
            return False

        return True

    def _prepare_trade_request(self, trade_signal, symbol_info) -> dict[str, Any]:
        """Prepare MT5 trade request."""
        # Determine order type
        if trade_signal.action == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = symbol_info.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = symbol_info.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": trade_signal.symbol,
            "volume": trade_signal.lot_size,
            "type": order_type,
            "price": price,
            "sl": trade_signal.stop_loss,
            "tp": trade_signal.take_profit,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": f"MRBEN {trade_signal.strategy}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        return request

    def close_position(self, position_id: int, symbol: str, volume: float) -> TradeResult:
        """
        Close a specific position.

        Args:
            position_id: Position ticket
            symbol: Trading symbol
            volume: Position volume

        Returns:
            TradeResult: Result of the close operation
        """
        try:
            # Get position info
            position = mt5.positions_get(ticket=position_id)
            if not position:
                return TradeResult(False, -1, "Position not found")

            position = position[0]

            # Determine close order type
            if position.type == mt5.POSITION_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask

            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": position_id,
                "price": price,
                "deviation": self.deviation,
                "magic": self.magic_number,
                "comment": "MRBEN Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }

            # Send close order
            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Position {position_id} closed successfully")
                return TradeResult(
                    success=True, retcode=result.retcode, comment=result.comment, price=result.price
                )
            else:
                self.logger.error(f"Failed to close position {position_id}: {result.comment}")
                return TradeResult(success=False, retcode=result.retcode, comment=result.comment)

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return TradeResult(False, -1, str(e))

    def modify_position(
        self, position_id: int, stop_loss: float, take_profit: float
    ) -> TradeResult:
        """
        Modify position stop loss and take profit.

        Args:
            position_id: Position ticket
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            TradeResult: Result of the modification
        """
        try:
            # Get position info
            position = mt5.positions_get(ticket=position_id)
            if not position:
                return TradeResult(False, -1, "Position not found")

            position = position[0]

            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "sl": stop_loss,
                "tp": take_profit,
                "position": position_id,
            }

            # Send modification order
            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Position {position_id} modified successfully")
                return TradeResult(success=True, retcode=result.retcode, comment=result.comment)
            else:
                self.logger.error(f"Failed to modify position {position_id}: {result.comment}")
                return TradeResult(success=False, retcode=result.retcode, comment=result.comment)

        except Exception as e:
            self.logger.error(f"Error modifying position: {e}")
            return TradeResult(False, -1, str(e))

    def get_account_info(self) -> dict[str, Any] | None:
        """Get account information."""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None

            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'profit': account_info.profit,
                'currency': account_info.currency,
                'leverage': account_info.leverage,
            }

        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None

    def get_positions(self, symbol: str | None = None) -> list:
        """Get open positions."""
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                return []

            return [
                {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'time': datetime.fromtimestamp(pos.time),
                }
                for pos in positions
            ]

        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []


# Global trade executor instance
trade_executor = TradeExecutor()
