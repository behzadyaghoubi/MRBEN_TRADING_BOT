"""
Enhanced trade execution for MR BEN Trading System.
Handles order placement, modification, and management.
"""

import logging
import time
from typing import Any

# Global MT5 availability flag
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class EnhancedTradeExecutor:
    """Enhanced trade execution with retry logic and error handling."""

    def __init__(self, risk_manager):
        """
        Initialize trade executor.

        Args:
            risk_manager: Risk manager instance
        """
        self.risk_manager = risk_manager
        self.trailing_meta = risk_manager.trailing_stops  # Reference to same dictionary
        self.logger = logging.getLogger("EnhancedTradeExecutor")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
            self.logger.addHandler(h)

    def modify_stop_loss(
        self,
        symbol: str,
        position_ticket: int,
        new_sl: float,
        current_tp: float | None = None,
        magic: int | None = None,
        deviation: int = 20,
    ) -> bool:
        """
        Safely modify stop loss of an open position.

        Args:
            symbol: Trading symbol
            position_ticket: Position ticket number
            new_sl: New stop loss price
            current_tp: Current take profit price (to preserve)
            magic: Magic number
            deviation: Price deviation allowance

        Returns:
            True if successful
        """
        try:
            if not MT5_AVAILABLE:
                return False

            # Get position
            pos = None
            positions = mt5.positions_get(symbol=symbol) or []
            for p in positions:
                if p.ticket == position_ticket:
                    pos = p
                    break

            if not pos:
                self.logger.error(f"Position {position_ticket} not found to modify SLTP")
                return False

            is_buy = pos.type == 0
            entry_like = pos.price_open  # Use entry price instead of current tick

            # Preserve current TP if not specified
            tp_to_send = current_tp if current_tp is not None else float(pos.tp or 0.0)

            # Enforce minimum distance and rounding
            from utils.helpers import enforce_min_distance_and_round

            adj_sl, adj_tp = enforce_min_distance_and_round(
                symbol, entry_like, new_sl, tp_to_send if tp_to_send else new_sl, is_buy
            )

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": int(position_ticket),
                "symbol": symbol,
                "sl": float(adj_sl),
                "tp": float(tp_to_send) if tp_to_send else 0.0,
            }

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(
                    f"SLTP modify failed pos {position_ticket}: {getattr(result,'retcode',None)} {getattr(result,'comment',None)}"
                )
                return False

            # Update trailing metadata
            st = self.trailing_meta.get(position_ticket)
            if st:
                st['current_sl'] = float(adj_sl)

            self.logger.info(
                f"SL modified for position {position_ticket}: SL={adj_sl:.2f} (kept TP={tp_to_send:.2f})"
            )
            return True
        except Exception as e:
            self.logger.error(f"Modify SL error: {e}")
            return False

    def update_trailing_stops(self, symbol: str) -> int:
        """
        Update trailing stops for all positions.

        Args:
            symbol: Trading symbol

        Returns:
            Number of positions updated
        """
        mods = self.risk_manager.update_trailing_stops(symbol)
        cnt = 0

        for m in mods:
            if self.modify_stop_loss(symbol, m['ticket'], m['new_sl'], None):
                cnt += 1
                self.logger.info(f"↗ Trailing move | pos={m['ticket']} new_sl={m['new_sl']:.2f}")

        if cnt > 0:
            self.logger.info(f"✅ Trailing updated: {cnt} position(s)")

        return cnt

    def get_account_info(self) -> dict[str, float]:
        """
        Get account information.

        Returns:
            Dictionary with account details
        """
        try:
            if not MT5_AVAILABLE:
                return {
                    'balance': 10000.0,
                    'equity': 10000.0,
                    'margin': 0.0,
                    'free_margin': 10000.0,
                }

            a = mt5.account_info()
            if not a:
                return {
                    'balance': 10000.0,
                    'equity': 10000.0,
                    'margin': 0.0,
                    'free_margin': 10000.0,
                }

            return {
                'balance': a.balance,
                'equity': a.equity,
                'margin': a.margin,
                'free_margin': a.margin_free,
            }
        except Exception:
            return {'balance': 10000.0, 'equity': 10000.0, 'margin': 0.0, 'free_margin': 10000.0}

    def place_order(self, order_params: dict[str, Any]) -> Any | None:
        """
        Place a new order with retry logic.

        Args:
            order_params: Order parameters dictionary

        Returns:
            MT5 result object or None
        """
        try:
            if not MT5_AVAILABLE:
                return None

            # Retry logic for better reliability
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = mt5.order_send(order_params)
                    if result and hasattr(result, 'retcode'):
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            return result
                        elif result.retcode in [
                            mt5.TRADE_RETCODE_REQUOTE,
                            mt5.TRADE_RETCODE_PRICE_OFF,
                        ]:
                            if attempt < max_retries - 1:
                                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                                continue
                    return result
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    raise

            return None
        except Exception as e:
            self.logger.error(f"Place order error: {e}")
            return None

    def close_position(
        self, position_ticket: int, symbol: str, volume: float, price: float, deviation: int = 20
    ) -> bool:
        """
        Close a position.

        Args:
            position_ticket: Position ticket number
            symbol: Trading symbol
            volume: Volume to close
            price: Close price
            deviation: Price deviation allowance

        Returns:
            True if successful
        """
        try:
            if not MT5_AVAILABLE:
                return False

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": mt5.ORDER_TYPE_BUY,  # Will be adjusted based on position type
                "position": int(position_ticket),
                "price": float(price),
                "deviation": max(1, min(deviation, 100)),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = self.place_order(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Position {position_ticket} closed successfully")
                return True
            else:
                self.logger.error(f"Failed to close position {position_ticket}")
                return False

        except Exception as e:
            self.logger.error(f"Close position error: {e}")
            return False
