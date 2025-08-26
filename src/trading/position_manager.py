"""
Position Management for MR BEN Trading Bot.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import MetaTrader5 as mt5

from ..core.logger import get_logger
from ..core.database import db_manager
from .trade_executor import trade_executor

logger = get_logger("trading.position_manager")


class PositionManager:
    """Manages open positions and position tracking."""
    
    def __init__(self):
        """Initialize the position manager."""
        self.logger = get_logger("position_manager")
        self.positions = {}  # Cache of open positions
        
        self.logger.info("Position Manager initialized")
    
    def initialize(self) -> None:
        """Initialize position manager and load existing positions."""
        try:
            self.refresh_positions()
            self.logger.info(f"Position Manager initialized with {len(self.positions)} positions")
        except Exception as e:
            self.logger.error(f"Error initializing position manager: {e}")
    
    def refresh_positions(self) -> None:
        """Refresh the positions cache from MT5."""
        try:
            mt5_positions = trade_executor.get_positions()
            
            # Update cache
            self.positions = {
                pos['ticket']: pos for pos in mt5_positions
            }
            
            self.logger.debug(f"Refreshed {len(self.positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error refreshing positions: {e}")
    
    def add_position(self, trade_signal) -> None:
        """Add a new position to tracking."""
        try:
            # Refresh positions to get the latest data
            self.refresh_positions()
            
            # Find the newly opened position
            for ticket, position in self.positions.items():
                if (position['symbol'] == trade_signal.symbol and 
                    position['type'] == trade_signal.action and
                    abs(position['price_open'] - trade_signal.entry_price) < 0.001):
                    
                    self.logger.info(f"Added position {ticket} to tracking")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of position dictionaries
        """
        try:
            self.refresh_positions()
            
            if symbol:
                return [
                    pos for pos in self.positions.values()
                    if pos['symbol'] == symbol
                ]
            else:
                return list(self.positions.values())
                
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []
    
    def get_position(self, ticket: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific position by ticket.
        
        Args:
            ticket: Position ticket
            
        Returns:
            Position dictionary or None
        """
        try:
            self.refresh_positions()
            return self.positions.get(ticket)
            
        except Exception as e:
            self.logger.error(f"Error getting position {ticket}: {e}")
            return None
    
    def close_position(self, ticket: int) -> bool:
        """
        Close a specific position.
        
        Args:
            ticket: Position ticket
            
        Returns:
            True if successful
        """
        try:
            position = self.get_position(ticket)
            if not position:
                self.logger.error(f"Position {ticket} not found")
                return False
            
            result = trade_executor.close_position(
                ticket, position['symbol'], position['volume']
            )
            
            if result.success:
                # Update database
                self._update_trade_in_database(ticket, position)
                
                # Remove from cache
                self.positions.pop(ticket, None)
                
                self.logger.info(f"Position {ticket} closed successfully")
                return True
            else:
                self.logger.error(f"Failed to close position {ticket}: {result.comment}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def modify_position(self, ticket: int, stop_loss: float, take_profit: float) -> bool:
        """
        Modify position stop loss and take profit.
        
        Args:
            ticket: Position ticket
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            True if successful
        """
        try:
            position = self.get_position(ticket)
            if not position:
                self.logger.error(f"Position {ticket} not found")
                return False
            
            result = trade_executor.modify_position(ticket, stop_loss, take_profit)
            
            if result.success:
                # Update cache
                self.positions[ticket]['sl'] = stop_loss
                self.positions[ticket]['tp'] = take_profit
                
                self.logger.info(f"Position {ticket} modified successfully")
                return True
            else:
                self.logger.error(f"Failed to modify position {ticket}: {result.comment}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error modifying position {ticket}: {e}")
            return False
    
    def update_position_prices(self) -> None:
        """Update current prices for all positions."""
        try:
            for ticket, position in self.positions.items():
                # Get current tick
                tick = mt5.symbol_info_tick(position['symbol'])
                if tick:
                    if position['type'] == 'BUY':
                        current_price = tick.bid
                    else:
                        current_price = tick.ask
                    
                    # Update position
                    position['price_current'] = current_price
                    
                    # Calculate unrealized profit
                    if position['type'] == 'BUY':
                        position['unrealized_profit'] = (current_price - position['price_open']) * position['volume'] * 100000
                    else:
                        position['unrealized_profit'] = (position['price_open'] - current_price) * position['volume'] * 100000
                        
        except Exception as e:
            self.logger.error(f"Error updating position prices: {e}")
    
    def get_total_exposure(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """
        Get total exposure by symbol.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Dictionary with exposure information
        """
        try:
            positions = self.get_open_positions(symbol)
            
            exposure = {
                'total_volume': 0.0,
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'total_profit': 0.0,
                'position_count': len(positions)
            }
            
            for position in positions:
                exposure['total_volume'] += position['volume']
                exposure['total_profit'] += position.get('unrealized_profit', 0.0)
                
                if position['type'] == 'BUY':
                    exposure['buy_volume'] += position['volume']
                else:
                    exposure['sell_volume'] += position['volume']
            
            return exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating exposure: {e}")
            return {
                'total_volume': 0.0,
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'total_profit': 0.0,
                'position_count': 0
            }
    
    def check_stop_losses(self) -> List[int]:
        """
        Check for positions that have hit stop loss.
        
        Returns:
            List of ticket numbers that hit stop loss
        """
        try:
            self.update_position_prices()
            hit_positions = []
            
            for ticket, position in self.positions.items():
                current_price = position['price_current']
                stop_loss = position['sl']
                
                if stop_loss > 0:
                    if position['type'] == 'BUY' and current_price <= stop_loss:
                        hit_positions.append(ticket)
                    elif position['type'] == 'SELL' and current_price >= stop_loss:
                        hit_positions.append(ticket)
            
            return hit_positions
            
        except Exception as e:
            self.logger.error(f"Error checking stop losses: {e}")
            return []
    
    def check_take_profits(self) -> List[int]:
        """
        Check for positions that have hit take profit.
        
        Returns:
            List of ticket numbers that hit take profit
        """
        try:
            self.update_position_prices()
            hit_positions = []
            
            for ticket, position in self.positions.items():
                current_price = position['price_current']
                take_profit = position['tp']
                
                if take_profit > 0:
                    if position['type'] == 'BUY' and current_price >= take_profit:
                        hit_positions.append(ticket)
                    elif position['type'] == 'SELL' and current_price <= take_profit:
                        hit_positions.append(ticket)
            
            return hit_positions
            
        except Exception as e:
            self.logger.error(f"Error checking take profits: {e}")
            return []
    
    def _update_trade_in_database(self, ticket: int, position: Dict[str, Any]) -> None:
        """Update trade record in database when position is closed."""
        try:
            # Find the trade record
            trades_df = db_manager.get_trades()
            if trades_df.empty:
                return
            
            # Find trade by matching symbol and action
            matching_trades = trades_df[
                (trades_df['symbol'] == position['symbol']) &
                (trades_df['action'] == position['type']) &
                (trades_df['status'] == 'open')
            ]
            
            if not matching_trades.empty:
                # Update the most recent matching trade
                trade_id = matching_trades.iloc[-1]['id']
                
                update_data = {
                    'exit_price': position['price_current'],
                    'profit': position.get('unrealized_profit', 0.0),
                    'status': 'closed'
                }
                
                db_manager.update_trade(trade_id, update_data)
                self.logger.info(f"Updated trade {trade_id} in database")
                
        except Exception as e:
            self.logger.error(f"Error updating trade in database: {e}")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        try:
            self.refresh_positions()
            
            summary = {
                'total_positions': len(self.positions),
                'symbols': {},
                'total_profit': 0.0,
                'total_volume': 0.0
            }
            
            for position in self.positions.values():
                symbol = position['symbol']
                profit = position.get('unrealized_profit', 0.0)
                volume = position['volume']
                
                summary['total_profit'] += profit
                summary['total_volume'] += volume
                
                if symbol not in summary['symbols']:
                    summary['symbols'][symbol] = {
                        'positions': 0,
                        'volume': 0.0,
                        'profit': 0.0
                    }
                
                summary['symbols'][symbol]['positions'] += 1
                summary['symbols'][symbol]['volume'] += volume
                summary['symbols'][symbol]['profit'] += profit
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting position summary: {e}")
            return {
                'total_positions': 0,
                'symbols': {},
                'total_profit': 0.0,
                'total_volume': 0.0
            }


# Global position manager instance
position_manager = PositionManager() 