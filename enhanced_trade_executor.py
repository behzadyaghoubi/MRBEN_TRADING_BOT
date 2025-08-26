import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import json
import os
from typing import Optional, Tuple, Dict, Any
from enhanced_risk_manager import EnhancedRiskManager

class EnhancedTradeExecutor:
    """
    Enhanced Trade Executor with:
    - Dynamic ATR-based TP/SL calculation
    - Integration with Enhanced Risk Manager
    - Position size calculation
    - Trailing stop management
    """
    
    def __init__(self, risk_manager: EnhancedRiskManager):
        self.risk_manager = risk_manager
        self.DEVIATION = 10
        self.MAGIC = 20250615
        self.LOG_FILE = "enhanced_live_trades_log.csv"
        
        # Initialize MT5 if not already done
        if not mt5.initialize():
            print("❌ Failed to initialize MT5")
            raise Exception("MT5 initialization failed")

    def get_account_info(self) -> Dict[str, float]:
        """Get account information"""
        account_info = mt5.account_info()
        if account_info is None:
            raise Exception("Failed to get account info")
            
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free
        }

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise Exception(f"Symbol {symbol} not found")
            
        return {
            'point': symbol_info.point,
            'digits': symbol_info.digits,
            'trade_mode': symbol_info.trade_mode,
            'trade_contract_size': symbol_info.trade_contract_size,
            'trade_tick_value': symbol_info.trade_tick_value,
            'trade_tick_size': symbol_info.trade_tick_size
        }

    def calculate_pip_value(self, symbol: str, lot_size: float) -> float:
        """Calculate pip value for position sizing"""
        symbol_info = self.get_symbol_info(symbol)
        tick_value = symbol_info['trade_tick_value']
        tick_size = symbol_info['trade_tick_size']
        point = symbol_info['point']
        
        # Calculate pip value
        pip_value = (tick_value * lot_size) / (tick_size / point)
        return pip_value

    def send_order(self, symbol: str, signal: str, volume: Optional[float] = None) -> Optional[int]:
        """
        Send order with dynamic TP/SL calculation
        
        Args:
            symbol: Trading symbol
            signal: "BUY" or "SELL"
            volume: Lot size (if None, will be calculated automatically)
            
        Returns:
            Ticket number if successful, None otherwise
        """
        try:
            # Get current market prices
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print(f"❌ Failed to get tick data for {symbol}")
                return None
                
            ask = tick.ask
            bid = tick.bid
            entry_price = ask if signal == "BUY" else bid
            
            # Calculate dynamic SL/TP
            sl, tp = self.risk_manager.calculate_dynamic_sl_tp(symbol, entry_price, signal)
            
            # Calculate position size if not provided
            if volume is None:
                account_info = self.get_account_info()
                balance = account_info['balance']
                
                # Get current open positions count
                open_positions = self.risk_manager.get_open_positions(symbol)
                open_trades_count = len(open_positions)
                
                # Check if we can open new trade
                if not self.risk_manager.can_open_new_trade(balance, balance, open_trades_count):
                    print(f"❌ Cannot open new trade for {symbol}")
                    return None
                
                # Calculate pip value for position sizing
                pip_value = self.calculate_pip_value(symbol, 0.01)  # Use 0.01 lot for calculation
                volume = self.risk_manager.calculate_position_size(symbol, balance, entry_price, sl, pip_value)
                
                if volume == 0:
                    print(f"❌ Calculated volume is 0 for {symbol}")
                    return None
            
            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": entry_price,
                "sl": sl,
                "tp": tp,
                "deviation": self.DEVIATION,
                "magic": self.MAGIC,
                "comment": "MRBEN Enhanced Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            # Log the trade
            self._log_trade(symbol, signal, entry_price, sl, tp, volume, result)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Order executed successfully: {signal} {symbol} at {entry_price}")
                print(f"   SL: {sl}, TP: {tp}, Volume: {volume}")
                
                # Add to trailing stop monitoring
                self.risk_manager.add_trailing_stop(result.order, entry_price, sl, signal == "BUY")
                
                return result.order
            else:
                print(f"❌ Order failed: {result.retcode} - {result.comment}")
                return None
                
        except Exception as e:
            print(f"❌ Error sending order: {e}")
            return None

    def modify_stop_loss(self, ticket: int, new_sl: float) -> bool:
        """Modify stop loss for existing position"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Stop loss modified for ticket {ticket}: {new_sl}")
                return True
            else:
                print(f"❌ Failed to modify SL for ticket {ticket}: {result.retcode}")
                return False
                
        except Exception as e:
            print(f"❌ Error modifying stop loss: {e}")
            return False

    def close_position(self, ticket: int) -> bool:
        """Close specific position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                print(f"❌ Position {ticket} not found")
                return False
                
            pos = position[0]
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": self.DEVIATION,
                "magic": self.MAGIC,
                "comment": "MRBEN Close Position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Position {ticket} closed successfully")
                # Remove from trailing stop monitoring
                self.risk_manager.remove_trailing_stop(ticket)
                return True
            else:
                print(f"❌ Failed to close position {ticket}: {result.retcode}")
                return False
                
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return False

    def update_trailing_stops(self, symbol: str) -> int:
        """Update all trailing stops for a symbol"""
        modifications = self.risk_manager.update_trailing_stops(symbol)
        
        updated_count = 0
        for mod in modifications:
            if self.modify_stop_loss(mod['ticket'], mod['new_sl']):
                updated_count += 1
                
        if updated_count > 0:
            print(f"🔄 Updated {updated_count} trailing stops for {symbol}")
            
        return updated_count

    def _log_trade(self, symbol: str, signal: str, entry_price: float, sl: float, tp: float, 
                   volume: float, result: Any):
        """Log trade details"""
        log = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": signal,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "volume": volume,
            "result_code": result.retcode,
            "comment": result.comment,
            "ticket": result.order if result.retcode == mt5.TRADE_RETCODE_DONE else None
        }
        
        # Save to CSV
        if os.path.exists(self.LOG_FILE):
            pd.DataFrame([log]).to_csv(self.LOG_FILE, mode='a', header=False, index=False)
        else:
            pd.DataFrame([log]).to_csv(self.LOG_FILE, index=False)

    def get_trade_history(self, symbol: str = None, days: int = 30) -> pd.DataFrame:
        """Get trade history"""
        from_date = datetime.now() - pd.Timedelta(days=days)
        
        if symbol:
            deals = mt5.history_deals_get(from_date, datetime.now(), symbol=symbol)
        else:
            deals = mt5.history_deals_get(from_date, datetime.now())
            
        if deals is None:
            return pd.DataFrame()
            
        history = []
        for deal in deals:
            history.append({
                'ticket': deal.ticket,
                'time': datetime.fromtimestamp(deal.time),
                'symbol': deal.symbol,
                'type': 'BUY' if deal.type == mt5.DEAL_TYPE_BUY else 'SELL',
                'volume': deal.volume,
                'price': deal.price,
                'profit': deal.profit,
                'magic': deal.magic
            })
            
        return pd.DataFrame(history)

    def shutdown(self):
        """Shutdown MT5 connection"""
        mt5.shutdown() 