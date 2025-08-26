import logging
import numpy as np
import MetaTrader5 as mt5
import pandas as pd
from scipy.stats import pearsonr
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

class EnhancedRiskManager:
    """
    Enhanced Risk Manager with:
    - Dynamic ATR-based TP/SL calculation
    - Trailing stop management
    - Adaptive confidence thresholds
    - Position monitoring and management
    """
    
    def __init__(
        self,
        base_risk=0.02,
        min_lot=0.01,
        max_lot=0.1,
        max_open_trades=2,
        max_drawdown=0.10,
        atr_period=14,
        sl_atr_multiplier=2.0,
        tp_atr_multiplier=4.0,  # 1:2 risk-reward ratio
        trailing_atr_multiplier=1.5,
        base_confidence_threshold=0.5,
        adaptive_confidence=True,
        performance_window=20,
        confidence_adjustment_factor=0.1
    ):
        self.base_risk = base_risk
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.max_open_trades = max_open_trades
        self.max_drawdown = max_drawdown
        self.atr_period = atr_period
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.base_confidence_threshold = base_confidence_threshold
        self.adaptive_confidence = adaptive_confidence
        self.performance_window = performance_window
        self.confidence_adjustment_factor = confidence_adjustment_factor
        
        # Performance tracking for adaptive thresholds
        self.recent_performances = []
        self.current_confidence_threshold = base_confidence_threshold
        
        # Trailing stop tracking
        self.trailing_stops = {}  # {ticket: {'entry_price': float, 'current_sl': float, 'is_buy': bool}}
        
        self.logger = logging.getLogger("EnhancedRiskManager")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_atr(self, symbol: str, timeframe=mt5.TIMEFRAME_M5, bars=100) -> Optional[float]:
        """Calculate ATR for dynamic TP/SL"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) < self.atr_period:
            self.logger.error(f"ATR data not available for {symbol}")
            return None
            
        df = pd.DataFrame(rates)
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift())
        df['L-PC'] = abs(df['low'] - df['close'].shift())
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=self.atr_period).mean()
        
        atr = df['ATR'].iloc[-1]
        if pd.isna(atr):
            return None
            
        return atr

    def calculate_dynamic_sl_tp(self, symbol: str, entry_price: float, signal: str) -> Tuple[float, float]:
        """Calculate dynamic Stop Loss and Take Profit based on ATR"""
        atr = self.get_atr(symbol)
        if atr is None:
            # Fallback to fixed values if ATR calculation fails
            self.logger.warning(f"Using fallback SL/TP for {symbol}")
            point = mt5.symbol_info(symbol).point
            sl_distance = 300 * point
            tp_distance = 500 * point
        else:
            # ATR-based calculation
            sl_distance = atr * self.sl_atr_multiplier
            tp_distance = atr * self.tp_atr_multiplier
            
        if signal == "BUY":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # SELL
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
            
        self.logger.info(f"Dynamic SL/TP for {symbol}: SL={sl:.5f}, TP={tp:.5f} (ATR={atr:.5f})")
        return round(sl, 5), round(tp, 5)

    def calculate_position_size(self, symbol: str, balance: float, entry_price: float, 
                              stop_loss: float, pip_value: float) -> float:
        """Calculate position size based on risk management rules"""
        # Calculate stop loss distance in pips
        point = mt5.symbol_info(symbol).point
        sl_distance_pips = abs(entry_price - stop_loss) / point
        
        # Calculate risk amount
        risk_amount = balance * self.base_risk
        
        # Calculate lot size
        lot_size = risk_amount / (sl_distance_pips * pip_value)
        
        # Apply constraints
        lot_size = np.clip(lot_size, self.min_lot, self.max_lot)
        
        self.logger.info(f"Position size for {symbol}: {lot_size:.2f} lots (risk: {self.base_risk:.1%})")
        return round(lot_size, 2)

    def add_trailing_stop(self, ticket: int, entry_price: float, stop_loss: float, is_buy: bool):
        """Add position to trailing stop monitoring"""
        self.trailing_stops[ticket] = {
            'entry_price': entry_price,
            'current_sl': stop_loss,
            'is_buy': is_buy,
            'highest_price': entry_price if is_buy else float('-inf'),
            'lowest_price': entry_price if not is_buy else float('inf')
        }
        self.logger.info(f"Added trailing stop for ticket {ticket}")

    def update_trailing_stops(self, symbol: str) -> List[Dict]:
        """Update all trailing stops and return modifications needed"""
        if not self.trailing_stops:
            return []
            
        atr = self.get_atr(symbol)
        if atr is None:
            return []
            
        current_price = mt5.symbol_info_tick(symbol).bid  # Use bid for trailing calculations
        modifications = []
        
        for ticket, stop_data in self.trailing_stops.items():
            # Update highest/lowest prices
            if stop_data['is_buy']:
                stop_data['highest_price'] = max(stop_data['highest_price'], current_price)
                new_sl = stop_data['highest_price'] - (atr * self.trailing_atr_multiplier)
                
                # Only move SL up for buy positions
                if new_sl > stop_data['current_sl']:
                    stop_data['current_sl'] = new_sl
                    modifications.append({
                        'ticket': ticket,
                        'new_sl': new_sl,
                        'reason': 'Trailing stop updated'
                    })
            else:
                stop_data['lowest_price'] = min(stop_data['lowest_price'], current_price)
                new_sl = stop_data['lowest_price'] + (atr * self.trailing_atr_multiplier)
                
                # Only move SL down for sell positions
                if new_sl < stop_data['current_sl']:
                    stop_data['current_sl'] = new_sl
                    modifications.append({
                        'ticket': ticket,
                        'new_sl': new_sl,
                        'reason': 'Trailing stop updated'
                    })
        
        return modifications

    def remove_trailing_stop(self, ticket: int):
        """Remove position from trailing stop monitoring"""
        if ticket in self.trailing_stops:
            del self.trailing_stops[ticket]
            self.logger.info(f"Removed trailing stop for ticket {ticket}")

    def update_performance(self, trade_result: float):
        """Update performance tracking for adaptive confidence thresholds"""
        self.recent_performances.append(trade_result)
        
        # Keep only recent performances
        if len(self.recent_performances) > self.performance_window:
            self.recent_performances.pop(0)
            
        # Update confidence threshold if adaptive mode is enabled
        if self.adaptive_confidence and len(self.recent_performances) >= 5:
            recent_win_rate = sum(1 for p in self.recent_performances if p > 0) / len(self.recent_performances)
            
            if recent_win_rate > 0.6:  # Good performance
                self.current_confidence_threshold = max(
                    self.base_confidence_threshold * 0.8,
                    self.current_confidence_threshold - self.confidence_adjustment_factor
                )
            elif recent_win_rate < 0.4:  # Poor performance
                self.current_confidence_threshold = min(
                    self.base_confidence_threshold * 1.2,
                    self.current_confidence_threshold + self.confidence_adjustment_factor
                )
                
            self.logger.info(f"Adaptive confidence threshold: {self.current_confidence_threshold:.3f} "
                           f"(win rate: {recent_win_rate:.2f})")

    def get_current_confidence_threshold(self) -> float:
        """Get current confidence threshold (adaptive or fixed)"""
        if self.adaptive_confidence:
            return self.current_confidence_threshold
        return self.base_confidence_threshold

    def can_open_new_trade(self, balance: float, start_balance: float, open_trades: int) -> bool:
        """Check if new trade can be opened"""
        # Check drawdown
        drawdown = 1 - (balance / start_balance)
        if drawdown > self.max_drawdown:
            self.logger.warning(f"Max drawdown exceeded: {drawdown:.2%}")
            return False
            
        # Check max open trades
        if open_trades >= self.max_open_trades:
            self.logger.warning(f"Max open trades reached: {open_trades}")
            return False
            
        return True

    def get_open_positions(self, symbol: str) -> List[Dict]:
        """Get all open positions for a symbol"""
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
            
        return [
            {
                'ticket': pos.ticket,
                'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit
            }
            for pos in positions
        ]

    def log_status(self):
        """Log current risk manager status"""
        self.logger.info(f"Risk Manager Status:")
        self.logger.info(f"  - Confidence threshold: {self.get_current_confidence_threshold():.3f}")
        self.logger.info(f"  - Trailing stops active: {len(self.trailing_stops)}")
        self.logger.info(f"  - Recent performance: {self.recent_performances[-5:] if self.recent_performances else 'None'}") 