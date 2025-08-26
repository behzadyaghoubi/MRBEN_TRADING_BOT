"""
Enhanced risk management for MR BEN Trading System.
Handles position sizing, stop loss, take profit, and risk controls.
"""

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal, ROUND_DOWN

# Global MT5 availability flag
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class EnhancedRiskManager:
    """Enhanced risk management with adaptive confidence and trailing stops."""
    
    def __init__(self,
                 base_risk: float = 0.01,
                 min_lot: float = 0.01,
                 max_lot: float = 2.0,
                 max_open_trades: int = 3,
                 max_drawdown: float = 0.10,
                 atr_period: int = 14,
                 sl_atr_multiplier: float = 2.0,
                 tp_atr_multiplier: float = 4.0,
                 trailing_atr_multiplier: float = 1.5,
                 base_confidence_threshold: float = 0.35,
                 adaptive_confidence: bool = True,
                 performance_window: int = 20,
                 confidence_adjustment_factor: float = 0.1,
                 tf_minutes: int = 15):
        """
        Initialize risk manager.
        
        Args:
            base_risk: Base risk per trade (1% = 0.01)
            min_lot: Minimum lot size
            max_lot: Maximum lot size
            max_open_trades: Maximum number of open trades
            max_drawdown: Maximum allowed drawdown
            atr_period: ATR calculation period
            sl_atr_multiplier: Stop loss ATR multiplier
            tp_atr_multiplier: Take profit ATR multiplier
            trailing_atr_multiplier: Trailing stop ATR multiplier
            base_confidence_threshold: Base confidence threshold
            adaptive_confidence: Enable adaptive confidence
            performance_window: Performance evaluation window
            confidence_adjustment_factor: Confidence adjustment factor
            tf_minutes: Timeframe in minutes
        """
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
        self.tf_minutes = tf_minutes
        
        # ATR cache fields
        self._atr_cache = {"value": None, "ts": 0.0}
        
        self.recent_performances: List[float] = []
        self.current_confidence_threshold = base_confidence_threshold
        self.trailing_stops: Dict[int, Dict[str, Any]] = {}

        self.logger = logging.getLogger("EnhancedRiskManager")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
            self.logger.addHandler(h)

    def get_atr(self, symbol: str) -> Optional[float]:
        """
        Get current ATR value for symbol using configured timeframe.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            ATR value or None
        """
        try:
            now = time.time()
            if self._atr_cache["value"] is not None and (now - self._atr_cache["ts"]) < 5.0:
                return self._atr_cache["value"]

            if not MT5_AVAILABLE:
                return None
                
            tf_map = {
                1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15,
                30: mt5.TIMEFRAME_M30, 60: mt5.TIMEFRAME_H1, 240: mt5.TIMEFRAME_H4,
                1440: mt5.TIMEFRAME_D1
            }
            tf = tf_map.get(self.tf_minutes, mt5.TIMEFRAME_M15)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, self.atr_period + 1)
            if rates is None or len(rates) < self.atr_period:
                return None
                
            df = pd.DataFrame(rates)
            hl = df['high'] - df['low']
            hc = (df['high'] - df['close'].shift()).abs()
            lc = (df['low'] - df['close'].shift()).abs()
            tr = np.maximum(hl, np.maximum(hc, lc))
            val = float(tr.rolling(self.atr_period, min_periods=self.atr_period).mean().iloc[-1])
            if pd.isna(val):
                return None
                
            self._atr_cache.update({"value": val, "ts": now})
            return val
        except Exception as e:
            self.logger.error(f"ATR error: {e}")
            return None

    def calculate_dynamic_sl_tp(self, symbol: str, entry_price: float, signal: str) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit based on ATR.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            signal: Trade signal (BUY/SELL)
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        atr = self.get_atr(symbol)
        if atr is None:
            # Fallback distances (instrument dependent; conservative)
            dist_sl = 0.5
            dist_tp = 1.0
        else:
            dist_sl = atr * self.sl_atr_multiplier
            dist_tp = atr * self.tp_atr_multiplier

        if signal == "BUY":
            sl = entry_price - dist_sl
            tp = entry_price + dist_tp
        else:
            sl = entry_price + dist_sl
            tp = entry_price - dist_tp

        self.logger.info(f"Dynamic SL/TP: SL={sl:.2f} TP={tp:.2f} (ATR={atr})")
        return sl, tp

    def calculate_lot_size(self, balance: float, risk_per_trade: float, sl_distance: float, symbol: str) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            balance: Account balance
            risk_per_trade: Risk per trade (0.01 = 1%)
            sl_distance: Stop loss distance in price units
            symbol: Trading symbol
            
        Returns:
            Lot size
        """
        try:
            if not MT5_AVAILABLE:
                return max(self.min_lot, min(0.1, self.max_lot))
                
            info = mt5.symbol_info(symbol)
            if info is None or sl_distance <= 0:
                return self.min_lot
                
            ticks = sl_distance / (info.trade_tick_size or info.point or 0.01)
            if ticks <= 0:
                return self.min_lot
                
            risk_amount = balance * risk_per_trade
            vpt = info.trade_tick_value or 1.0
            raw = risk_amount / (ticks * vpt)
            
            # Volume alignment using Decimal
            step_dec = Decimal(str(info.volume_step or 0.01))
            lot_dec = Decimal(str(max(self.min_lot, min(raw, self.max_lot))))
            lot_adj = (lot_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec
            lot = float(max(info.volume_min, min(float(lot_adj), info.volume_max)))
            
            return float(lot)
        except Exception:
            return self.min_lot

    def can_open_new_trade(self, current_balance: float, start_balance: float, open_trades_count: int) -> bool:
        """
        Check if new trade can be opened.
        
        Args:
            current_balance: Current account balance
            start_balance: Starting account balance
            open_trades_count: Number of currently open trades
            
        Returns:
            True if new trade can be opened
        """
        if open_trades_count >= self.max_open_trades:
            return False
        if start_balance > 0:
            dd = (start_balance - current_balance) / start_balance
            if dd > self.max_drawdown:
                return False
        return True

    def add_trailing_stop(self, ticket: int, entry_price: float, initial_sl: float, is_buy: bool) -> None:
        """
        Add trailing stop for a position.
        
        Args:
            ticket: Position ticket
            entry_price: Entry price
            initial_sl: Initial stop loss
            is_buy: Whether position is long
        """
        self.trailing_stops[ticket] = {
            'entry_price': entry_price,
            'current_sl': initial_sl,
            'highest_price': entry_price if is_buy else float('-inf'),
            'lowest_price': entry_price if not is_buy else float('inf'),
            'is_buy': is_buy
        }

    def remove_trailing_stop(self, ticket: int) -> None:
        """
        Remove trailing stop for a position.
        
        Args:
            ticket: Position ticket
        """
        if ticket in self.trailing_stops:
            del self.trailing_stops[ticket]

    def update_trailing_stops(self, symbol: str) -> List[Dict[str, float]]:
        """
        Update trailing stops for all positions.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of modifications to apply
        """
        mods = []
        atr = self.get_atr(symbol)
        if atr is None:
            return mods
            
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return mods
                
            price = (tick.bid + tick.ask) / 2.0
            
            for ticket, st in self.trailing_stops.items():
                if st['is_buy']:
                    if price > st['highest_price']:
                        st['highest_price'] = price
                    new_sl = st['highest_price'] - (atr * self.trailing_atr_multiplier)
                    if new_sl > st['current_sl']:
                        st['current_sl'] = new_sl
                        mods.append({'ticket': ticket, 'new_sl': new_sl})
                else:
                    if price < st['lowest_price']:
                        st['lowest_price'] = price
                    new_sl = st['lowest_price'] + (atr * self.trailing_atr_multiplier)
                    if new_sl < st['current_sl']:
                        st['current_sl'] = new_sl
                        mods.append({'ticket': ticket, 'new_sl': new_sl})
        except Exception:
            return mods
        
        if mods:
            self.logger.info(f"⛓️ Trailing candidates: {len(mods)}")
        
        return mods

    def get_current_confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        return float(self.current_confidence_threshold)

    def update_performance_from_history(self, symbol: str) -> None:
        """
        Update confidence threshold based on performance history.
        
        Args:
            symbol: Trading symbol
        """
        if not MT5_AVAILABLE:
            return
            
        try:
            start = datetime.combine(datetime.now().date(), datetime.min.time())
            deals = mt5.history_deals_get(start, datetime.now())
            if not deals:
                return
                
            symbol_deals = [d for d in deals if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT]
            if not symbol_deals:
                return
                
            last = max(symbol_deals, key=lambda d: d.time)
            self.recent_performances.append(last.profit)
            
            if len(self.recent_performances) > self.performance_window:
                self.recent_performances.pop(0)
                
            min_closed = max(10, self.performance_window // 2)
            if self.adaptive_confidence and len(self.recent_performances) >= min_closed:
                window = self.recent_performances[-self.performance_window:]
                wins = sum(1 for r in window if r > 0)
                wr = wins / len(window)
                prev = self.current_confidence_threshold
                
                if wr > 0.6:
                    self.current_confidence_threshold = max(
                        self.base_confidence_threshold - self.confidence_adjustment_factor,
                        self.current_confidence_threshold - self.confidence_adjustment_factor
                    )
                elif wr < 0.4:
                    self.current_confidence_threshold = min(
                        self.base_confidence_threshold + self.confidence_adjustment_factor,
                        self.current_confidence_threshold + self.confidence_adjustment_factor
                    )
                    
                if prev != self.current_confidence_threshold:
                    self.logger.info(f"Adaptive conf: {prev:.2f} -> {self.current_confidence_threshold:.2f} (winrate={wr:.2f})")
        except Exception as e:
            self.logger.error(f"Perf update error: {e}")
