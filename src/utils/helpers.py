"""
Helper functions for MR BEN Trading System.
"""

import logging
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import Tuple, Optional, Dict, Any

# Global MT5 availability flag
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


def round_price(symbol: str, price: float) -> float:
    """
    Round price to symbol's digits/point using MT5 symbol info if available.
    
    Args:
        symbol: Trading symbol
        price: Price to round
        
    Returns:
        Rounded price
    """
    from .error_handler import error_handler
    
    with error_handler(logging.getLogger("Helpers"), "round_price", price):
        if MT5_AVAILABLE:
            info = mt5.symbol_info(symbol)
            if info and info.point:
                step = Decimal(str(info.point))
                q = (Decimal(str(price)) / step).to_integral_value(rounding=ROUND_HALF_UP) * step
                return float(q)
        return float(Decimal(str(price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def enforce_min_distance_and_round(symbol: str, entry: float, sl: float, tp: float, is_buy: bool) -> Tuple[float, float]:
    """
    Ensure SL/TP respect broker min distance (trade_stops_level & trade_freeze_level) and then round.
    
    Args:
        symbol: Trading symbol
        entry: Entry price
        sl: Stop loss price
        tp: Take profit price
        is_buy: Whether this is a buy order
        
    Returns:
        Tuple of (adjusted_sl, adjusted_tp)
    """
    from .error_handler import error_handler
    
    with error_handler(logging.getLogger("Helpers"), "enforce_min_distance", (sl, tp)):
        if MT5_AVAILABLE:
            info = mt5.symbol_info(symbol)
        else:
            info = None
            
        if not info:
            return round_price(symbol, sl), round_price(symbol, tp)

        point = info.point or 0.01
        stops_pts = float(getattr(info, 'trade_stops_level', 0) or 0)
        freeze_pts = float(getattr(info, 'trade_freeze_level', 0) or 0)
        min_dist = max(stops_pts, freeze_pts) * float(point)

        # Validate input parameters
        if not all(isinstance(x, (int, float)) for x in [entry, sl, tp]):
            raise ValueError(f"Invalid price values: entry={entry}, sl={sl}, tp={tp}")

        if is_buy:
            if (entry - sl) < min_dist:
                sl = entry - min_dist
            if (tp - entry) < min_dist:
                tp = entry + min_dist
        else:
            if (sl - entry) < min_dist:
                sl = entry + min_dist
            if (entry - tp) < min_dist:
                tp = entry - min_dist

        return round_price(symbol, sl), round_price(symbol, tp)


def is_spread_ok(symbol: str, max_spread_points: int) -> Tuple[bool, float, float]:
    """
    Check if current spread (in points) is below threshold.
    
    Args:
        symbol: Trading symbol
        max_spread_points: Maximum allowed spread in points
        
    Returns:
        Tuple of (is_ok, spread_points, threshold_points)
    """
    from .error_handler import error_handler
    
    with error_handler(logging.getLogger("Helpers"), "spread_check", (True, 0.0, float(max_spread_points))):
        if not MT5_AVAILABLE:
            return True, 0.0, float(max_spread_points)
            
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        
        if not info or not tick:
            return True, 0.0, float(max_spread_points)
            
        point = info.point or 0.01
        spread_price = (tick.ask - tick.bid)
        
        # Validate spread data
        if spread_price < 0 or not np.isfinite(spread_price):
            return False, float('inf'), float(max_spread_points)
            
        spread_points = spread_price / point
        return (spread_points <= max_spread_points), float(spread_points), float(max_spread_points)


def _rolling_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate ATR with fallback and NaN protection.
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period
        
    Returns:
        ATR value
    """
    from .error_handler import error_handler
    
    with error_handler(logging.getLogger("Helpers"), "rolling_atr", 0.5):
        if df is None or len(df) < period:
            return 0.5
            
        # Validate DataFrame structure
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return 0.5
            
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = np.maximum(hl, np.maximum(hc, lc))
        
        # Use pandas rolling with better error handling
        atr = tr.rolling(period, min_periods=period).mean().iloc[-1]
        result = float(atr) if pd.notna(atr) and np.isfinite(atr) else 0.5
        
        # Sanity check
        if result <= 0 or result > 1000:  # Unrealistic ATR values
            return 0.5
            
        return result


def _swing_extrema(df: pd.DataFrame, bars: int = 10) -> Tuple[float, float]:
    """
    Calculate swing high/low in N recent bars.
    
    Args:
        df: DataFrame with OHLC data
        bars: Number of bars to look back
        
    Returns:
        Tuple of (swing_low, swing_high)
    """
    from .error_handler import error_handler
    
    with error_handler(logging.getLogger("Helpers"), "swing_extrema", (0.0, 0.0)):
        if df is None or len(df) < 2:
            return 0.0, 0.0
            
        bars = max(2, min(bars, len(df)))
        
        # Validate DataFrame structure
        if not all(col in df.columns for col in ['high', 'low']):
            return 0.0, 0.0
            
        lo = float(df['low'].iloc[-bars:].min())
        hi = float(df['high'].iloc[-bars:].max())
        
        # Sanity checks
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return 0.0, 0.0
        if lo > hi:  # Invalid data
            return 0.0, 0.0
            
        return lo, hi


def _apply_soft_gate(p_value: float,
                     base_thr: float,
                     base_consec: int,
                     *,
                     low_cut: float = 0.20,
                     med_cut: float = 0.10,
                     very_low_cut: float = 0.05,
                     bump_low: float = 0.00,
                     bump_med: float = 0.02,
                     bump_vlow: float = 0.03,
                     max_conf_bump: float = 0.03,
                     add_consec_med: int = 0,
                     add_consec_vlow: int = 1,
                     high_conf_override_margin: float = 0.02) -> Tuple[float, int, float]:
    """
    Apply soft conformal gate with penalties.
    
    Returns:
        Tuple of (adjusted_threshold, required_consecutive, override_margin)
    """
    # Default values
    conf_bump = bump_low
    add_consec = 0

    if p_value >= low_cut:
        conf_bump = 0.0
        add_consec = 0
    elif med_cut <= p_value < low_cut:
        conf_bump = bump_med
        add_consec = add_consec_med
    elif very_low_cut <= p_value < med_cut:
        conf_bump = bump_vlow
        add_consec = add_consec_vlow
    else:
        # Very low: slightly stricter, but capped
        conf_bump = min(max_conf_bump, bump_vlow + 0.005)
        add_consec = add_consec_vlow

    conf_bump = min(conf_bump, max_conf_bump)
    adj_thr = base_thr + conf_bump
    req_consec = max(1, base_consec + add_consec)  # Never < 1 if base_consec=1

    return adj_thr, req_consec, high_conf_override_margin
