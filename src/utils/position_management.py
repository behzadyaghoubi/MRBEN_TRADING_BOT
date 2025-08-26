"""
Position management utilities for MR BEN Trading System.
"""

import logging
from typing import Dict, Any, Optional, List
from .error_handler import error_handler

# Global MT5 availability flag
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


def _get_open_positions(symbol: str, magic: int, trailing_registry: Optional[Dict] = None) -> Dict[int, Any]:
    """
    Fetch open positions for this symbol & magic as dict[ticket]=position.
    
    Args:
        symbol: Trading symbol
        magic: Magic number for filtering
        trailing_registry: Optional trailing registry for cleanup
        
    Returns:
        Dictionary of open positions by ticket
    """
    with error_handler(logging.getLogger("PositionMgmt"), "get_open_positions", {}):
        if not MT5_AVAILABLE:
            return {}
            
        # Validate inputs
        if not symbol or not isinstance(magic, int):
            return {}
            
        pos = mt5.positions_get(symbol=symbol)
        if pos is None:
            return {}
            
        by_ticket = {}
        for p in pos:
            try:
                # Validate position object
                if not hasattr(p, 'ticket') or not hasattr(p, 'magic'):
                    continue
                    
                # Magic filtering with fallback
                if getattr(p, 'magic', 0) == magic or magic is None:
                    # Additional validation
                    if hasattr(p, 'price_open') and hasattr(p, 'type'):
                        by_ticket[p.ticket] = p
            except Exception as e:
                logging.getLogger("PositionMgmt").warning(f"Invalid position object: {e}")
                continue
        
        # Clean up trailing registry if provided
        if trailing_registry is not None:
            for t in list(trailing_registry.keys()):
                if t not in by_ticket:
                    trailing_registry.pop(t, None)
                
        return by_ticket


def _modify_position_sltp(position_ticket: int, symbol: str, new_sl: Optional[float] = None, 
                          new_tp: Optional[float] = None, magic: Optional[int] = None, 
                          deviation: int = 20) -> Optional[Any]:
    """
    Safely modify SL/TP of an open position by ticket.
    
    Args:
        position_ticket: Position ticket number
        symbol: Trading symbol
        new_sl: New stop loss price
        new_tp: New take profit price
        magic: Magic number
        deviation: Price deviation allowance
        
    Returns:
        MT5 result object or None
    """
    with error_handler(logging.getLogger("PositionMgmt"), "modify_position_sltp", None):
        # Input validation
        if not isinstance(position_ticket, int) or position_ticket <= 0:
            raise ValueError(f"Invalid position ticket: {position_ticket}")
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if new_sl is not None and not (isinstance(new_sl, (int, float)) and new_sl > 0):
            raise ValueError(f"Invalid SL value: {new_sl}")
        if new_tp is not None and not (isinstance(new_tp, (int, float)) and new_tp > 0):
            raise ValueError(f"Invalid TP value: {new_tp}")
            
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position_ticket,
            "deviation": max(1, min(deviation, 100)),  # Clamp deviation
        }
        
        if new_sl is not None:
            request["sl"] = float(new_sl)
        if new_tp is not None:
            request["tp"] = float(new_tp)
        if magic is not None:
            request["magic"] = int(magic)

        # Retry logic for better reliability
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = mt5.order_send(request)
                if result and hasattr(result, 'retcode'):
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        return result
                    elif result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_OFF]:
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                            continue
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.1 * (attempt + 1))
                    continue
                raise
                
        return None


def _prune_trailing_registry(trailing_registry: Dict, open_pos_dict: Dict, logger: logging.Logger) -> int:
    """
    Remove stale tickets not present in current open positions.
    
    Args:
        trailing_registry: Trailing registry dictionary
        open_pos_dict: Current open positions dictionary
        logger: Logger instance
        
    Returns:
        Number of pruned tickets
    """
    with error_handler(logger, "prune_trailing_registry", 0):
        if not isinstance(trailing_registry, dict) or not isinstance(open_pos_dict, dict):
            return 0
            
        stale = [t for t in list(trailing_registry.keys()) if t not in open_pos_dict]
        
        for t in stale:
            try:
                trailing_registry.pop(t, None)
            except Exception as e:
                logger.warning(f"Failed to remove stale ticket {t}: {e}")
                
        if stale:
            logger.info("🧹 Trailing prune: removed %d stale tickets: %s", len(stale), stale[:10])
            
        return len(stale)


def _count_open_positions(symbol: str, magic: int) -> int:
    """
    Count open positions with error handling.
    
    Args:
        symbol: Trading symbol
        magic: Magic number
        
    Returns:
        Number of open positions
    """
    with error_handler(logging.getLogger("PositionMgmt"), "count_open_positions", 0):
        return len(_get_open_positions(symbol, magic))


def validate_position_data(position: Any) -> bool:
    """
    Validate position object has required attributes.
    
    Args:
        position: Position object to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_attrs = ['ticket', 'type', 'price_open', 'sl', 'tp', 'volume']
    return all(hasattr(position, attr) for attr in required_attrs)


def get_position_summary(positions: Dict[int, Any]) -> Dict[str, Any]:
    """
    Get summary statistics for positions.
    
    Args:
        positions: Dictionary of positions
        
    Returns:
        Summary statistics dictionary
    """
    if not positions:
        return {"count": 0, "total_volume": 0.0, "types": {}}
        
    total_volume = sum(getattr(p, 'volume', 0) for p in positions.values())
    types = {}
    for p in positions.values():
        pos_type = getattr(p, 'type', 'unknown')
        types[pos_type] = types.get(pos_type, 0) + 1
        
    return {
        "count": len(positions),
        "total_volume": total_volume,
        "types": types
    }
