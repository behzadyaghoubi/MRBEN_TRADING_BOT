from typing import Tuple
import numpy as np


def _engulf(bars) -> Tuple[int, float]:
    """
    Detect engulfing pattern: +1 bullish, -1 bearish, 0 none.
    
    Args:
        bars: numpy array with shape [n, 4] containing O,H,L,C
        
    Returns:
        Tuple of (direction, score)
    """
    if len(bars) < 2:
        return 0, 0.0
        
    o1, h1, l1, c1 = bars[-2]  # Previous bar
    o2, h2, l2, c2 = bars[-1]  # Current bar
    
    # Bullish engulfing: current green bar engulfs previous red bar
    bull = (c2 > o2 and o1 > c1 and h2 >= h1 and l2 <= l1)
    
    # Bearish engulfing: current red bar engulfs previous green bar
    bear = (c2 < o2 and o1 < c1 and h2 >= h1 and l2 <= l1)
    
    if bull:
        return +1, 0.65
    if bear:
        return -1, 0.65
        
    return 0, 0.0


def _pin(bars) -> Tuple[int, float]:
    """
    Detect pin bar (hammer/shooting star): +1 bullish, -1 bearish, 0 none.
    
    Args:
        bars: numpy array with shape [n, 4] containing O,H,L,C
        
    Returns:
        Tuple of (direction, score)
    """
    if len(bars) < 2:
        return 0, 0.0
        
    o, h, l, c = bars[-1]
    body = abs(c - o)
    rng = h - l
    
    if rng == 0:
        return 0, 0.0
        
    upper = h - max(o, c)  # Upper shadow
    lower = min(o, c) - l  # Lower shadow
    
    # Hammer: long lower shadow, small body
    if lower / rng > 0.6 and body / rng < 0.3:
        return +1, 0.58
        
    # Shooting star: long upper shadow, small body
    if upper / rng > 0.6 and body / rng < 0.3:
        return -1, 0.58
        
    return 0, 0.0


def _inside(bars) -> Tuple[int, float]:
    """
    Detect inside bar pattern: consolidation signal.
    
    Args:
        bars: numpy array with shape [n, 4] containing O,H,L,C
        
    Returns:
        Tuple of (direction, score) - always 0 for inside bars
    """
    if len(bars) < 2:
        return 0, 0.0
        
    ph, pl = bars[-2][1], bars[-2][2]  # Previous high/low
    h, l = bars[-1][1], bars[-1][2]    # Current high/low
    
    # Inside bar: current bar completely inside previous bar
    if h <= ph and l >= pl:
        return 0, 0.52  # Neutral signal, low score
        
    return 0, 0.0


def _sweep(bars) -> Tuple[int, float]:
    """
    Detect sweep pattern: false breakout of recent swing high/low.
    
    Args:
        bars: numpy array with shape [n, 4] containing O,H,L,C
        
    Returns:
        Tuple of (direction, score)
    """
    if len(bars) < 5:
        return 0, 0.0
        
    # Find swing high/low from last 5 bars (excluding current)
    swing_h = max(bars[-5:-1, 1])  # Max high
    swing_l = min(bars[-5:-1, 2])  # Min low
    
    h, l, c = bars[-1][1], bars[-1][2], bars[-1][3]  # Current H,L,C
    
    # Bullish sweep: sweeps low but closes above
    if l < swing_l and c > swing_l:
        return +1, 0.60
        
    # Bearish sweep: sweeps high but closes below
    if h > swing_h and c < swing_h:
        return -1, 0.60
        
    return 0, 0.0


# Pattern registry
PATTERNS = {
    "engulf": _engulf,
    "pin": _pin,
    "inside": _inside,
    "sweep": _sweep,
}


def detect_pa(bars, patterns, min_score) -> Tuple[int, float]:
    """
    Detect price action patterns and calculate ensemble score.
    
    Args:
        bars: numpy array with shape [n, 4] containing O,H,L,C
        patterns: list of pattern names to check
        min_score: minimum score threshold for valid signal
        
    Returns:
        Tuple of (direction, score) where direction is +1/-1/0
    """
    votes = []
    
    # Check each requested pattern
    for name in patterns:
        if name in PATTERNS:
            direction, score = PATTERNS[name](bars)
            if score > 0:  # Only count valid patterns
                votes.append((direction, score))
    
    if not votes:
        return 0, 0.0
    
    # Calculate weighted ensemble
    # Simple approach: average score with direction weighted by score
    total_score = sum(abs(score) for _, score in votes)
    avg_score = total_score / len(votes)
    
    # Weighted direction: sum(direction * score)
    weighted_dir = sum(direction * score for direction, score in votes)
    
    # Final direction: sign of weighted sum
    final_direction = int(np.sign(weighted_dir))
    
    # Check minimum score threshold
    if avg_score < min_score:
        return 0, avg_score
        
    return final_direction, float(avg_score)


def get_pattern_details(bars, patterns) -> dict:
    """
    Get detailed breakdown of all pattern detections.
    
    Args:
        bars: numpy array with shape [n, 4] containing O,H,L,C
        patterns: list of pattern names to check
        
    Returns:
        Dictionary with pattern details and scores
    """
    results = {}
    
    for name in patterns:
        if name in PATTERNS:
            direction, score = PATTERNS[name](bars)
            results[name] = {
                "direction": direction,
                "score": score,
                "detected": score > 0
            }
    
    return results


def validate_bars(bars) -> bool:
    """
    Validate that bars array has correct structure.
    
    Args:
        bars: numpy array to validate
        
    Returns:
        True if valid, False otherwise
    """
    if bars is None or len(bars) == 0:
        return False
        
    if not isinstance(bars, np.ndarray):
        return False
        
    if bars.ndim != 2 or bars.shape[1] != 4:
        return False
        
    # Check for valid OHLC values
    if np.any(bars[:, 1] < bars[:, 2]):  # High < Low
        return False
        
    return True
