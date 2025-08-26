from datetime import datetime
import pytz


def detect_session(ts_utc: datetime) -> str:
    """
    Detect trading session based on UTC time.
    
    Asia: 23:00–07:00 UTC | London: 07:00–13:00 UTC | NY: 13:00–21:00 UTC | Else: off
    
    Args:
        ts_utc: UTC timestamp
        
    Returns:
        Session identifier: "asia", "london", "ny", or "off"
    """
    h = ts_utc.hour
    if 23 <= h or h < 7:
        return "asia"
    if 7 <= h < 13:
        return "london"
    if 13 <= h < 21:
        return "ny"
    return "off"


def get_session_info(ts_utc: datetime) -> dict:
    """
    Get detailed session information including multipliers and status.
    
    Args:
        ts_utc: UTC timestamp
        
    Returns:
        Dictionary with session details
    """
    session = detect_session(ts_utc)
    
    # Session multipliers (from config)
    multipliers = {
        "asia": 0.90,
        "london": 1.05,
        "ny": 1.00,
        "off": 0.80  # Conservative during off-hours
    }
    
    return {
        "session": session,
        "multiplier": multipliers.get(session, 1.0),
        "active": session != "off",
        "hour_utc": ts_utc.hour,
        "timestamp": ts_utc.isoformat()
    }
