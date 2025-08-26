"""
Exception classes for MR BEN Trading System.
"""


class TradingSystemError(Exception):
    """Base exception for trading system errors."""
    pass


class MT5ConnectionError(TradingSystemError):
    """MT5 connection related errors."""
    pass


class DataError(TradingSystemError):
    """Data related errors."""
    pass


class RiskError(TradingSystemError):
    """Risk management errors."""
    pass
