#!/usr/bin/env python3
"""
MR BEN - Context Factory for A/B Testing
Creates MarketContext objects from bar/tick data
"""

from __future__ import annotations

from typing import Any

from .loggingx import logger
from .regime import RegimeDetector
from .sessionx import detect_session
from .typesx import MarketContext


class ContextFactory:
    """Factory for creating MarketContext objects"""

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.session_cache = {}
        self.regime_cache = {}

        logger.bind(evt="CONTEXT").info("context_factory_initialized")

    def create_from_bar(self, bar_data: dict[str, Any]) -> MarketContext:
        """Create MarketContext from bar data"""
        try:
            # Extract basic price data
            price = bar_data.get('close', 0.0)
            bid = bar_data.get('bid', price)
            ask = bar_data.get('ask', price)
            atr_pts = bar_data.get('atr_pts', 0.0)

            # Extract indicators
            sma20 = bar_data.get('sma20', price)
            sma50 = bar_data.get('sma50', price)

            # Get session and regime
            timestamp = bar_data.get('timestamp')
            session = self._get_session(timestamp)
            regime = self._get_regime(atr_pts)

            # Extract account data
            equity = bar_data.get('equity', 10000.0)
            balance = bar_data.get('balance', 10000.0)
            spread_pts = bar_data.get('spread_pts', 20.0)
            open_positions = bar_data.get('open_positions', 0)

            # Create context
            context = MarketContext(
                price=price,
                bid=bid,
                ask=ask,
                atr_pts=atr_pts,
                sma20=sma20,
                sma50=sma50,
                session=session,
                regime=regime,
                equity=equity,
                balance=balance,
                spread_pts=spread_pts,
                open_positions=open_positions,
            )

            logger.bind(evt="CONTEXT").debug(
                "context_created_from_bar", price=price, session=session, regime=regime
            )

            return context

        except Exception as e:
            logger.bind(evt="CONTEXT").error(
                "context_creation_error", error=str(e), bar_data=bar_data
            )
            # Return default context
            return self._create_default_context()

    def create_from_tick(self, tick_data: dict[str, Any]) -> MarketContext:
        """Create MarketContext from tick data"""
        try:
            # Extract tick data
            bid = tick_data.get('bid', 0.0)
            ask = tick_data.get('ask', 0.0)
            price = (bid + ask) / 2.0
            atr_pts = tick_data.get('atr_pts', 0.0)

            # Use cached or default values for indicators
            sma20 = tick_data.get('sma20', price)
            sma50 = tick_data.get('sma50', price)

            # Get session and regime
            timestamp = tick_data.get('timestamp')
            session = self._get_session(timestamp)
            regime = self._get_regime(atr_pts)

            # Extract account data
            equity = tick_data.get('equity', 10000.0)
            balance = tick_data.get('balance', 10000.0)
            spread_pts = tick_data.get('spread_pts', ask - bid)
            open_positions = tick_data.get('open_positions', 0)

            # Create context
            context = MarketContext(
                price=price,
                bid=bid,
                ask=ask,
                atr_pts=atr_pts,
                sma20=sma20,
                sma50=sma50,
                session=session,
                regime=regime,
                equity=equity,
                balance=balance,
                spread_pts=spread_pts,
                open_positions=open_positions,
            )

            return context

        except Exception as e:
            logger.bind(evt="CONTEXT").error(
                "tick_context_creation_error", error=str(e), tick_data=tick_data
            )
            # Return default context
            return self._create_default_context()

    def _get_session(self, timestamp) -> str:
        """Get trading session for timestamp"""
        if timestamp is None:
            return "off"

        # Use cached session if available
        if timestamp in self.session_cache:
            return self.session_cache[timestamp]

        # Detect session
        try:
            from datetime import datetime

            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp

            session = detect_session(dt)
            self.session_cache[timestamp] = session
            return session

        except Exception as e:
            logger.bind(evt="CONTEXT").warning(
                "session_detection_error", error=str(e), timestamp=timestamp
            )
            return "off"

    def _get_regime(self, atr_pts: float) -> str:
        """Get market regime based on ATR"""
        try:
            regime = self.regime_detector.update(atr_pts)
            return regime
        except Exception as e:
            logger.bind(evt="CONTEXT").warning(
                "regime_detection_error", error=str(e), atr_pts=atr_pts
            )
            return "UNKNOWN"

    def _create_default_context(self) -> MarketContext:
        """Create a default context when data is missing"""
        return MarketContext(
            price=1.1000,
            bid=1.0999,
            ask=1.1001,
            atr_pts=20.0,
            sma20=1.1000,
            sma50=1.1000,
            session="off",
            regime="UNKNOWN",
            equity=10000.0,
            balance=10000.0,
            spread_pts=20.0,
            open_positions=0,
        )

    def clear_cache(self) -> None:
        """Clear cached session and regime data"""
        self.session_cache.clear()
        self.regime_cache.clear()

        logger.bind(evt="CONTEXT").info("context_cache_cleared")

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics"""
        return {
            "session_cache_size": len(self.session_cache),
            "regime_cache_size": len(self.regime_cache),
        }
