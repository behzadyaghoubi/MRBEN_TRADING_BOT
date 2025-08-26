from __future__ import annotations

from datetime import datetime
from typing import Any

from .regime import RegimeDetector
from .sessionx import detect_session, get_session_info


class MarketContext:
    """
    Unified market context combining session and regime detection.

    Provides context-aware multipliers and status for dynamic confidence calculation.
    """

    def __init__(self, config):
        """
        Initialize market context.

        Args:
            config: Configuration object with confidence multipliers
        """
        self.config = config
        self.regime_detector = RegimeDetector()
        self.last_update = None
        self.context_cache = {}

        # Advanced market analyzer disabled for Phase 1 testing
        self.advanced_analyzer = None

    def update_context(self, atr_val: float, ts_utc: datetime = None) -> dict:
        """
        Update market context with new ATR value and timestamp.

        Args:
            atr_val: Current ATR value
            ts_utc: UTC timestamp (defaults to current time)

        Returns:
            Complete market context dictionary
        """
        if ts_utc is None:
            ts_utc = datetime.utcnow()

        # Get session and regime information
        session_info = get_session_info(ts_utc)
        regime_info = self.regime_detector.get_regime_info(atr_val)

        # Build context
        context = {
            "timestamp": ts_utc.isoformat(),
            "session": session_info,
            "regime": regime_info,
            "atr": atr_val,
            "context_id": f"{session_info['session']}_{regime_info['regime']}_{ts_utc.hour}",
        }

        # Cache context
        self.context_cache = context
        self.last_update = ts_utc

        return context

    def get_context_log(self, atr_val: float, ts_utc: datetime = None) -> str:
        """
        Get formatted context log message.

        Args:
            atr_val: Current ATR value
            ts_utc: UTC timestamp

        Returns:
            Formatted log message
        """
        context = self.update_context(atr_val, ts_utc)

        session = context["session"]["session"]
        regime = context["regime"]["regime"]
        atr = context["atr"]

        return f"[CTX] session={session} regime={regime} atr={atr:.4f}"

    def get_dynamic_multipliers(self, atr_val: float, ts_utc: datetime = None) -> dict:
        """
        Get dynamic confidence multipliers based on current context.

        Args:
            atr_val: Current ATR value
            ts_utc: UTC timestamp

        Returns:
            Dictionary with regime and session multipliers
        """
        context = self.update_context(atr_val, ts_utc)

        # Get multipliers from config
        regime_mult = getattr(
            self.config.confidence.dynamic.regime, context["regime"]["regime"].lower(), 1.0
        )
        session_mult = getattr(
            self.config.confidence.dynamic.session, context["session"]["session"], 1.0
        )

        # Add advanced market analysis if available
        advanced_analysis = None
        if self.advanced_analyzer:
            try:
                # Create a mock MarketContext for analysis
                from .typesx import MarketContext

                mock_context = MarketContext(
                    price=1.0,  # Placeholder values
                    bid=0.9999,
                    ask=1.0001,
                    atr_pts=atr_val * 10000,  # Convert to points
                    sma20=1.0,
                    sma50=1.0,
                    session=context["session"]["session"],
                    regime=context["regime"]["regime"],
                    equity=10000.0,
                    balance=10000.0,
                    spread_pts=20.0,
                    open_positions=0,
                )

                advanced_analysis = self.advanced_analyzer.analyze_market_regime(mock_context)
                context["advanced_analysis"] = {
                    "regime": advanced_analysis.regime.value,
                    "confidence": advanced_analysis.confidence,
                    "trend_strength": advanced_analysis.trend_strength,
                    "volatility_level": advanced_analysis.volatility_level,
                }
            except Exception as e:
                print(f"Warning: Advanced market analysis failed: {e}")

        return {
            "regime": regime_mult,
            "session": session_mult,
            "combined": regime_mult * session_mult,
            "context": context,
            "advanced_analysis": advanced_analysis,
        }

    def is_active_session(self, ts_utc: datetime = None) -> bool:
        """
        Check if current time is during an active trading session.

        Args:
            ts_utc: UTC timestamp

        Returns:
            True if session is active
        """
        if ts_utc is None:
            ts_utc = datetime.utcnow()

        session = detect_session(ts_utc)
        return session != "off"

    def get_advanced_market_summary(self) -> dict[str, Any] | None:
        """
        Get advanced market analysis summary if available.

        Returns:
            Advanced market analysis summary or None if not available
        """
        if self.advanced_analyzer:
            try:
                return self.advanced_analyzer.get_market_summary()
            except Exception as e:
                print(f"Warning: Failed to get advanced market summary: {e}")
                return None
        return None
