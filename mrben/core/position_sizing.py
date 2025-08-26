#!/usr/bin/env python3
"""
MR BEN - Position Sizing Algorithms
Dynamic position sizing based on risk, confidence, and market conditions
"""

from __future__ import annotations

from dataclasses import dataclass

from .loggingx import logger


@dataclass
class PositionSizeInfo:
    """Information about calculated position size"""

    size: float
    risk_amount: float
    stop_loss_distance: float
    confidence_multiplier: float
    volatility_multiplier: float
    regime_multiplier: float
    session_multiplier: float
    final_multiplier: float
    method: str
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class PositionSizer:
    """
    Dynamic position sizing based on multiple factors:
    - Fixed risk percentage per trade
    - Dynamic confidence adjustments
    - Market volatility adaptation
    - Session-based adjustments
    - Account balance scaling
    """

    def __init__(self, config):
        self.config = config
        self.base_risk_pct = config.risk.base_r_pct
        self.min_lot = config.risk.min_lot
        self.max_lot = config.risk.max_lot

        # Risk scaling factors
        self.confidence_scaling = getattr(config.confidence.dynamic, 'position_scaling', True)
        self.volatility_scaling = getattr(config.atr, 'position_scaling', True)
        self.session_scaling = getattr(config.session, 'position_scaling', True)

        # Safety limits
        self.max_risk_per_trade = 0.05  # 5% maximum risk per trade
        self.min_risk_per_trade = 0.001  # 0.1% minimum risk per trade

        self.logger = logger

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_balance: float,
        confidence: float,
        atr_value: float,
        regime: str = "NORMAL",
        session: str = "london",
        direction: int = 1,
    ) -> PositionSizeInfo:
        """
        Calculate optimal position size based on multiple factors

        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            account_balance: Current account balance
            confidence: Signal confidence (0.0 - 1.0)
            atr_value: Current ATR value
            regime: Market volatility regime
            session: Trading session
            direction: Trade direction (+1/-1)

        Returns:
            PositionSizeInfo with calculated size and details
        """
        warnings = []

        # Calculate stop loss distance
        stop_loss_distance = abs(entry_price - stop_loss_price)

        if stop_loss_distance <= 0:
            return PositionSizeInfo(
                size=0.0,
                risk_amount=0.0,
                stop_loss_distance=0.0,
                confidence_multiplier=0.0,
                volatility_multiplier=0.0,
                regime_multiplier=0.0,
                session_multiplier=0.0,
                final_multiplier=0.0,
                method="error",
                warnings=["Invalid stop loss distance"],
            )

        # Base risk amount
        base_risk_amount = account_balance * (self.base_risk_pct / 100)

        # Calculate multipliers
        confidence_mult = self._calculate_confidence_multiplier(confidence)
        volatility_mult = self._calculate_volatility_multiplier(atr_value, entry_price)
        regime_mult = self._calculate_regime_multiplier(regime)
        session_mult = self._calculate_session_multiplier(session)

        # Combined multiplier
        final_multiplier = confidence_mult * volatility_mult * regime_mult * session_mult

        # Apply safety bounds
        final_multiplier = max(0.1, min(3.0, final_multiplier))

        # Calculate adjusted risk amount
        adjusted_risk_amount = base_risk_amount * final_multiplier

        # Ensure risk limits
        max_risk_amount = account_balance * self.max_risk_per_trade
        min_risk_amount = account_balance * self.min_risk_per_trade

        if adjusted_risk_amount > max_risk_amount:
            adjusted_risk_amount = max_risk_amount
            warnings.append(f"Risk capped at {self.max_risk_per_trade*100:.1f}%")

        if adjusted_risk_amount < min_risk_amount:
            adjusted_risk_amount = min_risk_amount
            warnings.append(f"Risk floored at {self.min_risk_per_trade*100:.3f}%")

        # Calculate position size
        position_size = adjusted_risk_amount / stop_loss_distance

        # Convert to lots for forex (assuming standard lot = 100,000 units)
        if self._is_forex_symbol(symbol):
            position_size = position_size / 100000

        # Apply lot size limits
        if position_size > self.max_lot:
            position_size = self.max_lot
            warnings.append(f"Position size capped at {self.max_lot} lots")

        if position_size < self.min_lot:
            position_size = self.min_lot
            warnings.append(f"Position size floored at {self.min_lot} lots")

        # Log calculation
        self.logger.bind(evt="POSITION").info(
            "position_size_calculated",
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            size=position_size,
            risk_amount=adjusted_risk_amount,
            confidence=confidence,
            multipliers={
                "confidence": confidence_mult,
                "volatility": volatility_mult,
                "regime": regime_mult,
                "session": session_mult,
                "final": final_multiplier,
            },
        )

        return PositionSizeInfo(
            size=position_size,
            risk_amount=adjusted_risk_amount,
            stop_loss_distance=stop_loss_distance,
            confidence_multiplier=confidence_mult,
            volatility_multiplier=volatility_mult,
            regime_multiplier=regime_mult,
            session_multiplier=session_mult,
            final_multiplier=final_multiplier,
            method="dynamic_risk_based",
            warnings=warnings,
        )

    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Calculate position size multiplier based on signal confidence"""
        if not self.confidence_scaling:
            return 1.0

        # Higher confidence = larger position
        # Confidence range: 0.5-1.0 -> Multiplier range: 0.5-1.5
        normalized_confidence = max(0.0, min(1.0, confidence))

        if normalized_confidence < 0.5:
            # Very low confidence - reduce position significantly
            return 0.3
        else:
            # Scale from 0.5 to 1.5 based on confidence above 0.5
            return 0.5 + (normalized_confidence - 0.5) * 2.0

    def _calculate_volatility_multiplier(self, atr_value: float, price: float) -> float:
        """Calculate position size multiplier based on volatility"""
        if not self.volatility_scaling:
            return 1.0

        # Calculate ATR as percentage of price
        atr_percent = (atr_value / price) * 100

        # High volatility = smaller position
        # ATR% ranges and multipliers:
        # < 0.5%: 1.2x (low volatility, increase size)
        # 0.5-1.0%: 1.0x (normal volatility)
        # 1.0-2.0%: 0.8x (high volatility, reduce size)
        # > 2.0%: 0.6x (very high volatility, reduce significantly)

        if atr_percent < 0.5:
            return 1.2
        elif atr_percent < 1.0:
            return 1.0
        elif atr_percent < 2.0:
            return 0.8
        else:
            return 0.6

    def _calculate_regime_multiplier(self, regime: str) -> float:
        """Calculate position size multiplier based on market regime"""
        regime_multipliers = {
            "LOW": 1.1,  # Low volatility - slightly larger positions
            "NORMAL": 1.0,  # Normal volatility - standard size
            "HIGH": 0.8,  # High volatility - smaller positions
            "UNKNOWN": 0.9,  # Unknown regime - slightly conservative
        }

        return regime_multipliers.get(regime.upper(), 1.0)

    def _calculate_session_multiplier(self, session: str) -> float:
        """Calculate position size multiplier based on trading session"""
        if not self.session_scaling:
            return 1.0

        # Session-based risk adjustments
        session_multipliers = {
            "london": 1.1,  # High liquidity session
            "ny": 1.0,  # Standard session
            "asia": 0.9,  # Lower liquidity, more conservative
            "off": 0.7,  # Off-hours, very conservative
        }

        return session_multipliers.get(session.lower(), 1.0)

    def _is_forex_symbol(self, symbol: str) -> bool:
        """Check if symbol is a forex pair"""
        # Simple heuristic - forex symbols are typically 6 characters (EURUSD)
        return len(symbol) == 6 and symbol.isalpha()

    def calculate_portfolio_heat(self, current_positions: dict, account_balance: float) -> dict:
        """
        Calculate current portfolio heat (total risk exposure)

        Returns:
            Dictionary with heat metrics
        """
        total_risk = 0.0
        position_count = len(current_positions)

        for symbol, position in current_positions.items():
            # Calculate risk for each position
            size = position.get('size', 0)
            entry_price = position.get('entry_price', 0)
            stop_loss = position.get('stop_loss', 0)

            if size != 0 and entry_price != 0 and stop_loss != 0:
                stop_distance = abs(entry_price - stop_loss)
                position_risk = abs(size) * stop_distance

                # Convert to USD if forex
                if self._is_forex_symbol(symbol):
                    position_risk *= 100000  # Standard lot size

                total_risk += position_risk

        heat_percent = (total_risk / account_balance) * 100 if account_balance > 0 else 0

        heat_info = {
            "total_risk_usd": total_risk,
            "heat_percent": heat_percent,
            "position_count": position_count,
            "account_balance": account_balance,
            "avg_risk_per_position": total_risk / position_count if position_count > 0 else 0,
        }

        # Log heat information
        self.logger.bind(evt="POSITION").info("portfolio_heat_calculated", **heat_info)

        return heat_info

    def suggest_stop_loss(
        self, symbol: str, entry_price: float, direction: int, atr_value: float, confidence: float
    ) -> tuple[float, str]:
        """
        Suggest optimal stop loss placement

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            direction: Trade direction (+1 for buy, -1 for sell)
            atr_value: Current ATR value
            confidence: Signal confidence

        Returns:
            Tuple of (stop_loss_price, method)
        """
        # Base stop loss using ATR
        atr_multiplier = getattr(self.config.atr, 'sl_mult', 1.6)

        # Adjust multiplier based on confidence
        if confidence > 0.8:
            # High confidence - tighter stop
            atr_multiplier *= 0.8
        elif confidence < 0.6:
            # Low confidence - wider stop
            atr_multiplier *= 1.2

        # Calculate stop loss
        stop_distance = atr_value * atr_multiplier

        if direction > 0:  # Long position
            stop_loss = entry_price - stop_distance
        else:  # Short position
            stop_loss = entry_price + stop_distance

        method = f"atr_{atr_multiplier:.1f}x"

        return stop_loss, method
