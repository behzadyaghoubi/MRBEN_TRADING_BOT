#!/usr/bin/env python3
"""
MR BEN - Risk Management Gates
Comprehensive risk management for trading system
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from .advanced_risk import AdvancedRiskAnalytics
from .loggingx import logger


class GateResult(Enum):
    """Risk gate evaluation results"""

    PASS = "pass"
    REJECT = "reject"
    WARNING = "warning"


@dataclass
class RiskGateResponse:
    """Response from a risk gate evaluation"""

    result: GateResult
    gate_name: str
    reason: str
    data: dict = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class SpreadGate:
    """
    Spread Gate: Rejects trades when spread is too wide
    Protects against poor execution conditions
    """

    def __init__(self, config):
        self.max_spread_pips = config.risk_management.gates.spread.max_pips
        self.max_spread_percent = config.risk_management.gates.spread.max_percent
        self.enabled = config.risk_management.gates.spread.enabled

    def evaluate(self, symbol: str, bid: float, ask: float) -> RiskGateResponse:
        """Evaluate spread conditions"""
        if not self.enabled:
            return RiskGateResponse(result=GateResult.PASS, gate_name="spread", reason="disabled")

        # Calculate spread in pips and percentage
        spread = ask - bid
        spread_pips = spread * 10000 if 'JPY' not in symbol else spread * 100
        spread_percent = (spread / ask) * 100

        data = {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "spread_pips": spread_pips,
            "spread_percent": spread_percent,
            "max_pips": self.max_spread_pips,
            "max_percent": self.max_spread_percent,
        }

        # Check pip limit
        if spread_pips > self.max_spread_pips:
            return RiskGateResponse(
                result=GateResult.REJECT,
                gate_name="spread",
                reason=f"spread_too_wide_pips: {spread_pips:.1f} > {self.max_spread_pips}",
                data=data,
            )

        # Check percentage limit
        if spread_percent > self.max_spread_percent:
            return RiskGateResponse(
                result=GateResult.REJECT,
                gate_name="spread",
                reason=f"spread_too_wide_percent: {spread_percent:.3f}% > {self.max_spread_percent}%",
                data=data,
            )

        # Warning for elevated spread
        warning_threshold = self.max_spread_pips * 0.8
        if spread_pips > warning_threshold:
            return RiskGateResponse(
                result=GateResult.WARNING,
                gate_name="spread",
                reason=f"spread_elevated: {spread_pips:.1f} pips",
                data=data,
            )

        return RiskGateResponse(
            result=GateResult.PASS, gate_name="spread", reason="spread_acceptable", data=data
        )


class ExposureGate:
    """
    Exposure Gate: Limits total position exposure
    Prevents over-leveraging and concentration risk
    """

    def __init__(self, config):
        self.max_exposure_usd = config.risk_management.gates.exposure.max_usd
        self.max_exposure_percent = config.risk_management.gates.exposure.max_percent
        self.per_symbol_limit = config.risk_management.gates.exposure.per_symbol_limit
        self.enabled = config.risk_management.gates.exposure.enabled

    def evaluate(
        self, symbol: str, position_size: float, current_positions: dict, account_balance: float
    ) -> RiskGateResponse:
        """Evaluate exposure limits"""
        if not self.enabled:
            return RiskGateResponse(result=GateResult.PASS, gate_name="exposure", reason="disabled")

        # Calculate current total exposure
        total_exposure = sum(abs(pos.get('size', 0)) for pos in current_positions.values())
        new_total_exposure = total_exposure + abs(position_size)

        # Calculate current symbol exposure
        symbol_exposure = abs(current_positions.get(symbol, {}).get('size', 0))
        new_symbol_exposure = symbol_exposure + abs(position_size)

        # Calculate exposure as percentage of balance
        exposure_percent = (new_total_exposure / account_balance) * 100

        data = {
            "symbol": symbol,
            "position_size": position_size,
            "current_total_exposure": total_exposure,
            "new_total_exposure": new_total_exposure,
            "symbol_exposure": symbol_exposure,
            "new_symbol_exposure": new_symbol_exposure,
            "account_balance": account_balance,
            "exposure_percent": exposure_percent,
            "max_exposure_usd": self.max_exposure_usd,
            "max_exposure_percent": self.max_exposure_percent,
            "per_symbol_limit": self.per_symbol_limit,
        }

        # Check total USD exposure
        if new_total_exposure > self.max_exposure_usd:
            return RiskGateResponse(
                result=GateResult.REJECT,
                gate_name="exposure",
                reason=f"total_exposure_exceeded: ${new_total_exposure:.0f} > ${self.max_exposure_usd:.0f}",
                data=data,
            )

        # Check total percentage exposure
        if exposure_percent > self.max_exposure_percent:
            return RiskGateResponse(
                result=GateResult.REJECT,
                gate_name="exposure",
                reason=f"exposure_percent_exceeded: {exposure_percent:.1f}% > {self.max_exposure_percent}%",
                data=data,
            )

        # Check per-symbol limit
        if new_symbol_exposure > self.per_symbol_limit:
            return RiskGateResponse(
                result=GateResult.REJECT,
                gate_name="exposure",
                reason=f"symbol_exposure_exceeded: {new_symbol_exposure:.2f} > {self.per_symbol_limit}",
                data=data,
            )

        # Warning for high exposure
        warning_threshold = self.max_exposure_percent * 0.8
        if exposure_percent > warning_threshold:
            return RiskGateResponse(
                result=GateResult.WARNING,
                gate_name="exposure",
                reason=f"exposure_elevated: {exposure_percent:.1f}%",
                data=data,
            )

        return RiskGateResponse(
            result=GateResult.PASS, gate_name="exposure", reason="exposure_acceptable", data=data
        )


class DailyLossGate:
    """
    Daily Loss Gate: Stops trading after daily loss limit
    Prevents catastrophic losses on bad days
    """

    def __init__(self, config):
        self.max_daily_loss_usd = config.risk_management.gates.daily_loss.max_usd
        self.max_daily_loss_percent = config.risk_management.gates.daily_loss.max_percent
        self.reset_time_utc = config.risk_management.gates.daily_loss.reset_time_utc
        self.enabled = config.risk_management.gates.daily_loss.enabled
        self.daily_pnl = 0.0
        self.last_reset_date = None

    def update_pnl(self, pnl_change: float):
        """Update daily P&L tracking"""
        self._check_daily_reset()
        self.daily_pnl += pnl_change

    def _check_daily_reset(self):
        """Reset daily P&L at configured time"""
        now = datetime.now(UTC)
        today = now.date()

        if self.last_reset_date != today:
            # Check if we've passed the reset time
            try:
                from datetime import time

                # Parse the time string manually
                time_parts = self.reset_time_utc.split(':')
                reset_time = time(int(time_parts[0]), int(time_parts[1]), int(time_parts[2]))
                reset_datetime = datetime.combine(today, reset_time).replace(tzinfo=UTC)

                if now >= reset_datetime:
                    if self.last_reset_date is not None:  # Don't log on first run
                        logger.bind(evt="RISK").info(
                            "daily_pnl_reset", date=str(today), previous_pnl=self.daily_pnl
                        )
                    self.daily_pnl = 0.0
                    self.last_reset_date = today
            except Exception as e:
                logger.error(f"Error parsing reset time: {e}")
                # Default to midnight UTC if parsing fails
                reset_datetime = datetime.combine(today, time(0, 0, 0)).replace(tzinfo=UTC)
                if now >= reset_datetime:
                    self.daily_pnl = 0.0
                    self.last_reset_date = today

    def evaluate(self, account_balance: float) -> RiskGateResponse:
        """Evaluate daily loss limits"""
        if not self.enabled:
            return RiskGateResponse(
                result=GateResult.PASS, gate_name="daily_loss", reason="disabled"
            )

        self._check_daily_reset()

        loss_percent = (abs(self.daily_pnl) / account_balance) * 100 if self.daily_pnl < 0 else 0

        data = {
            "daily_pnl": self.daily_pnl,
            "account_balance": account_balance,
            "loss_percent": loss_percent,
            "max_loss_usd": self.max_daily_loss_usd,
            "max_loss_percent": self.max_daily_loss_percent,
            "reset_time": str(self.reset_time_utc),
        }

        # Check USD loss limit
        if self.daily_pnl < -self.max_daily_loss_usd:
            return RiskGateResponse(
                result=GateResult.REJECT,
                gate_name="daily_loss",
                reason=f"daily_loss_usd_exceeded: ${self.daily_pnl:.2f} < -${self.max_daily_loss_usd}",
                data=data,
            )

        # Check percentage loss limit
        if loss_percent > self.max_daily_loss_percent:
            return RiskGateResponse(
                result=GateResult.REJECT,
                gate_name="daily_loss",
                reason=f"daily_loss_percent_exceeded: {loss_percent:.1f}% > {self.max_daily_loss_percent}%",
                data=data,
            )

        # Warning for approaching limits
        warning_threshold = self.max_daily_loss_percent * 0.8
        if loss_percent > warning_threshold:
            return RiskGateResponse(
                result=GateResult.WARNING,
                gate_name="daily_loss",
                reason=f"daily_loss_approaching: {loss_percent:.1f}%",
                data=data,
            )

        return RiskGateResponse(
            result=GateResult.PASS,
            gate_name="daily_loss",
            reason="daily_loss_acceptable",
            data=data,
        )


class ConsecutiveGate:
    """
    Consecutive Gate: Limits consecutive signals in same direction
    Prevents overtrading and reduces correlation risk
    """

    def __init__(self, config):
        self.max_consecutive = config.risk_management.gates.consecutive.max_signals
        self.reset_time_hours = config.risk_management.gates.consecutive.reset_time_hours
        self.enabled = config.risk_management.gates.consecutive.enabled
        self.signal_history: list[tuple[datetime, int]] = []  # (timestamp, direction)

    def add_signal(self, direction: int):
        """Add a new signal to history"""
        now = datetime.now(UTC)
        self.signal_history.append((now, direction))
        self._cleanup_old_signals()

    def _cleanup_old_signals(self):
        """Remove old signals outside the reset window"""
        if not self.signal_history:
            return

        cutoff_time = datetime.now(UTC) - timedelta(hours=self.reset_time_hours)
        self.signal_history = [
            (ts, direction) for ts, direction in self.signal_history if ts > cutoff_time
        ]

    def _count_consecutive_signals(self, direction: int) -> int:
        """Count consecutive signals in the same direction"""
        if not self.signal_history:
            return 0

        count = 0
        for ts, sig_dir in reversed(self.signal_history):
            if sig_dir == direction:
                count += 1
            else:
                break

        return count

    def evaluate(self, signal_direction: int) -> RiskGateResponse:
        """Evaluate consecutive signal limits"""
        if not self.enabled:
            return RiskGateResponse(
                result=GateResult.PASS, gate_name="consecutive", reason="disabled"
            )

        self._cleanup_old_signals()

        consecutive_count = self._count_consecutive_signals(signal_direction)
        new_consecutive_count = consecutive_count + 1

        data = {
            "signal_direction": signal_direction,
            "current_consecutive": consecutive_count,
            "new_consecutive": new_consecutive_count,
            "max_consecutive": self.max_consecutive,
            "reset_hours": self.reset_time_hours,
            "history_length": len(self.signal_history),
        }

        if new_consecutive_count > self.max_consecutive:
            return RiskGateResponse(
                result=GateResult.REJECT,
                gate_name="consecutive",
                reason=f"consecutive_limit_exceeded: {new_consecutive_count} > {self.max_consecutive}",
                data=data,
            )

        # Warning when approaching limit
        warning_threshold = max(1, self.max_consecutive - 1)
        if new_consecutive_count >= warning_threshold:
            return RiskGateResponse(
                result=GateResult.WARNING,
                gate_name="consecutive",
                reason=f"consecutive_approaching: {new_consecutive_count}/{self.max_consecutive}",
                data=data,
            )

        return RiskGateResponse(
            result=GateResult.PASS,
            gate_name="consecutive",
            reason="consecutive_acceptable",
            data=data,
        )


class CooldownGate:
    """
    Cooldown Gate: Enforces waiting period after losses
    Prevents revenge trading and emotional decisions
    """

    def __init__(self, config):
        self.cooldown_minutes = config.risk_management.gates.cooldown.minutes_after_loss
        self.loss_threshold = config.risk_management.gates.cooldown.loss_threshold_usd
        self.enabled = config.risk_management.gates.cooldown.enabled
        self.last_loss_time: datetime | None = None
        self.last_loss_amount: float = 0.0

    def record_loss(self, loss_amount: float):
        """Record a loss for cooldown tracking"""
        if loss_amount >= self.loss_threshold:
            self.last_loss_time = datetime.now(UTC)
            self.last_loss_amount = loss_amount

            logger.bind(evt="RISK").info(
                "cooldown_triggered",
                loss_amount=loss_amount,
                cooldown_minutes=self.cooldown_minutes,
            )

    def evaluate(self) -> RiskGateResponse:
        """Evaluate cooldown status"""
        if not self.enabled:
            return RiskGateResponse(result=GateResult.PASS, gate_name="cooldown", reason="disabled")

        if self.last_loss_time is None:
            return RiskGateResponse(
                result=GateResult.PASS, gate_name="cooldown", reason="no_recent_losses"
            )

        now = datetime.now(UTC)
        time_since_loss = now - self.last_loss_time
        cooldown_period = timedelta(minutes=self.cooldown_minutes)

        data = {
            "last_loss_time": self.last_loss_time.isoformat(),
            "last_loss_amount": self.last_loss_amount,
            "time_since_loss_minutes": time_since_loss.total_seconds() / 60,
            "cooldown_minutes": self.cooldown_minutes,
            "remaining_minutes": max(0, (cooldown_period - time_since_loss).total_seconds() / 60),
        }

        if time_since_loss < cooldown_period:
            remaining = cooldown_period - time_since_loss
            remaining_minutes = remaining.total_seconds() / 60

            return RiskGateResponse(
                result=GateResult.REJECT,
                gate_name="cooldown",
                reason=f"cooldown_active: {remaining_minutes:.1f} minutes remaining",
                data=data,
            )

        return RiskGateResponse(
            result=GateResult.PASS, gate_name="cooldown", reason="cooldown_expired", data=data
        )


class RiskManager:
    """
    Main Risk Manager: Coordinates all risk gates
    """

    def __init__(self, config):
        self.config = config
        self.enabled = config.risk_management.enabled

        # Initialize gates
        self.spread_gate = SpreadGate(config)
        self.exposure_gate = ExposureGate(config)
        self.daily_loss_gate = DailyLossGate(config)
        self.consecutive_gate = ConsecutiveGate(config)
        self.cooldown_gate = CooldownGate(config)

        # Initialize advanced risk analytics
        try:
            self.advanced_risk = AdvancedRiskAnalytics(
                config_path="risk_analytics_config.json",
                enable_ml=True,
                enable_dynamic_thresholds=True,
                enable_correlation_analysis=True,
            )
            self.logger.bind(evt="RISK").info("advanced_risk_analytics_initialized")
        except Exception as e:
            self.logger.bind(evt="RISK").warning(
                "advanced_risk_analytics_initialization_failed", error=str(e)
            )
            self.advanced_risk = None

        self.logger = logger

    def evaluate_all_gates(
        self,
        symbol: str,
        bid: float,
        ask: float,
        position_size: float,
        signal_direction: int,
        current_positions: dict,
        account_balance: float,
    ) -> tuple[bool, list[RiskGateResponse]]:
        """
        Evaluate all risk gates for a trading decision

        Returns:
            Tuple of (allowed, gate_responses)
        """
        if not self.enabled:
            return True, [
                RiskGateResponse(
                    result=GateResult.PASS,
                    gate_name="risk_manager",
                    reason="risk_management_disabled",
                )
            ]

        responses = []

        # Evaluate each gate
        gates_to_evaluate = [
            ("spread", lambda: self.spread_gate.evaluate(symbol, bid, ask)),
            (
                "exposure",
                lambda: self.exposure_gate.evaluate(
                    symbol, position_size, current_positions, account_balance
                ),
            ),
            ("daily_loss", lambda: self.daily_loss_gate.evaluate(account_balance)),
            ("consecutive", lambda: self.consecutive_gate.evaluate(signal_direction)),
            ("cooldown", lambda: self.cooldown_gate.evaluate()),
        ]

        rejected = False
        warnings = []

        for gate_name, gate_func in gates_to_evaluate:
            try:
                response = gate_func()
                responses.append(response)

                if response.result == GateResult.REJECT:
                    rejected = True
                    self.logger.bind(evt="RISK").warning(
                        "gate_rejection", gate=gate_name, reason=response.reason, data=response.data
                    )
                elif response.result == GateResult.WARNING:
                    warnings.append(response)
                    self.logger.bind(evt="RISK").info(
                        "gate_warning", gate=gate_name, reason=response.reason, data=response.data
                    )

            except Exception as e:
                self.logger.error(f"Error evaluating {gate_name} gate: {e}")
                responses.append(
                    RiskGateResponse(
                        result=GateResult.REJECT, gate_name=gate_name, reason=f"gate_error: {e}"
                    )
                )
                rejected = True

        # Log warnings if any
        if warnings and not rejected:
            self.logger.bind(evt="RISK").info(
                "risk_warnings", warning_count=len(warnings), warnings=[w.reason for w in warnings]
            )

        allowed = not rejected

        # Log overall result
        if allowed:
            self.logger.bind(evt="RISK").info(
                "risk_gates_passed",
                symbol=symbol,
                position_size=position_size,
                direction=signal_direction,
            )
        else:
            rejections = [r for r in responses if r.result == GateResult.REJECT]
            self.logger.bind(evt="RISK").warning(
                "risk_gates_rejected",
                symbol=symbol,
                position_size=position_size,
                direction=signal_direction,
                rejections=[r.reason for r in rejections],
            )

        return allowed, responses

    def record_trade_result(self, pnl: float, signal_direction: int):
        """Record trade result for risk tracking"""
        # Update daily P&L
        self.daily_loss_gate.update_pnl(pnl)

        # Add signal to consecutive tracking
        self.consecutive_gate.add_signal(signal_direction)

        # Record loss for cooldown if applicable
        if pnl < 0:
            self.cooldown_gate.record_loss(abs(pnl))

        self.logger.bind(evt="RISK").info(
            "trade_result_recorded",
            pnl=pnl,
            direction=signal_direction,
            daily_pnl=self.daily_loss_gate.daily_pnl,
        )

    def get_risk_status(self) -> dict:
        """Get current risk management status"""
        status = {
            "enabled": self.enabled,
            "daily_pnl": self.daily_loss_gate.daily_pnl,
            "consecutive_signals": len(self.consecutive_gate.signal_history),
            "cooldown_active": self.cooldown_gate.last_loss_time is not None,
            "last_reset": self.daily_loss_gate.last_reset_date,
        }

        # Add advanced risk analytics status if available
        if self.advanced_risk:
            try:
                risk_summary = self.advanced_risk.get_risk_summary()
                status["advanced_risk"] = risk_summary
            except Exception as e:
                self.logger.warning(f"Failed to get advanced risk status: {e}")
                status["advanced_risk"] = {"error": str(e)}

        return status
