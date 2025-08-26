#!/usr/bin/env python3
"""
MR BEN - A/B Testing Orchestrator
Runs Control vs Pro deciders simultaneously for performance comparison
"""

from __future__ import annotations

from collections.abc import Callable

from .agent_bridge import AgentBridge
from .deciders import ControlDecider, ProDecider
from .emergency_stop import EmergencyStop
from .loggingx import logger
from .metricsx import observe_block, update_context
from .paper import PaperBroker
from .trading_guard import TradingGuard
from .typesx import DecisionCard, MarketContext


class ABRunner:
    """A/B Testing runner for comparing Control vs Pro strategies"""

    def __init__(
        self,
        ctx_factory: Callable,
        symbol: str,
        emergency_stop: EmergencyStop | None = None,
        agent_bridge: AgentBridge | None = None,
    ):
        self.ctx_factory = ctx_factory  # Function that provides fresh context
        self.symbol = symbol
        self.paper = PaperBroker(symbol, track="control")

        # Emergency stop integration
        self.emergency_stop = emergency_stop
        self.trading_guard = TradingGuard(emergency_stop) if emergency_stop else None

        # AI Agent integration
        self.agent_bridge = agent_bridge

        # Statistics tracking
        self.control_decisions = 0
        self.pro_decisions = 0
        self.control_enters = 0
        self.pro_enters = 0

        logger.bind(evt="AB").info(
            "ab_runner_initialized",
            symbol=symbol,
            emergency_stop_enabled=emergency_stop is not None,
            agent_bridge_enabled=agent_bridge is not None,
        )

    def on_bar(self, bar_data: dict) -> None:
        """Process bar data and make decisions"""
        try:
            # Check emergency stop first
            if self.emergency_stop and not self.emergency_stop.is_trading_allowed():
                logger.bind(evt="AB").warning(
                    "bar_processing_blocked_by_emergency", symbol=self.symbol
                )
                return

            # Create fresh context from bar data
            ctx = self.ctx_factory(bar_data)

            # Generate decision cards with emergency stop protection
            dc_control = self._make_control_decision(ctx)
            dc_pro = self._make_pro_decision(ctx)

            # AI Agent review of Pro decision
            if self.agent_bridge:
                intervention = self.agent_bridge.review_decision(dc_pro, ctx)
                if intervention:
                    logger.bind(evt="AB").warning(
                        "agent_intervention_detected",
                        action=intervention.action,
                        reason=intervention.reason,
                        confidence=intervention.confidence,
                    )
                    # Apply agent intervention if needed
                    dc_pro = self._apply_agent_intervention(dc_pro, intervention)

            # Log decisions
            self._log_decisions(dc_control, dc_pro)

            # Track decision counts
            self.control_decisions += 1
            self.pro_decisions += 1

            # Execute strategies (only if trading is allowed)
            if self.emergency_stop and self.emergency_stop.is_trading_allowed():
                self._execute_control(dc_control)
                self._execute_pro(dc_pro, ctx)

                # Update context metrics
                self._update_metrics(ctx, dc_pro)
            else:
                logger.bind(evt="AB").info(
                    "strategy_execution_skipped_emergency_stop", symbol=self.symbol
                )

        except Exception as e:
            logger.bind(evt="AB").error("bar_processing_error", symbol=self.symbol, error=str(e))

    def on_tick(self, tick_data: dict) -> None:
        """Process tick data for position management"""
        try:
            # Check emergency stop for tick processing
            if self.emergency_stop and not self.emergency_stop.is_trading_allowed():
                logger.bind(evt="AB").debug(
                    "tick_processing_blocked_by_emergency", symbol=self.symbol
                )
                return

            # Paper side management
            bid = tick_data.get('bid', 0.0)
            ask = tick_data.get('ask', 0.0)
            atr_pts = tick_data.get('atr_pts', 0.0)

            self.paper.on_tick(bid=bid, ask=ask, atr_pts=atr_pts)

        except Exception as e:
            logger.bind(evt="AB").error("tick_processing_error", symbol=self.symbol, error=str(e))

    def _make_control_decision(self, ctx: MarketContext) -> DecisionCard:
        """Make control decision using SMA-only strategy"""
        try:
            # Apply emergency stop guard if available
            if self.trading_guard:

                @self.trading_guard.guard_decision_making("control_decision")
                def make_decision():
                    decider = ControlDecider(ctx)
                    return decider.decide()

                return make_decision()
            else:
                # No guard, proceed normally
                decider = ControlDecider(ctx)
                return decider.decide()

        except Exception as e:
            logger.bind(evt="AB").error("control_decision_error", symbol=self.symbol, error=str(e))
            # Return safe HOLD decision
            return DecisionCard(
                action="HOLD",
                dir=0,
                reason="control_error",
                score=0.0,
                dyn_conf=0.0,
                track="control",
            )

    def _make_pro_decision(self, ctx: MarketContext) -> DecisionCard:
        """Make pro decision using ensemble strategy"""
        try:
            # Apply emergency stop guard if available
            if self.trading_guard:

                @self.trading_guard.guard_decision_making("pro_decision")
                def make_decision():
                    decider = ProDecider(ctx)
                    return decider.decide()

                return make_decision()
            else:
                # No guard, proceed normally
                decider = ProDecider(ctx)
                return decider.decide()

        except Exception as e:
            logger.bind(evt="AB").error("pro_decision_error", symbol=self.symbol, error=str(e))
            # Return safe HOLD decision
            return DecisionCard(
                action="HOLD", dir=0, reason="pro_error", score=0.0, dyn_conf=0.0, track="pro"
            )

    def _log_decisions(self, dc_control: DecisionCard, dc_pro: DecisionCard) -> None:
        """Log both decisions for comparison"""
        logger.bind(evt="AB").info(
            "decisions_comparison",
            symbol=self.symbol,
            control_action=dc_control.action,
            control_reason=dc_control.reason,
            control_score=dc_control.score,
            pro_action=dc_pro.action,
            pro_reason=dc_pro.reason,
            pro_score=dc_pro.score,
        )

    def _execute_control(self, dc: DecisionCard) -> None:
        """Execute control strategy (paper trading only)"""
        if dc.action == "ENTER":
            self.control_enters += 1
            self.paper.open(dc)

            logger.bind(evt="AB").info(
                "control_position_opened",
                symbol=self.symbol,
                direction="buy" if dc.dir > 0 else "sell",
                lot=dc.lot,
            )
        else:
            # Log blocks
            observe_block(f"control_{dc.reason}")

            logger.bind(evt="AB").debug(
                "control_decision_blocked", symbol=self.symbol, reason=dc.reason
            )

    def _execute_pro(self, dc: DecisionCard, ctx: MarketContext) -> None:
        """Execute pro strategy (real execution simulation)"""
        if dc.action == "ENTER":
            self.pro_enters += 1

            # Simulate real execution (for now, just log)
            logger.bind(evt="AB").info(
                "pro_position_executed",
                symbol=self.symbol,
                direction="buy" if dc.dir > 0 else "sell",
                lot=dc.lot,
                entry=ctx.mid_price,
                sl=dc.levels.sl if dc.levels else 0.0,
                tp1=dc.levels.tp1 if dc.levels else 0.0,
                tp2=dc.levels.tp2 if dc.levels else 0.0,
            )

            # In real implementation, this would call ctx.execute_real(dc)
            # For now, we just simulate the execution
            self._simulate_pro_execution(dc, ctx)

        else:
            # Log blocks
            observe_block(f"pro_{dc.reason}")

            logger.bind(evt="AB").debug(
                "pro_decision_blocked", symbol=self.symbol, reason=dc.reason
            )

    def _simulate_pro_execution(self, dc: DecisionCard, ctx: MarketContext) -> None:
        """Simulate pro execution for demonstration"""
        # This is a placeholder for real execution
        # In production, this would integrate with the real execution system

        # Simulate execution success
        execution_success = True  # Assume success for demo

        if execution_success:
            logger.bind(evt="AB").info(
                "pro_execution_simulated",
                symbol=self.symbol,
                track="pro",
                action="ENTER",
                direction="buy" if dc.dir > 0 else "sell",
            )
        else:
            logger.bind(evt="AB").warning(
                "pro_execution_failed", symbol=self.symbol, track="pro", action="ENTER"
            )

    def _apply_agent_intervention(
        self, decision: DecisionCard, intervention: AgentIntervention
    ) -> DecisionCard:
        """Apply agent intervention to decision"""
        try:
            # Create a copy of the decision with intervention applied
            from copy import deepcopy

            modified_decision = deepcopy(decision)

            # Apply intervention based on action type
            if intervention.action.value == "block":
                # Block execution by setting action to HOLD
                modified_decision.action = "HOLD"
                modified_decision.lot = 0.0

            elif intervention.action.value == "intervene":
                # Reduce position size and confidence
                modified_decision.lot = max(0.1, decision.lot * 0.5)
                modified_decision.confidence = max(0.1, decision.confidence * 0.7)
                modified_decision.reason = f"INTERVENED: {intervention.reason}"

            elif intervention.action.value == "warn":
                # Add warning to reason but don't modify decision
                modified_decision.reason = f"WARNED: {intervention.reason} | {decision.reason}"

            logger.bind(evt="AB").info(
                "agent_intervention_applied",
                original_action=decision.action,
                modified_action=modified_decision.action,
                intervention_type=intervention.action.value,
            )

            return modified_decision

        except Exception as e:
            logger.bind(evt="AB").error("agent_intervention_application_failed", error=str(e))
            return decision  # Return original decision on error

    def _update_metrics(self, ctx: MarketContext, dc_pro: DecisionCard) -> None:
        """Update context metrics for monitoring"""
        try:
            update_context(
                equity=ctx.equity,
                balance=ctx.balance,
                spread_pts=ctx.spread_pts,
                session=ctx.session,
                regime=ctx.regime,
                dyn_conf=dc_pro.dyn_conf,
                score=dc_pro.score,
                open_positions=ctx.open_positions,
            )
        except Exception as e:
            logger.bind(evt="AB").error("metrics_update_error", symbol=self.symbol, error=str(e))

    def get_statistics(self) -> dict:
        """Get A/B testing statistics"""
        control_stats = self.paper.get_statistics()

        # Include emergency stop information if available
        emergency_info = {}
        if self.emergency_stop:
            emergency_info = {
                "trading_allowed": self.emergency_stop.is_trading_allowed(),
                "emergency_active": self.emergency_stop.get_state().is_active,
                "emergency_reason": self.emergency_stop.get_state().trigger_reason,
            }

        if self.trading_guard:
            guard_info = self.trading_guard.get_status_summary()
            emergency_info.update(guard_info)

        stats = {
            "symbol": self.symbol,
            "control": {
                "decisions": self.control_decisions,
                "enters": self.control_enters,
                "paper_trades": control_stats,
            },
            "pro": {"decisions": self.pro_decisions, "enters": self.pro_enters},
            "comparison": {
                "control_enter_rate": (self.control_enters / max(self.control_decisions, 1)) * 100,
                "pro_enter_rate": (self.pro_enters / max(self.pro_decisions, 1)) * 100,
            },
            "emergency_stop": emergency_info,
        }

        # Add agent status if available
        if self.agent_bridge:
            stats['agent_status'] = self.agent_bridge.get_status()
            stats['agent_recommendations'] = self.agent_bridge.get_recommendations()

        return stats

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self.paper.close_all()
            logger.bind(evt="AB").info("ab_runner_cleanup_complete", symbol=self.symbol)
        except Exception as e:
            logger.bind(evt="AB").error("cleanup_error", symbol=self.symbol, error=str(e))

    def reset_statistics(self) -> None:
        """Reset all statistics"""
        self.control_decisions = 0
        self.pro_decisions = 0
        self.control_enters = 0
        self.pro_enters = 0

        # Reset trading guard statistics if available
        if self.trading_guard:
            self.trading_guard.reset_blocked_operations_count()

        # Reset agent statistics if available
        if self.agent_bridge:
            self.agent_bridge.reset_statistics()

        logger.bind(evt="AB").info("statistics_reset", symbol=self.symbol)

    def get_emergency_status(self) -> dict | None:
        """Get emergency stop status if available"""
        if self.emergency_stop:
            return {
                "trading_allowed": self.emergency_stop.is_trading_allowed(),
                "state": self.emergency_stop.get_state(),
                "halt_file_info": self.emergency_stop.get_halt_file_info(),
            }
        return None
