#!/usr/bin/env python3
"""
MR BEN - AI Agent Bridge System
Provides AI agent supervision and intervention capabilities
"""

import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from .metricsx import observe_agent_decision, observe_agent_intervention
from .typesx import DecisionCard, MarketContext


class AgentAction(str, Enum):
    """AI Agent actions"""

    MONITOR = "monitor"
    WARN = "warn"
    INTERVENE = "intervene"
    BLOCK = "block"
    OPTIMIZE = "optimize"
    RECOMMEND = "recommend"


class AgentConfidence(str, Enum):
    """AI Agent confidence levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentIntervention(BaseModel):
    """AI Agent intervention details"""

    action: AgentAction
    confidence: AgentConfidence
    reason: str
    recommendation: str | None = None
    risk_level: str | None = None
    timestamp: datetime
    metadata: dict[str, Any] = {}


@dataclass
class AgentState:
    """AI Agent current state"""

    is_active: bool = False
    last_decision_review: datetime | None = None
    interventions_count: int = 0
    warnings_count: int = 0
    blocks_count: int = 0
    last_intervention: AgentIntervention | None = None
    performance_score: float = 0.0
    risk_assessment: str = "normal"


class AgentBridge:
    """
    AI Agent Bridge for trading supervision and intervention

    Provides real-time monitoring of trading decisions and can intervene
    when risk thresholds are exceeded or anomalies are detected.
    """

    def __init__(
        self,
        config_path: str | None = None,
        enable_intervention: bool = True,
        risk_threshold: float = 0.8,
        confidence_threshold: float = 0.7,
    ):
        self.config_path = config_path or "agent_config.json"
        self.enable_intervention = enable_intervention
        self.risk_threshold = risk_threshold
        self.confidence_threshold = confidence_threshold

        # Agent state
        self.state = AgentState()
        self.interventions: list[AgentIntervention] = []
        self.decision_history: list[dict[str, Any]] = []

        # Callbacks
        self.on_intervention_callbacks: list[Callable] = []
        self.on_warning_callbacks: list[Callable] = []
        self.on_optimization_callbacks: list[Callable] = []

        # Threading
        self._lock = threading.RLock()
        self._monitoring_thread: threading.Thread | None = None
        self._stop_monitoring = threading.Event()

        # Load configuration
        self._load_config()

        logger.bind(evt="AGENT").info(
            "agent_bridge_initialized",
            intervention_enabled=enable_intervention,
            risk_threshold=risk_threshold,
        )

    def _load_config(self) -> None:
        """Load agent configuration from file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path) as f:
                    config = json.load(f)

                # Update thresholds from config
                if 'risk_threshold' in config:
                    self.risk_threshold = config['risk_threshold']
                if 'confidence_threshold' in config:
                    self.confidence_threshold = config['confidence_threshold']
                if 'enable_intervention' in config:
                    self.enable_intervention = config['enable_intervention']

                logger.bind(evt="AGENT").info("agent_config_loaded", config_path=self.config_path)
            else:
                logger.bind(evt="AGENT").info(
                    "agent_config_not_found", config_path=self.config_path
                )
        except Exception as e:
            logger.bind(evt="AGENT").warning("agent_config_load_failed", error=str(e))

    def start_monitoring(self) -> None:
        """Start the AI agent monitoring thread"""
        with self._lock:
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                logger.bind(evt="AGENT").warning("agent_monitoring_already_running")
                return

            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True, name="AgentMonitor"
            )
            self._monitoring_thread.start()
            self.state.is_active = True

            logger.bind(evt="AGENT").info("agent_monitoring_started")

    def stop_monitoring(self) -> None:
        """Stop the AI agent monitoring thread"""
        with self._lock:
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._stop_monitoring.set()
                self._monitoring_thread.join(timeout=5.0)
                self._monitoring_thread = None

            self.state.is_active = False
            logger.bind(evt="AGENT").info("agent_monitoring_stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop for the AI agent"""
        logger.bind(evt="AGENT").info("agent_monitoring_loop_started")

        while not self._stop_monitoring.is_set():
            try:
                # Perform periodic risk assessment
                self._assess_overall_risk()

                # Check for configuration updates
                self._check_config_updates()

                # Sleep for monitoring interval
                time.sleep(10.0)  # Check every 10 seconds

            except Exception as e:
                logger.bind(evt="AGENT").error("agent_monitoring_loop_error", error=str(e))
                time.sleep(30.0)  # Wait longer on error

        logger.bind(evt="AGENT").info("agent_monitoring_loop_stopped")

    def review_decision(
        self, decision: DecisionCard, context: MarketContext
    ) -> AgentIntervention | None:
        """
        Review a trading decision and potentially intervene

        Args:
            decision: The trading decision to review
            context: Market context for the decision

        Returns:
            AgentIntervention if intervention is needed, None otherwise
        """
        try:
            with self._lock:
                # Record decision for analysis
                decision_record = {
                    'timestamp': datetime.now(UTC),
                    'decision': asdict(decision),
                    'context': asdict(context),
                    'reviewed': True,
                }
                self.decision_history.append(decision_record)

                # Update last decision review
                self.state.last_decision_review = datetime.now(UTC)

                # Analyze decision for potential intervention
                intervention = self._analyze_decision(decision, context)

                if intervention:
                    self._execute_intervention(intervention)
                    return intervention

                # Record successful decision review
                observe_agent_decision(decision.action, decision.dyn_conf, "approved")

                return None

        except Exception as e:
            logger.bind(evt="AGENT").error("decision_review_failed", error=str(e))
            return None

    def _analyze_decision(
        self, decision: DecisionCard, context: MarketContext
    ) -> AgentIntervention | None:
        """
        Analyze a decision for potential intervention

        Args:
            decision: The trading decision to analyze
            context: Market context for the decision

        Returns:
            AgentIntervention if intervention is needed, None otherwise
        """
        # Check risk level
        risk_score = self._calculate_risk_score(decision, context)

        # Check confidence level
        confidence_score = decision.dyn_conf

        # Check for anomalies
        anomalies = self._detect_anomalies(decision, context)

        # Determine if intervention is needed
        if risk_score > self.risk_threshold:
            return AgentIntervention(
                action=AgentAction.INTERVENE,
                confidence=AgentConfidence.HIGH,
                reason=f"Risk score {risk_score:.2f} exceeds threshold {self.risk_threshold}",
                risk_level="high",
                timestamp=datetime.now(UTC),
                metadata={"risk_score": risk_score, "threshold": self.risk_threshold},
            )

        if confidence_score < self.confidence_threshold:
            return AgentIntervention(
                action=AgentAction.WARN,
                confidence=AgentConfidence.MEDIUM,
                reason=f"Confidence {confidence_score:.2f} below threshold {self.confidence_threshold}",
                timestamp=datetime.now(UTC),
                metadata={"confidence": confidence_score, "threshold": self.confidence_threshold},
            )

        if anomalies:
            return AgentIntervention(
                action=AgentAction.WARN,
                confidence=AgentConfidence.MEDIUM,
                reason=f"Anomalies detected: {', '.join(anomalies)}",
                timestamp=datetime.now(UTC),
                metadata={"anomalies": anomalies},
            )

        return None

    def _calculate_risk_score(self, decision: DecisionCard, context: MarketContext) -> float:
        """Calculate risk score for a decision"""
        risk_score = 0.0

        # Base risk from decision confidence
        risk_score += (1.0 - decision.dyn_conf) * 0.3

        # Risk from market regime
        if context.regime == "high":
            risk_score += 0.2
        elif context.regime == "low":
            risk_score += 0.1

        # Risk from session
        if context.session == "overlap":
            risk_score += 0.1

        # Risk from position size
        if decision.lot > 1.0:
            risk_score += 0.1

        # Risk from consecutive signals
        if hasattr(decision, 'consecutive_count') and decision.consecutive_count > 3:
            risk_score += 0.2

        return min(risk_score, 1.0)

    def _detect_anomalies(self, decision: DecisionCard, context: MarketContext) -> list[str]:
        """Detect anomalies in decision or context"""
        anomalies = []

        # Check for unusual decision patterns
        if len(self.decision_history) > 10:
            recent_decisions = self.decision_history[-10:]
            buy_count = sum(
                1
                for d in recent_decisions
                if d['decision']['action'] == 'ENTER' and d['decision']['dir'] == 1
            )
            sell_count = sum(
                1
                for d in recent_decisions
                if d['decision']['action'] == 'ENTER' and d['decision']['dir'] == -1
            )

            if buy_count > 8 or sell_count > 8:
                anomalies.append("imbalanced_decision_pattern")

        # Check for unusual market conditions
        if context.atr_pts > 100:  # Very high volatility
            anomalies.append("extreme_volatility")

        if context.spread_pts > 50:  # Very high spread
            anomalies.append("excessive_spread")

        # Check for unusual account conditions
        if context.equity < 5000:
            anomalies.append("low_account_equity")

        return anomalies

    def _execute_intervention(self, intervention: AgentIntervention) -> None:
        """Execute an agent intervention"""
        try:
            with self._lock:
                # Record intervention
                self.interventions.append(intervention)
                self.state.last_intervention = intervention

                # Update counters
                if intervention.action == AgentAction.INTERVENE:
                    self.state.interventions_count += 1
                elif intervention.action == AgentAction.WARN:
                    self.state.warnings_count += 1
                elif intervention.action == AgentAction.BLOCK:
                    self.state.blocks_count += 1

                # Execute callbacks
                if intervention.action == AgentAction.INTERVENE:
                    for callback in self.on_intervention_callbacks:
                        try:
                            callback(intervention)
                        except Exception as e:
                            logger.bind(evt="AGENT").error(
                                "intervention_callback_failed", callback=str(callback), error=str(e)
                            )

                elif intervention.action == AgentAction.WARN:
                    for callback in self.on_warning_callbacks:
                        try:
                            callback(intervention)
                        except Exception as e:
                            logger.bind(evt="AGENT").error(
                                "warning_callback_failed", callback=str(callback), error=str(e)
                            )

                # Record metrics
                observe_agent_intervention(
                    intervention.action, intervention.confidence, intervention.reason
                )

                logger.bind(evt="AGENT").warning(
                    "agent_intervention_executed",
                    action=intervention.action,
                    reason=intervention.reason,
                    confidence=intervention.confidence,
                )

        except Exception as e:
            logger.bind(evt="AGENT").error("intervention_execution_failed", error=str(e))

    def _assess_overall_risk(self) -> None:
        """Assess overall system risk"""
        try:
            if len(self.decision_history) < 5:
                return

            # Calculate recent performance
            recent_decisions = self.decision_history[-20:]
            successful_decisions = sum(1 for d in recent_decisions if d.get('success', False))
            total_decisions = len(recent_decisions)

            if total_decisions > 0:
                success_rate = successful_decisions / total_decisions
                self.state.performance_score = success_rate

                # Update risk assessment
                if success_rate < 0.3:
                    self.state.risk_assessment = "critical"
                elif success_rate < 0.5:
                    self.state.risk_assessment = "high"
                elif success_rate < 0.7:
                    self.state.risk_assessment = "medium"
                else:
                    self.state.risk_assessment = "low"

        except Exception as e:
            logger.bind(evt="AGENT").error("risk_assessment_failed", error=str(e))

    def _check_config_updates(self) -> None:
        """Check for configuration file updates"""
        try:
            if Path(self.config_path).exists():
                # Simple file modification time check
                # In production, could use file watchers or API calls
                self._load_config()
        except Exception as e:
            logger.bind(evt="AGENT").error("config_update_check_failed", error=str(e))

    def add_intervention_callback(self, callback: Callable) -> None:
        """Add callback for intervention events"""
        self.on_intervention_callbacks.append(callback)

    def add_warning_callback(self, callback: Callable) -> None:
        """Add callback for warning events"""
        self.on_warning_callbacks.append(callback)

    def add_optimization_callback(self, callback: Callable) -> None:
        """Add callback for optimization events"""
        self.on_optimization_callbacks.append(callback)

    def get_status(self) -> dict[str, Any]:
        """Get current agent status"""
        with self._lock:
            return {
                "is_active": self.state.is_active,
                "last_decision_review": (
                    self.state.last_decision_review.isoformat()
                    if self.state.last_decision_review
                    else None
                ),
                "interventions_count": self.state.interventions_count,
                "warnings_count": self.state.warnings_count,
                "blocks_count": self.state.blocks_count,
                "performance_score": self.state.performance_score,
                "risk_assessment": self.state.risk_assessment,
                "last_intervention": (
                    asdict(self.state.last_intervention) if self.state.last_intervention else None
                ),
                "total_decisions_reviewed": len(self.decision_history),
                "recent_interventions": [asdict(i) for i in self.interventions[-5:]],
            }

    def get_recommendations(self) -> list[dict[str, Any]]:
        """Get AI agent recommendations"""
        recommendations = []

        # Performance-based recommendations
        if self.state.performance_score < 0.5:
            recommendations.append(
                {
                    "type": "performance",
                    "priority": "high",
                    "message": "Consider reducing position sizes or reviewing strategy parameters",
                    "confidence": "high",
                }
            )

        # Risk-based recommendations
        if self.state.risk_assessment in ["high", "critical"]:
            recommendations.append(
                {
                    "type": "risk",
                    "priority": "critical",
                    "message": "Risk level is elevated - consider emergency stop or position reduction",
                    "confidence": "high",
                }
            )

        # Anomaly-based recommendations
        if self.state.warnings_count > 5:
            recommendations.append(
                {
                    "type": "anomaly",
                    "priority": "medium",
                    "message": "Multiple anomalies detected - review market conditions and strategy",
                    "confidence": "medium",
                }
            )

        return recommendations

    def reset_statistics(self) -> None:
        """Reset agent statistics"""
        with self._lock:
            self.interventions.clear()
            self.decision_history.clear()
            self.state.interventions_count = 0
            self.state.warnings_count = 0
            self.state.blocks_count = 0
            self.state.performance_score = 0.0
            self.state.risk_assessment = "normal"
            self.state.last_intervention = None

            logger.bind(evt="AGENT").info("agent_statistics_reset")

    def cleanup(self) -> None:
        """Cleanup agent resources"""
        self.stop_monitoring()
        logger.bind(evt="AGENT").info("agent_bridge_cleanup_complete")
