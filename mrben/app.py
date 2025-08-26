#!/usr/bin/env python3
"""
MR BEN - Main Trading Application
Integrates all components with performance metrics and telemetry
"""

from __future__ import annotations

import argparse
import time

from core.ab import ABRunner
from core.agent_bridge import AgentBridge
from core.configx import cfg_to_json, load_config
from core.context_factory import ContextFactory
from core.emergency_stop import EmergencyStop
from core.loggingx import log_cfg, setup_logging
from core.metricsx import (
    init_metrics,
    observe_block,
    observe_order_send,
    observe_risk_gate,
    observe_trade_close,
    observe_trade_open,
    update_context,
)


def main():
    """Main application entry point"""
    ap = argparse.ArgumentParser(description="MR BEN Trading System")
    ap.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    ap.add_argument("--dry-run", action="store_true", help="Dry run mode - no real trading")
    ap.add_argument("--metrics-port", type=int, help="Override metrics port from config")
    ap.add_argument("--ab-test", action="store_true", help="Enable A/B testing mode")
    ap.add_argument(
        "--emergency-stop", action="store_true", help="Test emergency stop functionality"
    )
    ap.add_argument("--agent-supervision", action="store_true", help="Test AI agent supervision")
    args = ap.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Setup logging
    logger = setup_logging(cfg.logging.level)
    log_cfg(cfg_to_json(cfg))

    # Initialize metrics
    metrics_port = args.metrics_port or cfg.metrics.port
    if cfg.metrics.enabled:
        init_metrics(metrics_port)
        logger.bind(evt="BOOT").info("metrics_initialized", port=metrics_port)

    # Initialize emergency stop system
    emergency_stop = None
    if cfg.emergency_stop.enabled:
        emergency_stop = EmergencyStop(
            halt_file_path=cfg.emergency_stop.halt_file_path,
            check_interval=cfg.emergency_stop.check_interval,
            auto_recovery=cfg.emergency_stop.auto_recovery,
            recovery_delay=cfg.emergency_stop.recovery_delay,
        )

        if cfg.emergency_stop.monitoring_enabled:
            emergency_stop.start_monitoring()
            logger.bind(evt="BOOT").info("emergency_stop_monitoring_started")

        logger.bind(evt="BOOT").info(
            "emergency_stop_initialized",
            halt_file=cfg.emergency_stop.halt_file_path,
            auto_recovery=cfg.emergency_stop.auto_recovery,
        )

    # Initialize AI Agent Bridge
    agent_bridge = None
    try:
        agent_bridge = AgentBridge(
            config_path="agent_config.json",
            enable_intervention=True,
            risk_threshold=0.8,
            confidence_threshold=0.7,
        )
        agent_bridge.start_monitoring()
        logger.bind(evt="BOOT").info(
            "agent_bridge_initialized", intervention_enabled=True, risk_threshold=0.8
        )
    except Exception as e:
        logger.bind(evt="BOOT").warning("agent_bridge_initialization_failed", error=str(e))
        agent_bridge = None

    if args.dry_run:
        logger.bind(evt="BOOT").info("dry_run_ok")

        # Simulate some metrics for testing
        if cfg.metrics.enabled:
            simulate_metrics()
            logger.bind(evt="BOOT").info("metrics_simulation_complete")

        # Test emergency stop if requested
        if args.emergency_stop and emergency_stop:
            test_emergency_stop(emergency_stop)
            logger.bind(evt="BOOT").info("emergency_stop_testing_complete")

        # Run A/B testing if enabled
        if args.ab_test:
            run_ab_testing(emergency_stop, agent_bridge)
            logger.bind(evt="BOOT").info("ab_testing_complete")

        # Test agent supervision if requested
        if args.agent_supervision and agent_bridge:
            test_agent_supervision(agent_bridge)
            logger.bind(evt="BOOT").info("agent_supervision_testing_complete")

        # Cleanup
        if emergency_stop:
            emergency_stop.cleanup()

        if agent_bridge:
            agent_bridge.cleanup()

        return 0

    # TODO: Initialize remaining modules in future steps
    logger.bind(evt="BOOT").info("system_ready")

    return 0


def simulate_metrics():
    """Simulate metrics for testing purposes"""
    # Simulate context updates
    update_context(
        equity=10000.0,
        balance=10000.0,
        spread_pts=2.0,
        session="london",
        regime="NORMAL",
        dyn_conf=0.75,
        score=0.68,
        open_positions=2,
    )

    # Simulate some blocks
    observe_block("ml_low_conf")
    observe_block("risk_exposure")
    observe_block("risk_daily_loss")

    # Simulate order execution
    observe_order_send("ioc", 45.2, 1.5)
    observe_order_send("fok", 23.1, 0.8)
    observe_order_send("return", 67.8, 2.1)

    # Simulate trade lifecycle
    observe_trade_open("EURUSD", 1, "pro")
    observe_trade_open("GBPUSD", -1, "pro")

    observe_trade_close("EURUSD", 1, "pro", 1.2)
    observe_trade_close("GBPUSD", -1, "pro", -0.8)

    # Simulate risk gate observations
    observe_risk_gate("spread", True)
    observe_risk_gate("exposure", False)
    observe_risk_gate("daily_loss", False)


def test_emergency_stop(emergency_stop: EmergencyStop):
    """Test emergency stop functionality"""
    from core.loggingx import logger

    logger.bind(evt="EMERGENCY").info("starting_emergency_stop_test")

    # Test 1: Check initial state
    initial_state = emergency_stop.get_state()
    logger.bind(evt="EMERGENCY").info("initial_state", state=initial_state)

    # Test 2: Manual emergency stop
    logger.bind(evt="EMERGENCY").info("triggering_manual_emergency_stop")
    emergency_stop.manual_emergency_stop("Test emergency stop")

    # Wait a moment for state to update
    time.sleep(1.0)

    # Test 3: Check emergency state
    emergency_state = emergency_stop.get_state()
    logger.bind(evt="EMERGENCY").info("emergency_state", state=emergency_state)

    # Test 4: Check halt file
    halt_info = emergency_stop.get_halt_file_info()
    if halt_info:
        logger.bind(evt="EMERGENCY").info("halt_file_info", info=halt_info)

    # Test 5: Manual recovery
    logger.bind(evt="EMERGENCY").info("triggering_manual_recovery")
    emergency_stop.manual_recovery()

    # Wait a moment for state to update
    time.sleep(1.0)

    # Test 6: Check recovery state
    recovery_state = emergency_stop.get_state()
    logger.bind(evt="EMERGENCY").info("recovery_state", state=recovery_state)

    logger.bind(evt="EMERGENCY").info("emergency_stop_test_complete")


def run_ab_testing(emergency_stop: EmergencyStop = None, agent_bridge: AgentBridge = None):
    """Run A/B testing demonstration"""
    from core.loggingx import logger

    logger.bind(evt="AB").info("starting_ab_testing_demo")

    # Create context factory
    ctx_factory = ContextFactory()

    # Create A/B runner with emergency stop and agent bridge integration
    ab_runner = ABRunner(
        ctx_factory=ctx_factory.create_from_bar,
        symbol="EURUSD",
        emergency_stop=emergency_stop,
        agent_bridge=agent_bridge,
    )

    # Simulate some bar data for testing
    test_bars = [
        # Bar 1: Strong uptrend
        {
            'timestamp': '2024-01-01T10:00:00Z',
            'close': 1.1050,
            'bid': 1.1049,
            'ask': 1.1051,
            'atr_pts': 25.0,
            'sma20': 1.1040,
            'sma50': 1.1020,
            'equity': 10000.0,
            'balance': 10000.0,
            'spread_pts': 20.0,
            'open_positions': 0,
        },
        # Bar 2: Weak trend
        {
            'timestamp': '2024-01-01T10:01:00Z',
            'close': 1.1045,
            'bid': 1.1044,
            'ask': 1.1046,
            'atr_pts': 15.0,
            'sma20': 1.1042,
            'sma50': 1.1040,
            'equity': 10000.0,
            'balance': 10000.0,
            'spread_pts': 20.0,
            'open_positions': 0,
        },
        # Bar 3: Downtrend
        {
            'timestamp': '2024-01-01T10:02:00Z',
            'close': 1.1030,
            'bid': 1.1029,
            'ask': 1.1031,
            'atr_pts': 30.0,
            'sma20': 1.1035,
            'sma50': 1.1045,
            'equity': 10000.0,
            'balance': 10000.0,
            'spread_pts': 20.0,
            'open_positions': 0,
        },
    ]

    # Process each bar
    for i, bar_data in enumerate(test_bars):
        logger.bind(evt="AB").info(f"processing_bar_{i+1}", bar_data=bar_data)
        ab_runner.on_bar(bar_data)

        # Simulate some ticks for position management
        if i == 0:  # First bar opened a position
            simulate_ticks_for_position(ab_runner)

    # Get final statistics including emergency stop info
    stats = ab_runner.get_statistics()
    logger.bind(evt="AB").info("ab_testing_complete", statistics=stats)

    # Get emergency stop status if available
    if emergency_stop:
        emergency_status = ab_runner.get_emergency_status()
        logger.bind(evt="AB").info("emergency_status", status=emergency_status)

    # Get agent status if available
    if agent_bridge:
        agent_status = ab_runner.get_statistics().get('agent_status', {})
        logger.bind(evt="AB").info("agent_status", status=agent_status)

    # Cleanup
    ab_runner.cleanup()


def test_agent_supervision(agent_bridge: AgentBridge):
    """Test AI agent supervision functionality"""
    from core.loggingx import logger
    from core.typesx import DecisionCard, Levels, MarketContext

    logger.bind(evt="AGENT").info("starting_agent_supervision_test")

    # Test 1: Normal decision review
    normal_decision = DecisionCard(
        action="ENTER",
        direction=1,
        reason="SMA crossover with high confidence",
        score=0.85,
        confidence=0.92,
        lot=1.0,
        levels=Levels(sl=1.1000, tp1=1.1100, tp2=1.1150),
        track="pro",
    )

    normal_context = MarketContext(
        price=1.1050,
        bid=1.1049,
        ask=1.1051,
        atr_pts=25.0,
        sma20=1.1040,
        sma50=1.1020,
        session="london",
        regime="NORMAL",
        equity=10000.0,
        balance=10000.0,
        spread_pts=20.0,
        open_positions=0,
    )

    # Review normal decision
    intervention = agent_bridge.review_decision(normal_decision, normal_context)
    if intervention:
        logger.bind(evt="AGENT").warning(
            "unexpected_intervention_on_normal_decision",
            action=intervention.action,
            reason=intervention.reason,
        )
    else:
        logger.bind(evt="AGENT").info("normal_decision_approved")

    # Test 2: High-risk decision review
    risky_decision = DecisionCard(
        action="ENTER",
        direction=1,
        reason="Aggressive entry with low confidence",
        score=0.45,
        confidence=0.35,
        lot=2.5,
        levels=Levels(sl=1.1000, tp1=1.1100, tp2=1.1150),
        track="pro",
    )

    risky_context = MarketContext(
        price=1.1050,
        bid=1.1049,
        ask=1.1051,
        atr_pts=120.0,  # High volatility
        sma20=1.1040,
        sma50=1.1020,
        session="overlap",  # Overlap session
        regime="HIGH",
        equity=8000.0,  # Lower equity
        balance=10000.0,
        spread_pts=45.0,  # Higher spread
        open_positions=3,  # Multiple positions
    )

    # Review risky decision
    intervention = agent_bridge.review_decision(risky_decision, risky_context)
    if intervention:
        logger.bind(evt="AGENT").info(
            "risky_decision_intervention",
            action=intervention.action,
            reason=intervention.reason,
            confidence=intervention.confidence,
        )
    else:
        logger.bind(evt="AGENT").warning("risky_decision_not_intervened")

    # Test 3: Get agent status and recommendations
    status = agent_bridge.get_status()
    recommendations = agent_bridge.get_recommendations()

    logger.bind(evt="AGENT").info("agent_status", status=status)
    logger.bind(evt="AGENT").info("agent_recommendations", recommendations=recommendations)

    logger.bind(evt="AGENT").info("agent_supervision_test_complete")


def simulate_ticks_for_position(ab_runner):
    """Simulate ticks for position management"""
    from core.loggingx import logger

    # Simulate ticks that would trigger TP1 and TP2
    test_ticks = [
        {'bid': 1.1055, 'ask': 1.1057, 'atr_pts': 25.0},  # TP1 hit
        {'bid': 1.1065, 'ask': 1.1067, 'atr_pts': 25.0},  # TP2 hit
    ]

    for i, tick_data in enumerate(test_ticks):
        logger.bind(evt="AB").info(f"processing_tick_{i+1}", tick_data=tick_data)
        ab_runner.on_tick(tick_data)
        time.sleep(0.1)  # Small delay for demo


if __name__ == "__main__":
    raise SystemExit(main())
