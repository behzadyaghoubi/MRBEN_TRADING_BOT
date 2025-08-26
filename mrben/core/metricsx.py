#!/usr/bin/env python3
"""
MR BEN - Performance Metrics & Telemetry System
Prometheus metrics for real-time monitoring and performance tracking
"""

from __future__ import annotations
from typing import Optional
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, start_http_server
)

# ==== Counters ====
TRADES_OPENED = Counter(
    'mrben_trades_opened_total', 'Opened trades',
    ['symbol', 'dir', 'track']
)
TRADES_CLOSED = Counter(
    'mrben_trades_closed_total', 'Closed trades',
    ['symbol', 'dir', 'track', 'outcome']
)
BLOCKS = Counter(
    'mrben_blocks_total', 'Blocked decisions by reason',
    ['reason']
)
ORDERS_SENT = Counter(
    'mrben_orders_sent_total', 'Orders sent by mode',
    ['mode']
)

# ==== Gauges ====
EQUITY = Gauge('mrben_equity', 'Current equity (base currency)')
BALANCE = Gauge('mrben_balance', 'Account balance (base currency)')
DRAWDOWN = Gauge('mrben_drawdown_pct', 'Current drawdown %')
EXPOSURE = Gauge('mrben_exposure_positions', 'Open positions count')
SPREAD = Gauge('mrben_spread_points', 'Current spread (points)')
CONF_DYN = Gauge('mrben_confidence_dyn', 'Dynamic confidence [0..1]')
SCORE = Gauge('mrben_decision_score', 'Decision score [0..1]')
REGIME = Gauge('mrben_regime_code', 'Regime code: LOW=0, NORMAL=1, HIGH=2, UNKNOWN=-1')
SESSION = Gauge('mrben_session_code', 'Session code: asia=0,london=1,ny=2,off=3')

# ==== Histograms / Summaries ====
TRADE_R = Histogram(
    'mrben_trade_r', 'Per-trade R multiple',
    buckets=[-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 5]
)
SLIPPAGE = Histogram(
    'mrben_slippage_points', 'Execution slippage (points)',
    buckets=[0, 2, 5, 10, 20, 40, 80]
)
LATENCY = Histogram(
    'mrben_order_latency_ms', 'Order send latency (ms)',
    buckets=[5, 10, 20, 50, 100, 200, 400, 800, 1600]
)
PAYOUT = Summary('mrben_trade_payout_r', 'Per-trade payout (R) summary')

# ==== Mapping Constants ====
_REGIME_MAP = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "UNKNOWN": -1}
_SESSION_MAP = {"asia": 0, "london": 1, "ny": 2, "off": 3}

# ==== Global State ====
_started = False
_peak_equity: Optional[float] = None


def init_metrics(port: int = 8765):
    """Initialize Prometheus metrics server"""
    global _started
    if not _started:
        start_http_server(port)
        _started = True


# ===== Context updaters =====


def update_context(equity: float, balance: float, spread_pts: float,
                   session: str, regime: str, dyn_conf: float, score: float,
                   open_positions: int):
    """Update context metrics with current values"""
    global _peak_equity
    
    # Update basic metrics
    EQUITY.set(equity)
    BALANCE.set(balance)
    SPREAD.set(spread_pts)
    CONF_DYN.set(dyn_conf)
    SCORE.set(score)
    EXPOSURE.set(open_positions)
    
    # Update regime and session codes
    REGIME.set(_REGIME_MAP.get(regime.upper(), -1))
    SESSION.set(_SESSION_MAP.get(session, 3))
    
    # Calculate and update drawdown
    if _peak_equity is None or equity > _peak_equity:
        _peak_equity = equity
    
    dd = 0.0
    if _peak_equity and _peak_equity > 0:
        dd = max(0.0, 1.0 - equity / float(_peak_equity))
    
    DRAWDOWN.set(dd * 100.0)


# ===== Decision events =====


def observe_block(reason: str):
    """Observe a blocked decision"""
    BLOCKS.labels(reason=reason).inc()


# ===== Order events =====


def observe_order_send(mode: str, latency_ms: float, slippage_pts: float):
    """Observe order execution metrics"""
    ORDERS_SENT.labels(mode=mode).inc()
    LATENCY.observe(latency_ms)
    SLIPPAGE.observe(max(0.0, slippage_pts))


# ===== Trade lifecycle =====


def observe_trade_open(symbol: str, direction: int, track: str):
    """Observe trade opening"""
    dir_label = 'buy' if direction > 0 else 'sell'
    TRADES_OPENED.labels(symbol=symbol, dir=dir_label, track=track).inc()


def observe_trade_close(symbol: str, direction: int, track: str, r_multiple: float):
    """Observe trade closing with outcome"""
    dir_label = 'buy' if direction > 0 else 'sell'
    
    # Determine outcome
    if abs(r_multiple) < 1e-9:
        outcome = 'be'  # breakeven
    elif r_multiple > 0:
        outcome = 'win'
    else:
        outcome = 'loss'
    
    # Update counters
    TRADES_CLOSED.labels(symbol=symbol, dir=dir_label, track=track, outcome=outcome).inc()
    
    # Update histograms and summaries
    TRADE_R.observe(r_multiple)
    PAYOUT.observe(r_multiple)


# ===== Risk Management Metrics =====


def observe_risk_gate(gate_name: str, blocked: bool):
    """Observe risk gate decisions"""
    if blocked:
        observe_block(f"risk_{gate_name}")


def observe_position_management(action: str, symbol: str):
    """Observe position management actions"""
    # Could add specific metrics for TP-Split, Breakeven, Trailing Stop
    pass


# ===== Performance Metrics =====


def observe_decision_engine(ml_score: float, lstm_score: float, final_score: float):
    """Observe decision engine performance"""
    # Could add specific metrics for ML/LSTM performance
    pass


def observe_market_context(session: str, regime: str, volatility: float):
    """Observe market context changes"""
    # Could add specific metrics for market regime changes
    pass


# ===== AI Agent Metrics =====


def observe_agent_decision(action: str, confidence: float, status: str):
    """Observe AI agent decision review"""
    # This will be implemented when we add agent metrics
    pass


def observe_agent_intervention(action: str, confidence: str, reason: str):
    """Observe AI agent intervention"""
    # This will be implemented when we add agent metrics
    pass

# ===== Advanced Risk Analytics Metrics =====

def observe_risk_metric(metric_name: str, value: float):
    """Observe risk metric values"""
    try:
        risk_metric_gauge.labels(metric=metric_name).set(value)
        risk_metric_histogram.labels(metric=metric_name).observe(value)
    except Exception as e:
        logger.warning(f"Failed to observe risk metric {metric_name}: {e}")

def observe_risk_prediction(predicted_risk: float, actual_risk: float, accuracy: float):
    """Observe risk prediction accuracy"""
    try:
        risk_prediction_accuracy.observe(accuracy)
        risk_prediction_error.labels(type="absolute").observe(abs(predicted_risk - actual_risk))
        risk_prediction_error.labels(type="relative").observe(abs(predicted_risk - actual_risk) / max(actual_risk, 0.001))
    except Exception as e:
        logger.warning(f"Failed to observe risk prediction: {e}")

# Risk metrics
risk_metric_gauge = Gauge('mrben_risk_metric', 'Risk metric values', ['metric'])
risk_metric_histogram = Histogram('mrben_risk_metric_distribution', 'Risk metric distribution', ['metric'])
risk_prediction_accuracy = Histogram('mrben_risk_prediction_accuracy', 'Risk prediction accuracy')
risk_prediction_error = Histogram('mrben_risk_prediction_error', 'Risk prediction error', ['type'])

# ===== Advanced Position Management Metrics =====

def observe_position_metric(metric_name: str, value: float):
    """Observe position management metric values"""
    try:
        position_metric_gauge.labels(metric=metric_name).set(value)
        position_metric_histogram.labels(metric=metric_name).observe(value)
    except Exception as e:
        logger.warning(f"Failed to observe position metric {metric_name}: {e}")

def observe_position_adjustment(action: str, lot_change: float, confidence: float):
    """Observe position adjustment actions"""
    try:
        position_adjustment_counter.labels(action=action).inc()
        position_adjustment_lot_change.observe(lot_change)
        position_adjustment_confidence.observe(confidence)
    except Exception as e:
        logger.warning(f"Failed to observe position adjustment: {e}")

# Position management metrics
position_metric_gauge = Gauge('mrben_position_metric', 'Position management metric values', ['metric'])
position_metric_histogram = Histogram('mrben_position_metric_distribution', 'Position management metric distribution', ['metric'])
position_adjustment_counter = Counter('mrben_position_adjustment_total', 'Total position adjustments', ['action'])
position_adjustment_lot_change = Histogram('mrben_position_adjustment_lot_change', 'Position adjustment lot size changes')
position_adjustment_confidence = Histogram('mrben_position_adjustment_confidence', 'Position adjustment confidence levels')

# ===== Advanced Market Analysis Metrics =====

def observe_market_metric(metric_name: str, value: float):
    """Observe market analysis metric values"""
    try:
        market_metric_gauge.labels(metric=metric_name).set(value)
        market_metric_histogram.labels(metric=metric_name).observe(value)
    except Exception as e:
        logger.warning(f"Failed to observe market metric {metric_name}: {e}")

def observe_market_regime_change(old_regime: str, new_regime: str, confidence: float):
    """Observe market regime changes"""
    try:
        market_regime_change_counter.labels(old_regime=old_regime, new_regime=new_regime).inc()
        market_regime_confidence.observe(confidence)
    except Exception as e:
        logger.warning(f"Failed to observe market regime change: {e}")

# Market analysis metrics
market_metric_gauge = Gauge('mrben_market_metric', 'Market analysis metric values', ['metric'])
market_metric_histogram = Histogram('mrben_market_metric_distribution', 'Market analysis metric distribution', ['metric'])
market_regime_change_counter = Counter('mrben_market_regime_change_total', 'Total market regime changes', ['old_regime', 'new_regime'])
market_regime_confidence = Histogram('mrben_market_regime_confidence', 'Market regime change confidence levels')

# ===== Advanced Signal Generation Metrics =====

def observe_signal_metric(metric_name: str, value: float):
    """Observe signal generation metric values"""
    try:
        signal_metric_gauge.labels(metric=metric_name).set(value)
        signal_metric_histogram.labels(metric=metric_name).observe(value)
    except Exception as e:
        logger.warning(f"Failed to observe signal metric {metric_name}: {e}")

def observe_signal_quality(signal_type: str, quality_score: float, confidence: float):
    """Observe signal quality metrics"""
    try:
        signal_quality_gauge.labels(type=signal_type).set(quality_score)
        signal_confidence_gauge.labels(type=signal_type).set(confidence)
        signal_quality_histogram.labels(type=signal_type).observe(quality_score)
        signal_confidence_histogram.labels(type=signal_type).observe(confidence)
    except Exception as e:
        logger.warning(f"Failed to observe signal quality: {e}")

def observe_signal_fusion(method: str, quality_score: float, confidence: float):
    """Observe signal fusion performance"""
    try:
        signal_fusion_counter.labels(method=method).inc()
        signal_fusion_quality.observe(quality_score)
        signal_fusion_confidence.observe(confidence)
    except Exception as e:
        logger.warning(f"Failed to observe signal fusion: {e}")

# Signal generation metrics
signal_metric_gauge = Gauge('mrben_signal_metric', 'Signal generation metric values', ['metric'])
signal_metric_histogram = Histogram('mrben_signal_metric_distribution', 'Signal generation metric distribution', ['metric'])
signal_quality_gauge = Gauge('mrben_signal_quality', 'Signal quality scores', ['type'])
signal_confidence_gauge = Gauge('mrben_signal_confidence', 'Signal confidence levels', ['type'])
signal_quality_histogram = Histogram('mrben_signal_quality_distribution', 'Signal quality distribution', ['type'])
signal_confidence_histogram = Histogram('mrben_signal_confidence_distribution', 'Signal confidence distribution', ['type'])
signal_fusion_counter = Counter('mrben_signal_fusion_total', 'Total signal fusions', ['method'])
signal_fusion_quality = Histogram('mrben_signal_fusion_quality', 'Signal fusion quality scores')
signal_fusion_confidence = Histogram('mrben_signal_fusion_confidence', 'Signal fusion confidence levels')

# ===== Advanced Portfolio Management Metrics =====

def observe_portfolio_metric(metric_name: str, value: float):
    """Observe portfolio management metric values"""
    try:
        portfolio_metric_gauge.labels(metric=metric_name).set(value)
        portfolio_metric_histogram.labels(metric=metric_name).observe(value)
    except Exception as e:
        logger.warning(f"Failed to observe portfolio metric {metric_name}: {e}")

def observe_portfolio_allocation(strategy: str, method: str, confidence: float):
    """Observe portfolio allocation performance"""
    try:
        portfolio_allocation_counter.labels(strategy=strategy, method=method).inc()
        portfolio_allocation_confidence.observe(confidence)
    except Exception as e:
        logger.warning(f"Failed to observe portfolio allocation: {e}")

def observe_portfolio_risk(risk_type: str, value: float):
    """Observe portfolio risk metrics"""
    try:
        portfolio_risk_gauge.labels(type=risk_type).set(value)
        portfolio_risk_histogram.labels(type=risk_type).observe(value)
    except Exception as e:
        logger.warning(f"Failed to observe portfolio risk: {e}")

# Portfolio management metrics
portfolio_metric_gauge = Gauge('mrben_portfolio_metric', 'Portfolio management metric values', ['metric'])
portfolio_metric_histogram = Histogram('mrben_portfolio_metric_distribution', 'Portfolio management metric distribution', ['metric'])
portfolio_allocation_counter = Counter('mrben_portfolio_allocation_total', 'Total portfolio allocations', ['strategy', 'method'])
portfolio_allocation_confidence = Histogram('mrben_portfolio_allocation_confidence', 'Portfolio allocation confidence levels')
portfolio_risk_gauge = Gauge('mrben_portfolio_risk', 'Portfolio risk metric values', ['type'])
portfolio_risk_histogram = Histogram('mrben_portfolio_risk_distribution', 'Portfolio risk metric distribution', ['type'])


# ===== Utility Functions =====


def get_metrics_summary() -> dict:
    """Get a summary of current metrics"""
    return {
        "equity": EQUITY._value.get(),
        "balance": BALANCE._value.get(),
        "drawdown_pct": DRAWDOWN._value.get(),
        "exposure": EXPOSURE._value.get(),
        "spread": SPREAD._value.get(),
        "confidence": CONF_DYN._value.get(),
        "decision_score": SCORE._value.get(),
        "regime_code": REGIME._value.get(),
        "session_code": SESSION._value.get()
    }


def reset_metrics():
    """Reset all metrics (useful for testing)"""
    global _peak_equity
    _peak_equity = None
    
    # Reset all gauges
    for metric in [EQUITY, BALANCE, DRAWDOWN, EXPOSURE, SPREAD, CONF_DYN, SCORE, REGIME, SESSION]:
        metric._value.set(0.0)
    
    # Note: Counters, histograms, and summaries cannot be easily reset in prometheus_client
    # This is a limitation of the library
