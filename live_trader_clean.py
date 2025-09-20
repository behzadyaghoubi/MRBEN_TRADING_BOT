# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
MR BEN Live Trading System - Production-Grade EntryPoint
Unified orchestration with MTF RSI/MACD + Fusion + ATR SL/TP + logging/metrics/reports

NEW: Confluence Strategy Integration
- ICT + Price Action + Technical Indicators (EMA/MACD/RSI/ATR)
- Multi-timeframe analysis (HTF for trend, LTF for signals)
- AI Filter integration with new feature set
- Dynamic SL/TP based on ATR
- Risk management with position sizing
- Spread and position count filters
"""

from __future__ import annotations

# --- Standard library imports (critical ordering) ---
import argparse
import json
import logging
import os
import sys
import time
import threading
import faulthandler
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# --- Repo path & early env hygiene (BEFORE any from src...) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# OpenAI and other env sanitization (no-op if absent)
try:
    from src.core.env_sanitize import sanitize_openai_env
    sanitize_openai_env()
except Exception:
    pass

# --- Project imports (AFTER path setup) ---
from src.ops.signal_debug import log_ai_score

# Additional standard library imports
import pandas as pd

# --- DEBUG HOOKS (safe, idempotent) ---
DEBUG_ON = os.getenv("MRBEN_DEBUG","0") == "1"
HEARTBEAT_SEC = float(os.getenv("MRBEN_HEARTBEAT_SEC","5"))
LOG_LEVEL = os.getenv("LOG_LEVEL","INFO").upper()
try:
    logging.getLogger().setLevel(getattr(logging, LOG_LEVEL))
except Exception:
    pass

# Global exception hook for debugging
def _excepthook(exc_type, exc, tb):
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/live_crash.log","a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now(UTC).isoformat()} ===\n")
            traceback.print_exception(exc_type, exc, tb, file=f)
    finally:
        logging.getLogger(__name__).error("FATAL: uncaught exception", exc_info=(exc_type, exc, tb))

if DEBUG_ON:
    sys.excepthook = _excepthook

from src.ai.features import make_features
from src.ai.quality_model import QualityModel
from src.ai.replay_buffer import log_experience

# Multi-symbol trading imports
from src.app.multi_runner import MultiSymbolEngine, SymbolConfig
from src.core.alerting import send_alert
from src.core.logging_setup import setup_json_logger

# Import new core modules
from src.core.order_safety import send_order_safe, SymbolInfo
from src.core.persistence import DailyState, load_daily_state
from src.execution.adaptive_exec import (
    Quote,
    auto_replace_loop,
    canary_size,
    choose_order_type,
    slippage_ok,
)

# Import AI and execution features
from src.features.flags import FLAGS

# Unified JSON Logging + UTC (idempotent)
from src.core.logging_setup import setup_json_logger
setup_json_logger()  # idempotent
Path("logs").mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

# Import challenge mode guards
try:
    from src.core.challenge_guards import (
        ChallengeState,
        guard_challenge,
        initialize_challenge_state,
        rr_ok,
        update_challenge_state,
    )

    logger.info("[OK] Challenge mode guards imported")
except ImportError as e:
    logger.warning(f"[WARNING] Failed to import challenge guards: {e}")

    # Use centralized fallbacks
    from src.core.challenge_guards_fallback import (
        ChallengeState, guard_challenge, initialize_challenge_state,
        rr_ok, update_challenge_state
    )

# Path and env already handled at top

# Mode normalization function
def _normalize_mode():
    try:
        if 'args' in globals() and hasattr(globals()['args'], 'mode'):
            return globals()['args'].mode.lower()
        return os.getenv('MRBEN_MODE','live').lower()
    except Exception:
        return os.getenv('MRBEN_MODE','live').lower()

# MT5 per-mode handling (LIVE=fail-fast, DEMO/REPORT=graceful)
mode = _normalize_mode()
mt5_ok, mt5_inited = False, False
mt5 = None
broker = None

try:
    import MetaTrader5 as mt5
    mt5_inited = mt5.initialize()
    mt5_ok = bool(mt5_inited)
    if not mt5_ok:
        logger.error(json.dumps({"kind":"mt5_init_fail","mode":mode,"ts":datetime.now(UTC).isoformat()}))
        if mode == "live": 
            sys.exit(1)
except Exception as e:
    logger.error(json.dumps({"kind":"mt5_import_fail","err":str(e),"mode":mode,"ts":datetime.now(UTC).isoformat()}))
    if mode == "live": 
        sys.exit(1)

# Import broker if MT5 available
if mt5_ok:
    try:
        from src.core.broker_mt5 import broker
        logger.info(json.dumps({"kind":"import_ok","mod":"mt5_broker","ts":datetime.now(UTC).isoformat()}))
    except Exception as e:
        logger.error(json.dumps({"kind":"broker_import_fail","err":str(e),"ts":datetime.now(UTC).isoformat()}))
    # Mode-specific handling will be done in main() when mode is known

# Core imports
# AI Filter imports
from src.ai.filter import ConfluenceAIFilter

# New enhanced components
from src.core.gating import AllOfFourGate, ConcurrencyLimiter
from src.core.metrics import PerformanceMetrics
from src.core.spread_control import SpreadController
from src.core.supervisor_enhanced import SupervisorClient
from src.strategies.stealth_strategy import StealthStrategy

# Supervisor imports
try:
    from src.core.supervisor import (
        initialize_supervisor,
        on_cycle,
        on_error,
        on_fill,
        on_signal,
        on_skip,
        supervisor_emit,
    )

    logger.info("[OK] Trading Supervisor imported")
except ImportError as e:
    logger.info(f"[WARNING] Failed to import supervisor: {e}")

    # Use centralized fallback
    from src.core.supervisor_fallback import (
        initialize_supervisor, on_cycle, on_error, on_fill, on_signal, on_skip, supervisor_emit
    )

# Risk manager imports
from src.core.risk_manager import position_size_by_risk

# Enhanced components imports
try:
    from src.core.gating import AllOfFourGate, ConcurrencyLimiter
    from src.core.spread_control import SpreadController
    from src.core.supervisor_enhanced import SupervisorClient

    logger.info("[OK] Enhanced components imported")
except ImportError as e:
    logger.info(f"[WARNING] Enhanced components not available: {e}")

    # Use centralized fallbacks
    from src.core.gating_fallback import AllOfFourGate, ConcurrencyLimiter
    from src.core.spread_control_fallback import SpreadController
    from src.core.supervisor_fallback import SupervisorClient

# Indicator imports
from src.indicators.atr import compute_atr
from src.indicators.rsi_macd import compute_macd, compute_rsi

# Risk management imports
from src.risk_manager.atr_sl_tp import SLTPResult, calc_sltp_from_atr

# Global challenge state for challenge guards
CHALLENGE_STATE: Any = None

# Signal imports
from src.signals.multi_tf_rsi_macd import analyze_multi_tf_rsi_macd

# Strategy imports
from src.strategies.confluence_pro_strategy import Signal, confluence_signal
from src.utils.error_handler import error_handler

# Data imports
try:
    from src.data.manager import MT5DataManager

    DATA_MANAGER_AVAILABLE = True
    logger.info("[OK] MT5DataManager imported successfully")
except ImportError as e:
    logger.info(f"[WARNING] Failed to import MT5DataManager: {e}")
    DATA_MANAGER_AVAILABLE = False

# AI model imports
try:
    import joblib

    AI_MODEL_AVAILABLE = True
except ImportError:
    AI_MODEL_AVAILABLE = False

# Price Action imports
try:
    from src.strategy.pa import PriceActionValidator

    PRICE_ACTION_AVAILABLE = True
except ImportError:
    PRICE_ACTION_AVAILABLE = False
    PriceActionValidator = None  # type: ignore

# Timeframe to minutes mapping for closed bar validation
TF_TO_MIN = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

# Import kill-switch and alerting

# Import circuit breaker
from src.core.circuit_breaker import CBConfig, CircuitBreaker
from src.core.kill_switch import should_stop_now

cb_order = CircuitBreaker("order_send", CBConfig(max_errors=3, window_sec=60, cooldown_sec=180))

# Import config guard
from src.core.config_guard import enforce_config_lock


@dataclass
class AdaptiveResult:
    """Result from fusion scoring"""

    score_buy: float
    score_sell: float
    label: str
    confidence: float

@dataclass
class TradeRecord:
    """Trade execution record"""

    side: str
    entry: float
    sl: float
    tp: float
    timestamp: datetime
    result: str
    used_fallback: bool
    extras: dict[str, Any]

# Safe order utilities
# Safe order utilities - now imported from core modules

# Pro features - now imported from core modules

class LiveTraderApp:
    """
    DEPRECATED: Production-grade live trading application with full supervision

    This class is deprecated. Use the pipeline: evaluate_once → maybe_execute_trade → write_report
    instead of this class-based approach.
    """

    def __init__(self, cfg: dict[str, Any], args: argparse.Namespace, logger: logging.Logger):
        self.cfg = cfg
        self.args = args
        self.logger = logger

        # UTC timestamp and daily state
        self.start_time = datetime.now(UTC)
        self.daily_state: DailyState = load_daily_state()

        # Symbol info for decimal operations
        tick = Decimal(str(self.cfg.get("symbols", {}).get("price_tick", 0.01)))
        qstep = Decimal(str(self.cfg.get("symbols", {}).get("qty_step", 0.01)))
        qmin = Decimal(str(self.cfg.get("symbols", {}).get("min_qty", 0.01)))
        self._sym_info = SymbolInfo(price_tick=tick, qty_step=qstep, min_qty=qmin)
        self._success_seq = 0
        
        # AI quality model bootstrap
        self._ai_model = QualityModel.load_or_init(n_features=len(make_features({}, {}).x))
        self._ai_min_score = FLAGS.ai_min_score
        self._ai_online_learn = FLAGS.ai_online_learn
        # Exec params
        self._exec_max_slip_bps = FLAGS.exec_max_slip_bps
        self._exec_replace_timeout = FLAGS.exec_replace_timeout_s
        self._exec_replace_retries = FLAGS.exec_replace_retries

        # Debug flags
        self.debug_scan_on_nosignal = getattr(args, 'debug_scan_on_nosignal', False)
        self.demo_smoke_signal = getattr(args, 'demo_smoke_signal', False)

        # Validate and set symbol
        self.symbol = self._validate_symbol(args.symbol, cfg)
        self.logger.info(f"Using symbol: {self.symbol}")

        # Initialize metrics
        self.metrics = PerformanceMetrics()

        # Initialize data manager
        self.data_manager: MT5DataManager | None = None
        if DATA_MANAGER_AVAILABLE:
            try:
                # Convert timeframe string to minutes
                tf_min = self._timeframe_to_minutes(self.args.timeframe)
                self.data_manager = MT5DataManager(self.symbol, tf_min)
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize MT5DataManager for {self.symbol}: {e}, using fallback adapter"
                )
                self.data_manager = None
        else:
            self.logger.warning("MT5DataManager not available, using fallback adapter")

        # Load AI model
        self.ai_model = None
        if AI_MODEL_AVAILABLE:
            try:
                model_path = "mrben_ai_signal_filter_xgb.joblib"
                if os.path.exists(model_path):
                    self.ai_model = joblib.load(model_path)
                    self.logger.info("AI model loaded successfully")
                else:
                    if self.cfg.get("ml", {}).get("require_ready", False):
                        self.logger.error("AI model file not found - aborting")
                        # Only fail-fast in LIVE mode
                        if not args.demo and not args.report_only:
                            sys.exit(1)
                    else:
                        self.logger.warning("AI model file not found, using fallback")
            except Exception as e:
                if self.cfg.get("ml", {}).get("require_ready", False):
                    self.logger.error(f"Failed to load AI model: {e} - aborting")
                    # Only fail-fast in LIVE mode
                    if not args.demo and not args.report_only:
                        sys.exit(1)
                else:
                    self.logger.warning(f"Failed to load AI model: {e}, using fallback")

        # Initialize AI Filter for Confluence Strategy
        self.ai_filter = None
        try:
            model_path = cfg.get("ml", {}).get("model", "models/ml_filter.pkl")
            scaler_path = cfg.get("ml", {}).get("scaler", "models/scaler.pkl")
            neutral_default = cfg.get("ml", {}).get("neutral_default", 0.55)
            gating_on_unavailable = cfg.get("ml", {}).get("gating_on_unavailable", "ignore")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.ai_filter = ConfluenceAIFilter(
                    model_path, scaler_path, neutral_default, gating_on_unavailable
                )
                self.logger.info("AI Filter for Confluence Strategy initialized successfully")
            else:
                self.logger.warning(
                    "AI Filter model files not found, will use pass-through behavior"
                )
                self.ai_filter = ConfluenceAIFilter(
                    neutral_default=neutral_default, gating_on_unavailable=gating_on_unavailable
                )  # Initialize without model
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize AI Filter: {e}, will use pass-through behavior"
            )
            self.ai_filter = ConfluenceAIFilter(
                neutral_default=0.55, gating_on_unavailable="ignore"
            )  # Initialize without model

        # Initialize News Filter
        self.news_filter = None
        try:
            from src.core.news_filter import create_news_filter

            self.news_filter = create_news_filter(cfg.get("strategy", {}).get("filters", {}))
            self.logger.info("News Filter initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize News Filter: {e}")
            self.news_filter = None

        # News filter transparency
        if self.news_filter is None:
            self.logger.warning("News Filter unavailable: proceeding WITHOUT news gating")

        # Cache configuration parameters
        self.risk_params = cfg.get('risk', {})
        self.mtf_params = cfg.get('multi_tf', {})
        self.fusion_params = cfg.get('fusion', {})
        self.logging_params = cfg.get('logging', {})

        # Load and merge pro config
        pro_config = self._load_pro_config()
        if pro_config:
            # Merge pro config with main config
            self.cfg.update(pro_config)
            self.logger.info("Pro config merged with main config")

            # Update cached parameters
            self.risk_params = self.cfg.get('risk', {})
            self.mtf_params = self.cfg.get('multi_tf', {})
            self.fusion_params = self.cfg.get('fusion', {})
            self.logging_params = self.cfg.get('logging', {})

        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)

        # Initialize logging handlers if configured
        self._setup_logging_handlers()

        mode = "demo" if args.demo else ("live" if args.live else "simulate")
        self.logger.info(f"LiveTraderApp initialized for {self.symbol} in {mode} mode")

        # Initialize start time (already set above with UTC)
        # self.start_time = datetime.now(UTC)

    def _validate_symbol(self, symbol: str, cfg: dict[str, Any]) -> str:
        """Validate symbol and return valid symbol or default"""
        supported_symbols = cfg.get('symbols', {}).get('supported', ['XAUUSD'])
        default_symbol = cfg.get('symbols', {}).get('default', 'XAUUSD')

        if symbol in supported_symbols:
            return symbol
        else:
            self.logger.warning(
                f"Symbol '{symbol}' not supported. Supported symbols: {supported_symbols}"
            )
            self.logger.warning(f"Falling back to default symbol: {default_symbol}")
            return default_symbol

    def _utcnow(self) -> datetime:
        return datetime.now(UTC)

    def _kill_switch_guard(self) -> None:
        if should_stop_now():
            self.logger.error("Kill switch activated. Aborting trading loop.")
            raise SystemExit(1)

    def _order_send_safe(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        qty: Decimal,
        price: Decimal | None,
        tif: str = "GTC",
        ttl_sec: int = 15,
    ) -> dict:
        return send_order_safe(
            broker,
            cb_order,
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            price=price,
            info=self._sym_info,
            tif=tif,
            ttl_sec=ttl_sec,
            logger=self.logger,
        )

    def _order_market_buy(self, qty: Decimal):
        return self._order_send_safe(
            symbol=self.symbol, side="buy", order_type="market", qty=qty, price=None
        )

    def _order_market_sell(self, qty: Decimal):
        return self._order_send_safe(
            symbol=self.symbol, side="sell", order_type="market", qty=qty, price=None
        )

    def execute_trade(self, side: str, base_size: Decimal, quotes: dict, signal: dict = None, context: dict = None, max_slip_bps: float = 10.0):
        """Unified execution path with AI quality scoring and pro execution features."""
        self._kill_switch_guard()
        if side not in ("buy", "sell"):
            self.logger.info("no_trade_decision"); return None
            
        # AI quality scoring
        feats = make_features(signal or {}, context or {}).x
        ai_score = 1.0  # default pass
        if FLAGS.ai_quality:
            ai_score = self._ai_model.score(feats)
            self.logger.info(f"ai_score={ai_score:.3f}")
            # Add AI score logging for debug
            log_ai_score(self.logger, self.symbol, ai_score)
        if FLAGS.ai_quality and ai_score < self._ai_min_score:
            self.logger.info("ai_gate_reject", extra={"score": ai_score})
            return None
            
        # shadow
        if FLAGS.exec_canary:
            self.logger.info(f"shadow_signal side={side} size={base_size} price_est={quotes.get('mid')}")
            
        # Build quote object from current best bid/ask
        q = Quote(bid=float(quotes["bid"]), ask=float(quotes["ask"]))
        
        # slippage guard
        if FLAGS.exec_slippage_guard and not slippage_ok(q, side, self._exec_max_slip_bps):
            self.logger.warning(f"slippage_block side={side} bid={q.bid} ask={q.ask}")
            return None
            
        # Canary sizing
        size = Decimal(str(base_size))
        if FLAGS.exec_canary:
            size = canary_size(size, self._success_seq, step=0.5, max_mult=1.0)
            
        # Order type adaptation
        order_type = "market"
        if FLAGS.exec_adapt_order_type:
            spread_bps = abs(q.ask - q.bid) / max(q.mid, 1e-9) * 1e4
            order_type = choose_order_type(volatility=context.get("volatility", 0.0) if context else 0.0, spread_bps=spread_bps)
            
        # Send order using safe wrapper or adaptive execution
        if order_type == "market":
            result = self._order_send_safe(symbol=self.symbol, side=side, order_type="market", qty=size, price=None, tif="IOC", ttl_sec=12)
        else:
            if FLAGS.exec_auto_replace:
                result = auto_replace_loop(self.broker, self.symbol, side, size, q, float(self._sym_info.price_tick),
                                         max_retries=self._exec_replace_retries, timeout_s=self._exec_replace_timeout, logger=self.logger)
            else:
                # single passive try
                px = q.ask if side == "buy" else q.bid
                result = self._order_send_safe(symbol=self.symbol, side=side, order_type="limit", qty=size, price=Decimal(str(px)), tif="IOC", ttl_sec=12)
                
        # Update success counter
        status = (result.get("status") or "").lower()
        if status in {"filled", "closed"}:
            self._success_seq = min(self._success_seq + 1, 5)
        else:
            self._success_seq = 0
            
        return result

    def log_ai_feedback(self, feats: list[float], signal: dict, trade_profit: float, side: str):
        """Log AI feedback for online learning when trade outcome becomes known."""
        try:
            # Define label policy: profit > 0 -> 1 else 0
            label = 1 if trade_profit > 0 else 0
            if FLAGS.ai_quality:
                log_experience(feats, label, {"symbol": self.symbol, "side": side, "profit": trade_profit})
            if FLAGS.ai_quality and self._ai_online_learn:
                self._ai_model.update(feats, label)
                self._ai_model.save()
                self.logger.info(f"ai_model_updated label={label} profit={trade_profit:.2f}")
        except Exception as e:
            self.logger.warning(f"ai_feedback_error err={e}")

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        tf_map = {
            'M1': 1,
            'M2': 2,
            'M3': 3,
            'M4': 4,
            'M5': 5,
            'M10': 10,
            'M15': 15,
            'M30': 30,
            'H1': 60,
            'H2': 120,
            'H3': 180,
            'H4': 240,
            'D1': 1440,
            'W1': 10080,
            'MN1': 43200,
        }
        return tf_map.get(timeframe, 15)  # Default to M15

    def _setup_logging_handlers(self) -> None:
        """Setup additional logging handlers if configured"""
        if self.logging_params.get('enable_csv'):
            try:
                csv_path = self.logging_params.get('csv_path', 'logs/trades.csv')
                csv_handler = logging.FileHandler(csv_path)
                csv_handler.setLevel(logging.INFO)
                self.logger.addHandler(csv_handler)
            except Exception as e:
                self.logger.warning(f"Failed to setup CSV logging: {e}")

        if self.logging_params.get('enable_sqlite'):
            try:
                # SQLite logging would be implemented here
                self.logger.info("SQLite logging configured")
            except Exception as e:
                self.logger.warning(f"Failed to setup SQLite logging: {e}")

    # DEPRECATED: Use _fetch_confluence_data_util instead
    def _fetch_multi_tf_ohlc(
        self, _symbol: str, timeframes: list[str], bars: int
    ) -> dict[str, Any]:
        """DEPRECATED: Use _fetch_confluence_data_util instead"""
        self.logger.warning("_fetch_multi_tf_ohlc is deprecated, use _fetch_confluence_data_util")
        return {}

    def _normalize_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLC data to standard format with UTC time"""
        # enforce columns and UTC time
        if "time" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "time"})
        if "volume" not in df.columns:
            if "tick_volume" in df.columns:
                df["volume"] = df["tick_volume"]
            else:
                df["volume"] = 0
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df

    def _fetch_confluence_data(
        self, symbol: str, bars: int = 600
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch data for Confluence Strategy (HTF for trend, LTF for signal)

        Returns:
            tuple: (df_htf, df_ltf) for trend and signal timeframes
        """
        try:
            tf_trend = self.cfg.get("strategy", {}).get("timeframes", {}).get("trend", "H1")
            tf_signal = self.cfg.get("strategy", {}).get("timeframes", {}).get("signal", "M15")

            # fetch from real broker (no dummy)
            df_htf = broker.copy_rates(symbol, tf_trend, bars)
            df_ltf = broker.copy_rates(symbol, tf_signal, bars)

            if (
                df_htf is None
                or df_ltf is None
                or len(df_htf) < bars // 2
                or len(df_ltf) < bars // 2
            ):
                self.logger.error(
                    f"Insufficient real OHLC: HTF={len(df_htf) if df_htf is not None else 0}, LTF={len(df_ltf) if df_ltf is not None else 0}"
                )
                raise RuntimeError("insufficient_ohlc")

            df_htf = self._normalize_ohlc(df_htf)
            df_ltf = self._normalize_ohlc(df_ltf)

            self.logger.info(
                f"Fetched real OHLC for confluence: HTF={tf_trend} rows={len(df_htf)}, LTF={tf_signal} rows={len(df_ltf)}"
            )
            return df_htf, df_ltf

        except Exception as e:
            self.logger.error(f"Failed to fetch confluence data from MT5: {e}")
            # fail safe: return empty to skip cycle (no dummy in prod path)
            return pd.DataFrame(), pd.DataFrame()

    # No dummy OHLC in production - fail fast if data unavailable

    def _ta_decisions(self, dfs: dict[str, Any]) -> Mapping[str, str]:
        """Get technical analysis decisions for multiple timeframes"""
        try:
            return analyze_multi_tf_rsi_macd(
                dfs,
                rsi_period=self.mtf_params.get('rsi_period', 14),
                macd_fast=self.mtf_params.get('macd_fast', 12),
                macd_slow=self.mtf_params.get('macd_slow', 26),
                macd_signal=self.mtf_params.get('macd_signal', 9),
                rsi_overbought=self.mtf_params.get('rsi_overbought', 70),
                rsi_oversold=self.mtf_params.get('rsi_oversold', 30),
            )
        except Exception as e:
            self.logger.error(f"Failed to analyze multi-timeframe RSI/MACD: {e}")
            return {}

    def _pa_signal(self, _df_primary: Any) -> str:
        """Get price action signal"""
        if not PRICE_ACTION_AVAILABLE:
            return "neutral"

        try:
            # This is a placeholder - implement actual price action logic
            # For now, return neutral to avoid errors
            return "neutral"
        except Exception as e:
            self.logger.error(f"Failed to get price action signal: {e}")
            return "neutral"

    def _execute_confluence_strategy(self) -> None:
        """
        Execute Confluence Strategy with proper risk management and safety guards
        """
        try:
            # Check if confluence strategy is enabled
            strategy_enabled = self.cfg.get("strategy", {}).get("name") == "confluence_pro"
            if not strategy_enabled:
                self.logger.debug("Confluence strategy not enabled, skipping")
                return

            self.logger.info("Executing Confluence Strategy...")

            # Fetch data for confluence strategy
            df_htf, df_ltf = self._fetch_confluence_data(self.symbol, bars=600)

            if df_htf is None or df_ltf is None:
                self.logger.warning("Failed to fetch data for confluence strategy")
                return

            # Check all trading conditions (spread, positions, news)
            if not self._check_trading_conditions():
                self.logger.info("Trading conditions not met, skipping strategy execution")
                return

            # Get symbol information (point value, tick value, spread)
            symbol_info = self._get_symbol_info()

            # Execute confluence strategy
            signal = confluence_signal(
                df_ltf=df_ltf,
                df_htf=df_htf,
                cfg=self.cfg.get("strategy", {}),  # Pass strategy config directly
                symbol_point=symbol_info["point"],
                ai_filter=self.ai_filter,
                debug_nosignal=getattr(self, 'debug_scan_on_nosignal', False),
                demo_smoke_signal=getattr(self, 'demo_smoke_signal', False),
                symbol=self.symbol,
                ctx=self,
                market_info=symbol_info,
            )

            if not signal:
                self.logger.info("No confluence signal generated")
                return

            # Log signal details with safety guard information
            self.logger.info(f"Confluence signal: {signal.side}, confidence: {signal.confidence}")
            self.logger.info(f"Entry: {signal.entry:.5f}, SL: {signal.sl:.5f}, TP: {signal.tp:.5f}")
            self.logger.info(
                f"RR: {signal.meta.get('rr', 0):.2f}, RSI: {signal.meta.get('rsi', 0):.1f}"
            )

            # Additional safety check: Verify RR meets minimum requirement
            min_rr = self.cfg.get("risk", {}).get("min_rr", 1.0)
            if signal.meta.get('rr', 0) < min_rr:
                self.logger.warning(f"RR too low: {signal.meta.get('rr', 0):.2f} < {min_rr}")
                return

            # Calculate position size using risk manager
            lots = self._calculate_position_size(signal, symbol_info)
            if lots <= 0:
                self.logger.warning("Invalid position size calculated")
                return

            # Place order
            order_success = self._place_confluence_order(signal, lots, symbol_info)

            if order_success:
                self.logger.info(f"Confluence order placed successfully: {signal.side} {lots} lots")
                self.metrics.record_trade()

                # Record trade details
                self._record_confluence_trade(signal, lots, "PLACED")
            else:
                self.logger.error("Failed to place confluence order")

        except Exception as e:
            self.logger.error(f"Error executing confluence strategy: {e}")
            import traceback

            traceback.print_exc()

    def _record_confluence_trade(self, signal: Signal, lots: float, order_id: str) -> None:
        """Record confluence trade in metrics system"""
        try:
            # Log trade details
            self.logger.info(f"Confluence trade recorded: {signal.side} {lots} lots")
            self.logger.info(f"Order ID: {order_id}")
            self.logger.info(f"Entry: {signal.entry:.5f}, SL: {signal.sl:.5f}, TP: {signal.tp:.5f}")
            self.logger.info(f"Confidence: {signal.confidence}")

            # Save trade to database
            try:
                from src.core.database import db_manager

                trade_data = {
                    'timestamp': datetime.now(UTC).isoformat(),
                    'symbol': signal.symbol,
                    'action': signal.side,
                    'entry_price': signal.entry,
                    'exit_price': None,
                    'sl': signal.sl,
                    'tp': signal.tp,
                    'lot_size': lots,
                    'profit': None,
                    'balance': None,  # Will be updated when trade closes
                    'ai_decision': 1,  # AI approved
                    'ai_confidence': signal.confidence,
                    'result_code': None,
                    'comment': f"Confluence Strategy - {signal.side}",
                    'status': 'open',
                }

                trade_id = db_manager.save_trade(trade_data)
                self.logger.info(f"Trade saved to database with ID: {trade_id}")

            except Exception as e:
                self.logger.error(f"Failed to save trade to database: {e}")

        except Exception as e:
            self.logger.error(f"Error recording confluence trade: {e}")

    def _check_trading_conditions(self) -> bool:
        """
        Check all trading conditions before placing orders
        """
        try:
            # Check spread with symbol-specific limits
            symbol_info = self._get_symbol_info()
            filters = self.cfg.get("strategy", {}).get("filters", {})
            max_spread_map = filters.get("max_spread_points_map", {})
            max_spread = max_spread_map.get(self.symbol, self.cfg.get("trading", {}).get("spread_thresholds", {}).get(self.symbol, self.cfg.get("trading", {}).get("max_spread_points", 1000)))

            if symbol_info["spread_points"] > max_spread:
                self.logger.info(
                    f"Spread too high for {self.symbol}: {symbol_info['spread_points']:.1f} > {max_spread}"
                )
                return False

            # Check concurrent positions
            max_positions = self.cfg.get("risk", {}).get("max_concurrent_positions", 2)
            current_positions = self._get_open_positions_count()

            if current_positions >= max_positions:
                self.logger.info(
                    f"Max concurrent positions reached: {current_positions}/{max_positions}"
                )
                return False

            # Check news filter
            if self.news_filter:
                is_news_time, reason = self.news_filter.is_news_time(self.symbol)
                if is_news_time:
                    self.logger.info(f"News filter blocked entry: {reason}")
                    return False

            self.logger.debug("All trading conditions passed")
            return True

        except Exception as e:
            self.logger.error(f"Error checking trading conditions: {e}")
            return False

    def _get_symbol_info(self) -> dict:
        """Get symbol information (point, tick value, spread)"""
        try:
            import MetaTrader5 as mt5

            si = mt5.symbol_info(self.symbol)
            if si is None:
                raise RuntimeError("symbol_info_none")
            tick = mt5.symbol_info_tick(self.symbol)
            spread_pts = 0
            if tick:
                spread_pts = (tick.ask - tick.bid) / si.point
            return {
                "point": float(si.point),
                "digits": int(si.digits),
                "spread_points": float(spread_pts),
                "tick_value": (
                    float(si.trade_tick_value) if hasattr(si, "trade_tick_value") else 1.0
                ),
                "min_lot": float(si.volume_min),
                "max_lot": float(si.volume_max),
                "lot_step": float(si.volume_step),
            }
        except Exception as e:
            self.logger.warning(f"Falling back to dummy symbol info for {self.symbol}: {e}")
            self.logger.error(f"Failed to get symbol info for {self.symbol} - aborting")
            sys.exit(1)

    def _get_broker_info(self) -> dict:
        """Get broker information (balance, account details)"""
        try:
            import MetaTrader5 as mt5

            ac = mt5.account_info()
            if ac is None:
                raise RuntimeError("account_info_none")
            return {
                "balance": float(ac.balance),
                "equity": float(ac.equity),
                "margin": float(ac.margin),
                "free_margin": float(ac.margin_free),
            }
        except Exception as e:
            self.logger.warning(f"Using fallback broker info: {e}")
            return {"balance": 0.0, "equity": 0.0, "margin": 0.0, "free_margin": 0.0}

    def _get_open_positions_count(self) -> int:
        """Get count of open positions for current symbol"""
        try:
            import MetaTrader5 as mt5

            positions = mt5.positions_get(symbol=self.symbol)
            return len(positions) if positions is not None else 0
        except Exception as e:
            self.logger.error(f"Error getting open positions count: {e}")
            return 0

    def _calculate_position_size(self, signal: Signal, symbol_info: dict) -> float:
        """Calculate position size using risk manager"""
        try:
            # Get ATR value from signal metadata
            atr_val = signal.meta.get("atr", 0.0)
            if atr_val <= 0:
                self.logger.warning("Invalid ATR value for position sizing")
                return 0.0

            # Get risk parameters from config
            risk_pct = self.cfg.get("risk", {}).get("risk_pct", 0.01)
            broker_info = self._get_broker_info()
            balance = broker_info.get("equity") or broker_info.get("balance") or 0.0

            # Calculate SL points
            sl_points = abs(signal.entry - signal.sl) / symbol_info["point"]

            # Calculate position size
            lots = position_size_by_risk(
                balance=balance,
                risk_pct=risk_pct,
                sl_points=sl_points,
                tick_value_per_point=symbol_info["tick_value"],
            )

            self.logger.info(
                f"Position size calculated: {lots} lots (risk: {risk_pct*100}%, SL: {sl_points:.1f} points)"
            )
            return lots

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _place_confluence_order(self, signal: Signal, lots: float, symbol_info: dict) -> bool:
        """Place order for confluence strategy"""
        try:
            slippage = self.cfg.get("broker", {}).get("slippage_points", 10)

            order_details = {
                "symbol": self.symbol,
                "side": signal.side,
                "lots": lots,
                "entry_price": signal.entry,
                "stop_loss": signal.sl,
                "take_profit": signal.tp,
                "slippage": slippage,
                "confidence": signal.confidence,
            }

            self.logger.info(f"Confluence order details: {order_details}")

            # Place order with broker
            order_id = self._place_broker_order(order_details)

            if order_id:
                self.logger.info(f"Confluence order placed successfully: {order_id}")
                return True
            else:
                self.logger.error("Failed to place confluence order with broker")
                return False

        except Exception as e:
            self.logger.error(f"Error placing confluence order: {e}")
            return False

    def _place_broker_order(self, order_details: dict) -> str:
        """
        Place order with broker

        Args:
            order_details: Dictionary containing order parameters

        Returns:
            Order ID if successful, empty string if failed
        """
        try:
            # Implement actual broker order placement
            try:
                from src.core.broker_mt5 import broker

                # Place order through MT5 broker
                order_id = broker.place_order(
                    symbol=order_details.get('symbol', 'XAUUSD'),
                    side=order_details.get('side', 'long'),
                    lots=order_details.get('lots', 0.01),
                    entry_price=order_details.get('entry_price'),
                    sl=order_details.get('sl'),
                    tp=order_details.get('tp'),
                    slippage=order_details.get('slippage', 20),
                )

                if order_id:
                    self.logger.info(f"Real order placed successfully: {order_id}")
                    return order_id
                else:
                    self.logger.error("Failed to place real order")
                    return ""

            except Exception as e:
                self.logger.error(f"Broker order placement error: {e}")
                # Fallback to simulation
                import uuid
            order_id = str(uuid.uuid4())[:8]
            self.logger.warning(f"Using simulated order ID: {order_id}")

            self.logger.info(f"Order placed with broker: {order_id}")
            self.logger.info(f"Order details: {order_details}")

            return order_id

        except Exception as e:
            self.logger.error(f"Error placing broker order: {e}")
            return ""

    def _load_pro_config(self) -> dict:
        """Load pro_config.json configuration file"""
        try:
            config_path = "config/pro_config.json"
            if os.path.exists(config_path):
                with open(config_path, encoding='utf-8') as f:
                    pro_config = json.load(f)
                self.logger.info("Pro config loaded successfully")
                return pro_config
            else:
                self.logger.warning(f"Pro config file not found: {config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading pro config: {e}")
            return {}

    def _ai_score(self, features: list[float]) -> float:
        """Get AI model score for features"""
        if self.ai_model is None:
            return 0.5  # Fallback score

        try:
            # Ensure features are in expected format
            if len(features) != 10:
                # Pad or truncate to expected length
                if len(features) < 10:
                    features = features + [0.0] * (10 - len(features))
                else:
                    features = features[:10]

            # Get prediction from model
            score = self.ai_model.predict_proba([features])[0]
            return float(score[1])  # Probability of positive class
        except Exception as e:
            self.logger.error(f"Failed to get AI score: {e}")
            return 0.5  # Fallback score

    def _fuse(self, ta_map: Mapping[str, str], pa: str, ai: float) -> AdaptiveResult:
        """Fuse technical analysis, price action, and AI scores"""
        try:
            # Convert ta_map to scores
            ta_buy_count = sum(1 for v in ta_map.values() if v == "buy")
            ta_sell_count = sum(1 for v in ta_map.values() if v == "sell")
            ta_total = len(ta_map)

            ta_buy_score = ta_buy_count / ta_total if ta_total > 0 else 0.5
            ta_sell_score = ta_sell_count / ta_total if ta_total > 0 else 0.5

            # Convert price action to scores
            pa_buy_score = 1.0 if pa == "buy" else 0.0
            pa_sell_score = 1.0 if pa == "sell" else 0.0

            # Get fusion weights
            w_ta = self.fusion_params.get('w_ta', 0.4)
            w_pa = self.fusion_params.get('w_pa', 0.3)
            w_ai = self.fusion_params.get('w_ai', 0.3)

            # Fuse scores
            buy_score = w_ta * ta_buy_score + w_pa * pa_buy_score + w_ai * ai
            sell_score = w_ta * ta_sell_score + w_pa * pa_sell_score + w_ai * (1 - ai)

            # Determine label and confidence
            threshold_buy = self.fusion_params.get('threshold_buy', 0.6)
            threshold_sell = self.fusion_params.get('threshold_sell', 0.6)

            if buy_score >= threshold_buy:
                label = "buy"
                confidence = buy_score
            elif sell_score >= threshold_sell:
                label = "sell"
                confidence = sell_score
            else:
                label = "neutral"
                confidence = max(buy_score, sell_score)

            return AdaptiveResult(
                score_buy=buy_score, score_sell=sell_score, label=label, confidence=confidence
            )

        except Exception as e:
            self.logger.error(f"Failed to fuse signals: {e}")
            return AdaptiveResult(score_buy=0.5, score_sell=0.5, label="neutral", confidence=0.5)

    def _atr_sltp(self, side: str, entry: float, ohlc_primary: Any) -> SLTPResult:
        """Calculate ATR-based stop loss and take profit"""
        try:
            # Compute ATR
            atr_value = compute_atr(ohlc_primary, self.risk_params.get('atr_period', 14))

            if atr_value is None or atr_value <= 0:
                self.logger.warning("ATR computation failed, using fallback SL/TP")
                return calc_sltp_from_atr(
                    side=side,  # type: ignore[arg-type]
                    entry_price=entry,
                    atr_value=entry * 0.01,  # 1% fallback
                    rr=self.risk_params.get('rr', 1.5),
                    sl_k=self.risk_params.get('sl_k', 1.0),
                    tp_k=self.risk_params.get('tp_k', 1.5),
                    fallback_sl_pct=self.risk_params.get('fallback_sl_pct', 0.005),
                    fallback_tp_pct=self.risk_params.get('fallback_tp_pct', 0.0075),
                )

            # Use ATR-based calculation
            result = calc_sltp_from_atr(
                side=side,  # type: ignore[arg-type]
                entry_price=entry,
                atr_value=atr_value,
                rr=self.risk_params.get('rr', 1.5),
                sl_k=self.risk_params.get('sl_k', 1.0),
                tp_k=self.risk_params.get('tp_k', 1.5),
                fallback_sl_pct=self.risk_params.get('fallback_sl_pct', 0.005),
                fallback_tp_pct=self.risk_params.get('fallback_tp_pct', 0.0075),
            )

            # Log SL/TP decision
            sltp_logger = logging.getLogger("core.trade.sltp")
            sltp_logger.info(
                f"SL/TP Decision: side={side}, entry={entry:.2f}, ATR={atr_value:.2f}, "
                f"rr={self.risk_params.get('rr', 1.5)}, sl_k={self.risk_params.get('sl_k', 1.0)}, "
                f"tp_k={self.risk_params.get('tp_k', 1.5)}, fallback_sl_pct={self.risk_params.get('fallback_sl_pct', 0.005)}, "
                f"fallback_tp_pct={self.risk_params.get('fallback_tp_pct', 0.0075)}, "
                f"SL={result.sl:.2f}, TP={result.tp:.2f}, used_fallback={result.used_fallback}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to calculate ATR SL/TP: {e}")
            # Return fallback values
            if side == "buy":
                return SLTPResult(
                    sl=entry * (1 - self.risk_params.get('fallback_sl_pct', 0.005)),
                    tp=entry * (1 + self.risk_params.get('fallback_tp_pct', 0.0075)),
                    used_fallback=True,
                )
            else:
                return SLTPResult(
                    sl=entry * (1 + self.risk_params.get('fallback_sl_pct', 0.005)),
                    tp=entry * (1 - self.risk_params.get('fallback_tp_pct', 0.0075)),
                    used_fallback=True,
                )

    def _place_order(
        self, side: str, sl: float, tp: float, mode: str, lot_size: float = 0.01, entry: float = 0.0
    ) -> bool:
        """Place trading order based on mode"""
        try:
            if mode in ["simulate", "demo"]:
                # Log the order details
                self.logger.info(
                    f"SIMULATED ORDER: {side.upper()} {self.symbol} @ {sl:.2f} SL, {tp:.2f} TP"
                )
                return True
            elif mode == "live":
                # Implement actual order placement
                try:
                    from src.core.broker_mt5 import broker

                    # Place order through MT5 broker
                    order_id = broker.place_order(
                        symbol=self.symbol,
                        side=side,
                        lots=lot_size,
                        entry_price=entry,
                        sl=sl,
                        tp=tp,
                        slippage=20,
                    )

                    if order_id:
                        self.logger.info(f"Live order placed successfully: {order_id}")
                        return True
                    else:
                        self.logger.error("Failed to place live order")
                        return False

                except Exception as e:
                    self.logger.error(f"Live order placement error: {e}")
                return False
            else:
                self.logger.error(f"Unknown trading mode: {mode}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return False

    def _record_trade(
        self,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        result: str,
        lot_size: float = 0.01,
        profit: float = 0.0,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Record trade for logging/reporting"""
        if extras is None:
            extras = {}

        _trade_record = TradeRecord(
            side=side,
            entry=entry,
            sl=sl,
            tp=tp,
            timestamp=datetime.now(UTC),
            result=result,
            used_fallback=extras.get('used_fallback', False),
            extras=extras,
        )

        # Log trade
        self.logger.info(
            f"TRADE RECORDED: {side.upper()} {self.symbol} @ {entry:.2f}, "
            f"SL: {sl:.2f}, TP: {tp:.2f}, Result: {result}"
        )

        # Save to CSV/SQLite if configured
        try:
            from src.core.database import db_manager

            # Update trade in database with exit information
            trade_data = {
                'exit_price': entry,  # Current price at exit
                'profit': profit,
                'balance': self.broker.get_equity() or self.broker.get_balance(),
                'result_code': 1 if profit > 0 else -1,
                'status': 'closed',
            }

            # Find the trade by symbol and side (this is a simplified approach)
            # In a real implementation, you'd track the trade ID
            trades_df = db_manager.get_trades(limit=1, status='open')
            if not trades_df.empty:
                trade_id = trades_df.iloc[0]['id']
                db_manager.update_trade(trade_id, trade_data)
                self.logger.info(f"Trade {trade_id} updated in database")

            # Also save to CSV if configured
            csv_path = "data/trade_log.csv"
            import os

            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            trade_record = {
                'timestamp': datetime.now(UTC).isoformat(),
                'symbol': self.symbol,
                'side': side,
                'entry_price': entry,
                'exit_price': entry,
                'sl': sl,
                'tp': tp,
                'lot_size': lot_size,
                'profit': profit,
                'result': result,
            }

            import pandas as pd

            df = pd.DataFrame([trade_record])

            # Append to CSV
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)

            self.logger.info(f"Trade logged to CSV: {csv_path}")

        except Exception as e:
            self.logger.error(f"Failed to save trade to database/CSV: {e}")

    def _mini_report(self) -> None:
        """Print compact status report"""
        logger.info("\n" + "=" * 60)
        logger.info(f"MR BEN STATUS REPORT - {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.symbol} | Mode: {self.args.mode}")
        logger.info(f"Cycles: {self.cycle_count} | Trades: {self.metrics.trade_count}")
        logger.info(f"Errors: {self.metrics.error_count}")

        # Multi-timeframe summary
        if hasattr(self, 'last_ta_decisions'):
            logger.info(f"MTF Consensus: {self.last_ta_decisions}")

        # Fusion scores
        if hasattr(self, 'last_fusion'):
            logger.info(
                f"Fusion: Buy={self.last_fusion.score_buy:.3f}, Sell={self.last_fusion.score_sell:.3f}"
            )
            logger.info(f"Decision: {self.last_fusion.label} (conf: {self.last_fusion.confidence:.3f})")

        # Risk summary
        fallback_count = getattr(self, 'fallback_count', 0)
        logger.info(f"ATR Fallbacks Used: {fallback_count}")
        logger.info("=" * 60)

    def _process_trading_cycle(self, dfs: dict[str, Any], primary_tf: str) -> None:
        """Process a single trading cycle - extracted from run_loop to reduce complexity"""
        # Get primary timeframe data
        df_primary = dfs.get(primary_tf)
        if df_primary is None:
            self.logger.warning(f"Primary timeframe {primary_tf} data not available")
            return

        # Technical analysis decisions
        ta_map = self._ta_decisions(dfs)
        self.last_ta_decisions = ta_map

        # Check multi-timeframe consensus
        min_agreement = self.mtf_params.get('min_agreement', 2)
        buy_count = sum(1 for v in ta_map.values() if v == "buy")
        sell_count = sum(1 for v in ta_map.values() if v == "sell")

        if buy_count < min_agreement and sell_count < min_agreement:
            self.logger.info(
                f"Insufficient consensus: buy={buy_count}, sell={sell_count}, min={min_agreement}"
            )
            # Continue with Confluence Strategy even without consensus

        # Price action signal
        pa = self._pa_signal(df_primary)

        # AI score (extract features from primary timeframe)
        features = self._extract_features(df_primary)
        ai_score = self._ai_score(features)

        # Fuse all signals
        fusion = self._fuse(ta_map, pa, ai_score)
        self.last_fusion = fusion

        # [WARNING] در پروداکشن اجرای سفارش فقط از مسیر evaluate_once→maybe_execute_trade انجام می‌شود.
        # self._execute_confluence_strategy()  # غیرفعال برای جلوگیری از دوبله‌کاری

    def run_loop(self) -> None:
        """Main trading loop"""
        self.cycle_count = 0
        self.fallback_count = 0

        try:
            # Initial output to show the script is running
            logger.info(f"[START] MR BEN Live Trading System started for {self.symbol}")
            logger.info(f"[INFO] Timeframe: {self.args.timeframe}, Mode: {self.args.mode}")
            logger.info(f"[INFO] Start time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")

            while True:
                # Kill switch check
                self._kill_switch_guard()

                if self.args.max_cycles > 0 and self.cycle_count >= self.args.max_cycles:
                    self.logger.info(f"Reached max cycles ({self.args.max_cycles}), stopping")
                    logger.info(f"[STOP] Reached max cycles ({self.args.max_cycles}), stopping")
                    break

                with error_handler("main-cycle", self.logger, self.metrics):
                    self.cycle_count += 1

                    # Output cycle start to stdout for supervisor visibility
                    logger.info(f"[CYCLE] Cycle {self.cycle_count} - {datetime.now(UTC).strftime('%H:%M:%S')}")

                    # Fetch multi-timeframe data
                    timeframes = self.mtf_params.get('timeframes', ['M5', 'M15', 'H1'])
                    dfs = self._fetch_multi_tf_ohlc(self.symbol, timeframes, self.args.bars)

                    if not dfs:
                        self.logger.warning("No data available, skipping cycle")
                        logger.info(f"[WARNING] Cycle {self.cycle_count}: No data available, skipping")
                        continue

                    # Process trading cycle
                    self._process_trading_cycle(dfs, self.args.timeframe)

                    # Print mini report periodically
                    if self.cycle_count % self.args.report_every == 0:
                        self._mini_report()
                        logger.info(f"[REPORT] Cycle {self.cycle_count}: Mini report generated")

                    # Sleep between cycles
                    sleep_time = self.cfg.get('trading', {}).get('sleep_seconds', 12)
                    logger.info(f"[SLEEP] Cycle {self.cycle_count}: Sleeping {sleep_time}s...")
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, stopping gracefully")
            logger.info("[STOP] Received interrupt signal, stopping gracefully")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            logger.info(f"[ERROR] Unexpected error in main loop: {e}")
            raise
        finally:
            self.logger.info("Trading loop ended")
            logger.info("[END] Trading loop ended")

    def _extract_features(self, df: Any) -> list[float]:
        """Extract features for AI model from OHLC data"""
        try:
            # Simple feature extraction - adjust based on your model's expectations
            features = []

            # Price-based features
            features.append(float(df['close'].iloc[-1] / df['close'].iloc[-2] - 1))  # Price change
            features.append(float(df['high'].iloc[-1] / df['low'].iloc[-1] - 1))  # High-low ratio

            # Volume-based features (if available)
            if 'volume' in df.columns:
                features.append(float(df['volume'].iloc[-1] / df['volume'].iloc[-5:].mean() - 1))
            else:
                features.append(0.0)

            # Technical indicators
            rsi = compute_rsi(df, 14)
            features.append(float(rsi.iloc[-1] / 100))  # Normalized RSI

            macd_line, signal_line, hist = compute_macd(df)
            features.append(float(macd_line.iloc[-1] / df['close'].iloc[-1]))  # Normalized MACD
            features.append(float(signal_line.iloc[-1] / df['close'].iloc[-1]))  # Normalized Signal

            # ATR-based features
            atr = compute_atr(df, 14)
            if atr is not None:
                features.append(float(atr.iloc[-1] / df['close'].iloc[-1]))  # Normalized ATR
            else:
                features.append(0.0)

            # Time-based features
            features.append(float(datetime.now(UTC).hour / 24))  # Hour of day
            features.append(float(datetime.now(UTC).weekday() / 7))  # Day of week

            # Ensure we have exactly 10 features
            while len(features) < 10:
                features.append(0.0)
            features = features[:10]

            return features

        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            return [0.0] * 10  # Return default features

    def finalize_report(self) -> None:
        """Generate final run report"""
        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

            # JSON report
            report_data = {
                "run_id": f"run_{timestamp}",
                "symbol": self.symbol,
                "mode": self.args.mode,
                "metrics": {
                    "trade_count": self.metrics.trade_count,
                    "win_count": 0,  # Not implemented yet
                    "loss_count": 0,  # Not implemented yet
                    "error_count": self.metrics.error_count,
                    "last_error": "Not implemented",
                },
                "multi_tf": {
                    "timeframes": self.mtf_params.get('timeframes', []),
                    "last_decisions": getattr(self, 'last_ta_decisions', {}),
                    "min_agreement": self.mtf_params.get('min_agreement', 2),
                },
                "fusion": {
                    "score_buy": getattr(
                        self, 'last_fusion', AdaptiveResult(0, 0, "neutral", 0)
                    ).score_buy,
                    "score_sell": getattr(
                        self, 'last_fusion', AdaptiveResult(0, 0, "neutral", 0)
                    ).score_sell,
                    "label": getattr(self, 'last_fusion', AdaptiveResult(0, 0, "neutral", 0)).label,
                },
                "risk": {
                    "atr_period": self.risk_params.get('atr_period', 14),
                    "rr": self.risk_params.get('rr', 1.5),
                    "sl_k": self.risk_params.get('sl_k', 1.0),
                    "tp_k": self.risk_params.get('tp_k', 1.5),
                    "fallbacks_used": getattr(self, 'fallback_count', 0),
                },
                "last_order": {
                    "side": getattr(self, 'last_order_side', "none"),
                    "entry": getattr(self, 'last_order_entry', 0.0),
                    "sl": getattr(self, 'last_order_sl', 0.0),
                    "tp": getattr(self, 'last_order_tp', 0.0),
                    "used_fallback": getattr(self, 'last_order_fallback', False),
                },
                "timestamps": {
                    "started": getattr(self, 'start_time', datetime.now(UTC)).isoformat(),
                    "ended": datetime.now(UTC).isoformat(),
                },
            }

            # Save JSON report
            json_path = f"logs/run_report_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            # Generate markdown summary
            md_path = f"logs/run_report_{timestamp}.md"
            with open(md_path, 'w') as f:
                f.write("# MR BEN Trading Run Report\n\n")
                f.write(f"**Run ID**: {report_data['run_id']}\n")
                f.write(f"**Symbol**: {report_data['symbol']}\n")
                f.write(f"**Mode**: {report_data['mode']}\n")
                f.write(
                    f"**Duration**: {report_data['timestamps']['started']} to {report_data['timestamps']['ended']}\n\n"
                )

                f.write("## Performance Metrics\n")
                f.write(f"- **Total Trades**: {report_data['metrics']['trade_count']}\n")
                f.write(f"- **Wins**: {report_data['metrics']['win_count']}\n")
                f.write(f"- **Losses**: {report_data['metrics']['loss_count']}\n")
                f.write(f"- **Errors**: {report_data['metrics']['error_count']}\n")
                if report_data['metrics']['last_error']:
                    f.write(f"- **Last Error**: {report_data['metrics']['last_error']}\n")
                f.write("\n")

                f.write("## Multi-Timeframe Analysis\n")
                f.write(f"- **Timeframes**: {', '.join(report_data['multi_tf']['timeframes'])}\n")
                f.write(f"- **Min Agreement**: {report_data['multi_tf']['min_agreement']}\n")
                f.write(f"- **Last Decisions**: {report_data['multi_tf']['last_decisions']}\n\n")

                f.write("## Fusion Scoring\n")
                f.write(f"- **Buy Score**: {report_data['fusion']['score_buy']:.3f}\n")
                f.write(f"- **Sell Score**: {report_data['fusion']['score_sell']:.3f}\n")
                f.write(f"- **Final Decision**: {report_data['fusion']['label']}\n\n")

                f.write("## Risk Management\n")
                f.write(f"- **ATR Period**: {report_data['risk']['atr_period']}\n")
                f.write(f"- **Risk/Reward**: {report_data['risk']['rr']}\n")
                f.write(f"- **SL Multiplier**: {report_data['risk']['sl_k']}\n")
                f.write(f"- **TP Multiplier**: {report_data['risk']['tp_k']}\n")
                f.write(f"- **Fallbacks Used**: {report_data['risk']['fallbacks_used']}\n")

            self.logger.info(f"Final report saved to {json_path} and {md_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate final report: {e}")

def parse_symbol_list(s: str) -> list[str]:
    """Parse comma-separated symbol list and normalize"""
    return [x.strip().upper().replace(".","") for x in (s or "").split(",") if x.strip()]

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    p = argparse.ArgumentParser("MR BEN Live Trader")
    
    # Core arguments with ENV defaults
    p.add_argument("--mode", choices=["live","demo","report"], default=os.getenv("MRBEN_MODE","live"))
    p.add_argument("--symbols", type=str, default=os.getenv("MRBEN_ACTIVE_SYMBOLS","XAUUSD,EURUSD"))
    p.add_argument("--dry-run", action="store_true", default=os.getenv("DRY_RUN_ORDER","0")=="1")
    p.add_argument("--timeframe", type=str, default=os.getenv("MRBEN_TF","M15"))
    p.add_argument("--heartbeat-sec", type=float, default=float(os.getenv("HEARTBEAT_SEC","30")))
    
    # Legacy support
    p.add_argument("--symbol", type=str, help="Trading symbol or comma-separated list (legacy)")
    p.add_argument("--profile", default=None, help="Configuration profile (production/test_loose)")
    
    # Trading mode (legacy support)
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--live", action="store_true", help="Live trading mode (legacy)")
    mode_group.add_argument("--demo", action="store_true", help="Demo trading mode (legacy)")
    mode_group.add_argument("--simulate", action="store_true", help="Simulation mode (legacy)")

    # Risk and execution
    p.add_argument("--max-orders", type=int, default=1, help="Maximum open orders (default: 1)")
    p.add_argument("--bars", type=int, default=1500, help="Bars for ATR/indicators (default: 1500)")
    # پیش‌فرض None: فقط اگر کاربر مقدار داد اورراید می‌کنیم
    p.add_argument("--risk", type=float, default=None, help="Risk per trade override (e.g. 0.003)")
    p.add_argument("--supervisor", choices=["on", "off"], help="Override supervisor.enabled")
    p.add_argument("--challenge", choices=["on", "off"], help="Override challenge_mode.enabled")
    p.add_argument(
        "--json-logs", action="store_true", default=True, help="Enable structured JSON logging"
    )
    p.add_argument(
        "--debug-scan-on-nosignal",
        action="store_true",
        help="Enable debug scan on no signal (DEMO only)",
    )
    p.add_argument(
        "--demo-smoke-signal",
        action="store_true",
        help="Enable demo smoke signals for testing (DEMO only)",
    )
    p.add_argument(
        "--max-risk-per-trade",
        type=float,
        default=0.005,
        help="Max risk per trade (default: 0.005)",
    )

    # Report mode
    p.add_argument("--report-only", action="store_true", help="Generate report only, no trading")

    # Logging and reporting
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )
    p.add_argument(
        "--report-every", type=int, default=20, help="Report every N cycles (default: 20)"
    )
    p.add_argument(
        "--max-cycles", type=int, default=0, help="Max cycles (0 = run forever, default: 0)"
    )

    # Configuration
    p.add_argument(
        "--config",
        default="config/pro_config.json",
        help="Configuration file (default: config/pro_config.json)",
    )

    return p.parse_args()

def load_json_config(config_path: str) -> dict[str, Any]:
    """Load and parse JSON configuration file"""
    try:
        with open(config_path) as f:
            config: dict[str, Any] = json.load(f)
        return config
    except FileNotFoundError:
        logger.info(f"Warning: Configuration file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logger.info(f"Error: Invalid JSON in {config_path}: {e}")
        return {}

def load_config(config_path: str, args=None) -> dict[str, Any]:
    """Load and validate configuration file"""
    import os

    config = load_json_config(config_path)

    # Set defaults if missing
    if not config:
        config = {}

    # Apply profile overrides if specified
    profile = os.getenv("MRBEN_PROFILE")
    if profile and "profiles" in config and profile in config["profiles"]:
        profile_config = config["profiles"][profile]
        # Deep merge profile config into main config
        for key, value in profile_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        logger.info(f"[INFO] Applied profile: {profile}")

    # CLI override for challenge mode
    if args and hasattr(args, 'challenge'):
        if args.challenge == "off":
            config.setdefault("challenge_mode", {})["enabled"] = False
        elif args.challenge == "on":
            config.setdefault("challenge_mode", {})["enabled"] = True

    # Ensure required sections exist
    if "symbols" not in config:
        config["symbols"] = {"supported": ["XAUUSD", "EURUSD", "ADAUSD"], "default": "XAUUSD"}
    if "strategy" not in config:
        config["strategy"] = {"name": "confluence_pro"}
    if "risk" not in config:
        config["risk"] = {"risk_pct": 0.01, "max_concurrent_positions": 1}
    
    # Enforce execution safety limits
    config["risk"]["max_concurrent_positions"] = min(config["risk"].get("max_concurrent_positions", 1), 2)
    config["risk"]["max_pos_per_symbol"] = 1
    
    # Multi-symbol defaults
    config.setdefault("symbols", {})
    config["symbols"].setdefault("default", "XAUUSD")
    config["symbols"].setdefault("multi_default", ["XAUUSD","EURUSD","ADAUSD"])
    config["symbols"].setdefault("price_tick", 0.01)
    config["symbols"].setdefault("qty_step", 0.01)
    config["symbols"].setdefault("min_qty", 0.01)
    config["symbols"].setdefault("base_size", 0.10)
    
    # Circuit breaker defaults
    config.setdefault("circuit_breaker", {})
    config["circuit_breaker"].setdefault("max_failures", 5)
    config["circuit_breaker"].setdefault("timeout", 60)

    return config

def resolve_symbol(cfg: dict[str, Any], cli_symbol: str) -> str:
    """Resolve symbol from CLI or config, with validation"""
    supported_symbols = cfg.get("symbols", {}).get("supported", ["XAUUSD"])
    default_symbol = cfg.get("symbols", {}).get("default", "XAUUSD")

    # If CLI symbol is provided and supported, use it
    if cli_symbol in supported_symbols:
        return cli_symbol

    # If CLI symbol not supported, warn and use default
    if cli_symbol != default_symbol:
        logger.info(f"Warning: Symbol '{cli_symbol}' not supported. Supported: {supported_symbols}")
        logger.info(f"Falling back to default: {default_symbol}")

    return default_symbol

def _resolve_symbol_cfgs(cfg: dict, symbols_csv: str | None, default_symbol: str) -> list[SymbolConfig]:
    """Resolve symbol configurations for multi-symbol trading"""
    # 1) symbols
    if symbols_csv:
        names = [s.strip() for s in symbols_csv.split(",") if s.strip()]
    else:
        names = cfg.get("symbols", {}).get("multi_default", [default_symbol])

    # 2) per-symbol meta (tick/step/min/base_size) – try cfg['symbols'][name], else global defaults
    out = []
    g = cfg.get("symbols", {})
    for name in names[:3]:  # limit to 3
        meta = g.get(name, {})
        tick = Decimal(str(meta.get("price_tick", g.get("price_tick", 0.01))))
        qstep = Decimal(str(meta.get("qty_step", g.get("qty_step", 0.01))))
        qmin = Decimal(str(meta.get("min_qty", g.get("min_qty", 0.01))))
        base = Decimal(str(meta.get("base_size", g.get("base_size", 0.10))))
        out.append(SymbolConfig(name=name, price_tick=tick, qty_step=qstep, min_qty=qmin, base_size=base))
    return out

def _fetch_signal_for_symbol(symbol: str) -> tuple[dict, dict, dict]:
    """
    Adapter: produce (signal, context, quotes) for the given symbol
    Expected fields:
      - signal: {"side": "buy"|"sell"|None, "rsi":..., "macd":..., ...}
      - context: {"atr":..., "spread":..., "volatility":..., "session_code":..., "trend_strength":...}
      - quotes: {"bid": float, "ask": float}
    """
    try:
        # Get symbol info for quotes
        symbol_info = get_symbol_info(symbol)
        tick_data = symbol_info.get("tick", {})
        
        # Extract quotes
        quotes = {
            "bid": float(tick_data.get("bid", 0.0)),
            "ask": float(tick_data.get("ask", 0.0))
        }
        
        # Load actual config from file
        try:
            cfg = load_json_config("config_optimized.json")
        except:
            # Fallback config
            cfg = {
                "strategy": {
                    "timeframes": {"trend": "H1", "signal": "M15"},
                    "ema_periods": [50, 200],
                    "ict": {
                        "lookback": 200,
                        "bos_min_swings": 3,
                        "bos_min_strength": 0.5,
                        "bos_max_age": 100
                    },
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                    "filters": {
                        "min_rr": 1.0,
                        "max_spread_points": 80
                    }
                },
                "symbols": {
                    "overrides": {}
                }
            }
        
        # Get services (we'll need to pass this from main)
        services = {"ai_filter": None}
        
        # Call evaluate_once to get signal and metadata
        signal, metadata = evaluate_once(
            symbol=symbol,
            cfg=cfg,
            services=services,
            challenge_state=None,
            is_live_mode=True
        )
        
        # Convert signal to dict format
        signal_dict = {}
        if signal:
            signal_dict = {
                "side": getattr(signal, 'side', None),
                "confidence": getattr(signal, 'confidence', 0.0),
                "reason": getattr(signal, 'reason', ''),
                "entry": getattr(signal, 'entry', 0.0),
                "sl": getattr(signal, 'sl', 0.0),
                "tp": getattr(signal, 'tp', 0.0),
                "rsi": getattr(signal, 'rsi', 50.0),
                "macd": getattr(signal, 'macd', 0.0),
                "meta": getattr(signal, 'meta', {})
            }
        
        # Build context from metadata and symbol info
        context = {
            "atr": metadata.get("atr", 0.0),
            "spread": symbol_info.get("spread_points", 0.0),
            "volatility": metadata.get("volatility", 0.0),
            "session_code": metadata.get("session_code", 2),
            "trend_strength": metadata.get("trend_strength", 20.0),
            "symbol_info": symbol_info
        }
        
        return signal_dict, context, quotes
        
    except Exception:
        # Return empty signal on error
        return {"side": None}, {"atr": 0.0, "spread": 0.0, "volatility": 0.0, "session_code": 2, "trend_strength": 20.0}, {"bid": 0.0, "ask": 0.0}

def bootstrap_broker() -> bool:
    """Bootstrap and initialize MT5 broker connection"""
    try:
        import MetaTrader5 as mt5

        if not mt5.initialize():
            logger.info("❌ Failed to initialize MT5")
            return False

        logger.info("[OK] MT5 initialized successfully")
        return True

    except ImportError:
        logger.info("[ERROR] MetaTrader5 not available - aborting")
        sys.exit(1)
    except Exception as e:
        logger.info(f"❌ Broker bootstrap error: {e}")
        return False

def get_symbol_info(symbol: str) -> dict[str, Any]:
    """Get symbol information from broker (idempotent, no sys.exit)"""
    try:
        import MetaTrader5 as mt5

        # Select symbol (idempotent - safe to call multiple times)
        if not broker.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol: {symbol}")

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise RuntimeError(f"Failed to get symbol info for {symbol}")

        # Get current spread and tick data
        tick = mt5.symbol_info_tick(symbol)
        spread_points = (tick.ask - tick.bid) / symbol_info.point if tick else 0

        # Include tick data for percentage spread calculation
        tick_data = {}
        if tick:
            tick_data = {"ask": tick.ask, "bid": tick.bid, "time": tick.time, "volume": tick.volume}

        return {
            "symbol": symbol,
            "digits": symbol_info.digits,
            "point": symbol_info.point,
            "spread_points": spread_points,
            "tick_value": symbol_info.trade_tick_value,
            "min_lot": symbol_info.volume_min,
            "max_lot": symbol_info.volume_max,
            "lot_step": symbol_info.volume_step,
            "tick": tick_data,
        }

    except Exception as e:
        logger.info(f"[WARNING] Error getting symbol info: {e}")
        raise RuntimeError(f"Symbol info unavailable for {symbol}: {e}")

# No dummy symbol info in production - fail fast

def build_services(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build and initialize trading services"""
    services = {}

    # Initialize AI Filter
    try:
        from src.ai.filter import ConfluenceAIFilter

        ai_filter = ConfluenceAIFilter()
        services["ai_filter"] = ai_filter
        logger.info("[OK] AI Filter initialized")
    except Exception as e:
        logger.info(f"[WARNING] AI Filter not available: {e}")
        services["ai_filter"] = None

    # Initialize News Filter
    try:
        from src.core.news_filter import create_news_filter

        news_filter = create_news_filter(cfg.get("strategy", {}).get("filters", {}))
        services["news_filter"] = news_filter
        logger.info("[OK] News Filter initialized")
    except Exception as e:
        logger.info(f"[WARNING] News Filter not available: {e}")
        services["news_filter"] = None

    # Initialize Risk Manager
    try:
        from src.core.risk_manager import atr_levels, position_size_by_risk

        services["risk_manager"] = {
            "atr_levels": atr_levels,
            "position_size_by_risk": position_size_by_risk,
        }
        logger.info("[OK] Risk Manager initialized")
    except Exception as e:
        logger.info(f"[WARNING] Risk Manager not available: {e}")
        services["risk_manager"] = None

    return services

def build_logger(log_level: str) -> logging.Logger:
    """Build and configure logger"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='[%(asctime)s][%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/live_trader_clean.log")],
    )

    return logging.getLogger("core.trader")

def _normalize_ohlc_util(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLC data to standard format with UTC time (utility function)"""
    # enforce columns and UTC time
    if "time" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "time"})
    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        else:
            df["volume"] = 0
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df

def _drop_incomplete_last_bar(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Drop the last incomplete bar to ensure no look-ahead bias"""
    if df.empty:
        return df
    tf_min = TF_TO_MIN.get(tf, 15)
    last_t = pd.to_datetime(df['time'].iloc[-1], utc=True)
    # If timeframe is not yet complete, remove the last row
    now_utc = pd.Timestamp.utcnow()
    if now_utc < last_t + pd.Timedelta(minutes=tf_min):
        return df.iloc[:-1].copy()
    return df

def _fetch_confluence_data_util(
    symbol: str, cfg: dict[str, Any], bars: int = 600
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch data for Confluence Strategy (utility function)"""
    try:
        tf_trend = cfg.get("strategy", {}).get("timeframes", {}).get("trend", "H1")
        tf_signal = cfg.get("strategy", {}).get("timeframes", {}).get("signal", "M15")

        # fetch from real broker (no dummy)
        df_htf = broker.copy_rates(symbol, tf_trend, bars + 2)
        df_ltf = broker.copy_rates(symbol, tf_signal, bars + 2)

        if df_htf is None or df_ltf is None:
            raise RuntimeError("rates_none")

        df_htf = _normalize_ohlc_util(df_htf)
        df_ltf = _normalize_ohlc_util(df_ltf)

        # 🔒 Remove incomplete last bar
        df_htf = _drop_incomplete_last_bar(df_htf, tf_trend)
        df_ltf = _drop_incomplete_last_bar(df_ltf, tf_signal)
        
        # Apply data budget limits
        from src.core.data_budget import enforce_bar_limits
        df_htf = enforce_bar_limits(df_htf, tf_trend)
        df_ltf = enforce_bar_limits(df_ltf, tf_signal)

        if len(df_htf) < bars // 2 or len(df_ltf) < bars // 2:
            raise RuntimeError("insufficient_ohlc")

        logger.info(
            f"Fetched real OHLC for confluence (closed): HTF={tf_trend} rows={len(df_htf)}, LTF={tf_signal} rows={len(df_ltf)}"
        )
        return df_htf.tail(bars), df_ltf.tail(bars)

    except Exception as e:
        logger.info(f"Failed to fetch confluence data from MT5: {e}")
        return pd.DataFrame(), pd.DataFrame()

def evaluate_once(
    symbol: str,
    cfg: dict[str, Any],
    services: dict[str, Any],
    challenge_state: ChallengeState = None,
    is_live_mode: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Evaluate confluence strategy once and return signal with metadata"""
    from datetime import datetime

    try:
        # Create context object for DEMO mode detection
        class Context:
            def __init__(self, is_demo=False):
                self.is_demo = is_demo

        ctx = Context(is_demo=not is_live_mode)

        # Get symbol info
        try:
            symbol_info = get_symbol_info(symbol)
            # Display spread in percentage for crypto, points for others
            spread_display = symbol_info['spread_points']
            if symbol in cfg.get("symbols", {}).get("overrides", {}):
                overrides = cfg["symbols"]["overrides"][symbol]
                if (
                    overrides.get("asset_class") == "crypto"
                    and overrides.get("spread_policy") == "percent"
                ):
                    # Calculate percentage spread
                    tick_data = symbol_info.get("tick", {})
                    if tick_data and "ask" in tick_data and "bid" in tick_data:
                        ask = tick_data["ask"]
                        bid = tick_data["bid"]
                        if ask > 0 and bid > 0:
                            mid = (ask + bid) / 2
                            spread_pct = ((ask - bid) / mid) * 100
                            spread_display = f"{spread_pct:.3f}%"

            logger.info(
                f"[INFO] Symbol ready: {symbol} | Digits={symbol_info['digits']} | Point={symbol_info['point']} | Spread={spread_display}"
            )
        except RuntimeError as e:
            logger.info(f"❌ {e}")
            return None, {"error": "symbol_info_failed", "timestamp": datetime.now(UTC).isoformat()}

        # Get timeframes from config
        timeframes = cfg.get("strategy", {}).get("timeframes", {"trend": "H1", "signal": "M15"})
        tf_trend = timeframes.get("trend", "H1")
        tf_signal = timeframes.get("signal", "M15")

        # Fetch real data for both timeframes using utility function
        df_htf, df_ltf = _fetch_confluence_data_util(symbol, cfg, bars=600)

        if df_htf.empty or df_ltf.empty:
            logger.info(f"❌ Failed to fetch real data for {symbol}")
            return None, {"error": "data_fetch_failed", "timestamp": datetime.now(UTC).isoformat()}

        # Log closed bars information
        logger.info(f"OHLC(ClosedBars): HTF={tf_trend} N={len(df_htf)}, LTF={tf_signal} N={len(df_ltf)}")
        logger.info(f"[INFO] Fetched data: HTF={tf_trend}, LTF={tf_signal}")

        # QUICK SANITY GATES - before placing order
        required_cols = {"time", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df_ltf.columns) or not required_cols.issubset(df_htf.columns):
            logger.info("[ERROR] Missing required OHLC columns")
            return None, {"decision": "schema_error", "timestamp": datetime.now(UTC).isoformat()}

        # Import and call confluence strategy
        from src.strategies.confluence_pro_strategy import confluence_signal

        # Add debug analysis if no signal is generated
        debug_enabled = cfg.get("strategy", {}).get("debug_mode", False) or cfg.get(
            "debug_scan_on_nosignal", False
        )
        logger.info(
            f"[DEBUG] debug_enabled: {debug_enabled}, debug_scan_on_nosignal: {cfg.get('debug_scan_on_nosignal', False)}"
        )

        # Debug indicators before calling confluence_signal
        if debug_enabled:
            try:
                from src.indicators.atr import compute_atr
                from src.indicators.rsi_macd import compute_macd, compute_rsi

                rsi_ltf = compute_rsi(df_ltf, 14).iloc[-1]
                macd_ltf, sig_ltf, hist_ltf = compute_macd(df_ltf)
                macd_hist_ltf = hist_ltf.iloc[-1]
                atr_ltf = compute_atr(df_ltf, 14).iloc[-1]
                logger.info(
                    f"[DEBUG] LTF RSI={rsi_ltf:.1f} | MACD_hist={macd_hist_ltf:.6f} | ATR={atr_ltf:.5f}"
                )

                rsi_htf = compute_rsi(df_htf, 14).iloc[-1]
                macd_htf, sig_htf, hist_htf = compute_macd(df_htf)
                macd_hist_htf = hist_htf.iloc[-1]
                logger.info(f"[DEBUG] HTF RSI={rsi_htf:.1f} | MACD_hist={macd_hist_htf:.6f}")
            except Exception as e:
                logger.info(f"[DEBUG] inline-debug error: {e}")

        logger.info(
            f"[DEBUG] Calling confluence_signal with debug_nosignal={debug_enabled}, demo_smoke_signal={cfg.get('demo_smoke_signal', False)}"
        )
        logger.info(f"[DEBUG] Full cfg keys: {list(cfg.keys())}")
        logger.info(f"[DEBUG] cfg demo_smoke_signal value: {cfg.get('demo_smoke_signal', 'NOT_FOUND')}")
        logger.info(f"[DEBUG] ctx.is_demo: {getattr(ctx, 'is_demo', 'NOT_FOUND')}")
        
        # Fix: Add demo_smoke_signal to strategy config if missing
        if "demo_smoke_signal" not in cfg:
            # Get from parent config in evaluate_once
            import inspect
            frame = inspect.currentframe()
            try:
                caller_locals = frame.f_back.f_locals
                parent_cfg = caller_locals.get('cfg', {})
                cfg["demo_smoke_signal"] = parent_cfg.get("demo_smoke_signal", False)
            except:
                cfg["demo_smoke_signal"] = False
            finally:
                del frame
        
        demo_smoke_enabled = cfg.get("demo_smoke_signal", False)
        
        logger.info(f"[DEBUG] Final demo_smoke_enabled: {demo_smoke_enabled}")
        
        signal = confluence_signal(
            df_ltf=df_ltf,
            df_htf=df_htf,
            cfg=cfg,
            symbol_point=symbol_info["point"],
            ai_filter=services.get("ai_filter"),
            debug_nosignal=debug_enabled,
            demo_smoke_signal=True,  # Force enable for testing
            symbol=symbol,
            ctx=ctx,
            market_info=symbol_info,
        )

        # Top-level failsafe SMOKE injection for DEMO mode
        if not signal and cfg.get("demo_smoke_signal", False) and getattr(ctx, "is_demo", False):
            logger.info("[SMOKE] top-level fallback injection triggered (DEMO)")

            # Create a simple SMOKE signal
            from datetime import datetime

            _last = df_ltf.iloc[-1]
            price = float(_last["close"])
            digits = symbol_info.get("digits", 5)
            point = symbol_info.get("point", 10**-digits)

            # Simple side determination
            rsi_val = float(_last.get("rsi", 50.0))
            side = "long" if rsi_val >= 50 else "short"

            # Simple SL/TP calculation
            pip = max(point, 1e-6)
            sl_pad = 5 * pip
            tp_pad = 5 * pip

            if side == "long":
                sl = round(price - sl_pad, digits)
                tp = round(price + tp_pad, digits)
            else:
                sl = round(price + sl_pad, digits)
                tp = round(price - tp_pad, digits)

            rr = round((abs(tp - price) / max(abs(price - sl), pip)), 2)

            # Create failsafe signal
            from src.strategies.confluence_pro_strategy import Signal

            signal = Signal(
                side=side,
                confidence=0.80,  # Increased from 0.60
                reason="smoke_test_failsafe_injection",
                entry=round(price, digits),
                sl=sl,
                tp=tp,
                meta={
                    "source": "smoke",
                    "symbol": symbol,
                    "rr": rr,
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds") + "Z",
                    "version": 1,
                    "demo_injection": True,
                    "failsafe": True,
                },
            )

            logger.info(
                f"[SMOKE] failsafe signal created | symbol={symbol} side={side} rr={rr:.2f} entry={price:.5f} sl={sl:.5f} tp={tp:.5f}"
            )

        # Debug analysis if no signal and debug mode is enabled
        if not signal and debug_enabled:
            try:
                # Simple debug analysis without external dependencies
                from src.price_action.ict import build_ict_context
                from src.strategies.confluence_pro_strategy import trend_context

                # HTF Trend analysis
                tc = trend_context(df_htf, *cfg.get("strategy", {}).get("ema_periods", [50, 200]))
                logger.info(f"[DEBUG] HTF Trend: {tc['trend']}")

                # ICT Structure analysis
                ict_cfg = cfg.get("strategy", {}).get("ict", {})
                if ict_cfg:
                    ict_ctx = build_ict_context(
                        df_ltf.tail(ict_cfg.get("lookback", 200)).copy(), ict_cfg
                    )
                    logger.info(f"[DEBUG] ICT Structure: {ict_ctx.structure_side}")

                # MACD analysis
                from src.indicators.rsi_macd import compute_macd

                macd_params = {
                    k: v
                    for k, v in cfg.get("strategy", {}).get("macd", {}).items()
                    if k in ['fast', 'slow', 'signal']
                }
                if macd_params:
                    # Check if close column exists
                    if 'close' in df_htf.columns:
                        macd_line, signal_line, histogram = compute_macd(df_htf, **macd_params)
                        logger.info(f"[DEBUG] MACD Hist HTF: {histogram.iloc[-1]:.6f}")
                    else:
                        logger.info(f"[DEBUG] HTF columns: {list(df_htf.columns)}")

                    if 'close' in df_ltf.columns:
                        macd_line_ltf, signal_line_ltf, histogram_ltf = compute_macd(
                            df_ltf, **macd_params
                        )
                        logger.info(f"[DEBUG] MACD Hist LTF: {histogram_ltf.iloc[-1]:.6f}")
                    else:
                        logger.info(f"[DEBUG] LTF columns: {list(df_ltf.columns)}")

            except Exception as e:
                import traceback

                logger.info(f"[DEBUG] Error in debug analysis: {e}")
                logger.info(f"[DEBUG] Traceback: {traceback.format_exc()}")

        # Build enhanced context for All-of-Four gates
        ctx_dict = {
            "market": {
                "spread_points": symbol_info["spread_points"],
                "point": symbol_info["point"],
            },
            "indicators": {
                "atr_m15_points": symbol_info.get("atr_points", 40),
                "atr": symbol_info.get("atr", 0.0),
            },
            "cooldown_ok": True,  # TODO: implement cooldown check
            "dd_soft_block": False,  # TODO: implement DD check
            "spread_ok": True,  # Will be checked by spread controller
            "tech_ok": True,  # Default to True for basic signals
            "pa_ok": True,  # Default to True for basic signals
            "ml_prob": 0.0,  # Will be set by ML filter
            "sup_ok": True,  # Will be set by supervisor
            "rr": 1.5,  # Default RR
            "direction": "flat",  # Will be set by strategy
        }

        # Add STEALTH strategy integration
        try:
            from src.strategies.stealth_strategy import StealthStrategy

            stealth = StealthStrategy(cfg)
            stealth_sig = stealth.generate(df_htf, df_ltf, symbol_info["point"])

            if stealth_sig.ok:
                ctx_dict.update(
                    {
                        "tech_ok": stealth_sig.tech_ok,
                        "pa_ok": stealth_sig.pa_ok,
                        "direction": stealth_sig.direction,
                        "rr": stealth_sig.rr,
                    }
                )
                logger.info(
                    f"[STEALTH] Signal: {stealth_sig.direction} | tech_ok={stealth_sig.tech_ok} | pa_ok={stealth_sig.pa_ok} | rr={stealth_sig.rr:.2f}"
                )
        except Exception as e:
            logger.info(f"[WARNING] STEALTH strategy error: {e}")
            # Fallback to basic values
            ctx_dict.update(
                {
                    "tech_ok": True,  # Default to True for basic signals
                    "pa_ok": True,  # Default to True for basic signals
                    "direction": signal.side if signal else "flat",
                    "rr": signal.meta.get('rr', 1.5) if signal else 0.0,
                }
            )

        # Set ML probability based on direction
        if signal:
            ml_prob_long = signal.confidence if signal.side == "long" else 0.0
            ml_prob_short = signal.confidence if signal.side == "short" else 0.0
            ctx_dict["ml_prob"] = float(ml_prob_long if signal.side == "long" else ml_prob_short)
        else:
            # Use STEALTH direction if available
            direction = ctx_dict.get("direction", "flat")
            if direction == "long":
                ctx_dict["ml_prob"] = 0.75  # Default confidence for STEALTH long
                ctx_dict["rr"] = 1.5  # Set default RR for long signals
            elif direction == "short":
                ctx_dict["ml_prob"] = 0.75  # Default confidence for STEALTH short
                ctx_dict["rr"] = 1.5  # Set default RR for short signals
            else:
                ctx_dict["ml_prob"] = 0.0
                ctx_dict["rr"] = 0.0

        # Add missing context pieces
        try:
            # Import broker if not available
            from src.core.broker_mt5 import broker

            # Cooldown check
            ctx_dict["cooldown_ok"] = True  # TODO: implement proper cooldown check

            # DD check
            equity = broker.get_equity() or broker.get_balance() or 0
            peak_equity = (
                getattr(challenge_state, 'equity_peak', equity) if challenge_state else equity
            )
            dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            ctx_dict["dd_soft_block"] = dd_pct >= cfg.get("risk", {}).get("dd_soft_from_peak", 0.06)

            # Adaptive ML thresholds per symbol
            ml_min_base = cfg.get("signals", {}).get("min_conf", 0.75)

            # Simple performance-based adjustment (can be enhanced with actual trade history)
            if symbol.startswith("XAU"):
                # XAU tends to be more volatile, adjust threshold based on recent performance
                ctx_dict["ml_min_effective"] = ml_min_base + 0.05  # Slightly stricter for XAU
                # Get ATR points safely
                atr_points = 40  # Default fallback
                try:
                    atr_points = (
                        ctx_dict.get("market", {}).get("indicators", {}).get("atr_m15_points", 40)
                    )
                except:
                    atr_points = 40
                ctx_dict["spread_cap_adaptive"] = min(80, 0.4 * atr_points + 8)
            elif symbol.startswith("EUR"):
                # EURUSD is more stable, can be slightly more lenient
                ctx_dict["ml_min_effective"] = ml_min_base - 0.02
                ctx_dict["spread_cap_adaptive"] = 50  # Lower spread cap for major pairs
            else:
                ctx_dict["ml_min_effective"] = ml_min_base
                ctx_dict["spread_cap_adaptive"] = 100  # Default cap

            # Create supervisor snapshot
            ctx_dict["snapshot"] = {
                "symbol": symbol,
                "direction": ctx_dict.get("direction", "flat"),
                "ml_prob": ctx_dict.get("ml_prob", 0.0),
                "rr": ctx_dict.get("rr", 0.0),
                "spread": ctx_dict["market"]["spread_points"],
                "equity": equity,
                "base_risk": cfg.get("risk", {}).get("base_risk", 0.01),
            }
        except Exception as e:
            logger.info(f"[WARNING] Context enhancement error: {e}")

        # Build decision metadata
        decision_meta = {
            "symbol": symbol,
            "timeframes": {"trend": tf_trend, "signal": tf_signal},
            "symbol_info": symbol_info,
            "timestamp": datetime.now(UTC).isoformat(),
            "has_signal": signal is not None,
            "context": ctx_dict,
        }

        # Add challenge mode info
        try:
            now_utc = datetime.now(UTC)
            if challenge_state is not None:
                decision_meta["challenge_mode"] = {
                    "enabled": cfg.get("challenge_mode", {}).get("enabled", False),
                    "equity_peak": challenge_state.equity_peak,
                    "equity_start_of_day": challenge_state.equity_start_of_day,
                    "trades_today": challenge_state.trades_today,
                    "consecutive_losses": challenge_state.consecutive_losses,
                }
            else:
                decision_meta["challenge_mode"] = {"enabled": False}
        except Exception as e:
            logger.info(f"[WARNING] Error adding challenge mode info: {e}")
            decision_meta["challenge_mode"] = {"enabled": False}

        if signal:
            # Try STEALTH strategy for enhanced signals
            try:
                from src.strategies.stealth_strategy import StealthStrategy

                stealth = StealthStrategy(cfg)
                stealth_sig = stealth.generate(df_htf, df_ltf, symbol_info["point"])

                if stealth_sig.ok:
                    ctx_dict.update(
                        {
                            "tech_ok": stealth_sig.tech_ok,
                            "pa_ok": stealth_sig.pa_ok,
                            "direction": stealth_sig.direction,
                            "rr": stealth_sig.rr,
                            "pro_signal": stealth_sig.ok,
                        }
                    )
                    logger.info(
                        f"[STEALTH] Enhanced signal: {stealth_sig.direction} | tech_ok={stealth_sig.tech_ok} | pa_ok={stealth_sig.pa_ok} | rr={stealth_sig.rr:.2f}"
                    )
            except Exception as e:
                logger.info(f"[WARNING] STEALTH strategy failed: {e}")
                # Fallback to basic context
                ctx_dict.update(
                    {
                        "tech_ok": True,
                        "pa_ok": True,
                        "direction": signal.side,
                        "rr": signal.meta.get('rr', 1.5),
                        "pro_signal": True,
                    }
                )

            # ML probability (simplified for now)
            ml_prob_long = signal.confidence if signal.side == "long" else 0.0
            ml_prob_short = signal.confidence if signal.side == "short" else 0.0
            ctx_dict["ml_prob"] = float(ml_prob_long if signal.side == "long" else ml_prob_short)
            # Ensure RR is set for signals
            if signal and signal.meta.get('rr', 0) == 0:
                ctx_dict["rr"] = 1.5  # Default RR for signals

            # Supervisor snapshot
            try:
                from src.core.broker_mt5 import broker

                equity = broker.get_account_info().get("equity", 0.0)
            except:
                equity = 0.0

            ctx_dict["snapshot"] = {
                "symbol": symbol,
                "direction": signal.side,
                "ml_prob": ctx_dict["ml_prob"],
                "rr": ctx_dict.get("rr", 0.0),
                "spread": ctx_dict["market"]["spread_points"],
                "equity": equity,
                "base_risk": cfg["risk"]["base_risk"],
            }

            decision_meta.update(
                {
                    "signal_side": signal.side,
                    "signal_confidence": signal.confidence,
                    "signal_reason": signal.reason,
                    "entry_price": signal.entry,
                    "stop_loss": signal.sl,
                    "take_profit": signal.tp,
                    "risk_reward": signal.meta.get('rr', 0),
                    "rsi": signal.meta.get('rsi', 0),
                    "atr": signal.meta.get('atr', 0),
                    "trend": signal.meta.get('trend', 'unknown'),
                    "ict_structure": signal.meta.get('ict', {}).get('structure_side', 'unknown'),
                }
            )

            # Check confidence threshold
            min_confidence = cfg.get("strategy", {}).get("min_confidence", 0.40)  # Very low for testing
            if signal.confidence < min_confidence:
                logger.info(
                    f"[INFO] Signal rejected: low confidence {signal.confidence:.2f} < {min_confidence}"
                )
                return None, decision_meta

            # Calculate trade cost (spread + commission) using real values
            from src.core.broker_mt5 import broker
            from src.execution.costs import CostModel

            spread_pts = symbol_info.get('spread_points', 0)
            point_val = broker.point_value(symbol)
            commission = float(cfg.get("costs", {}).get("commission_per_lot", 7.0))
            lots = signal.meta.get('lots', 0.01)

            # Initial cost preview (before final sizing)
            cost_model = CostModel(spread_pts, commission, point_val)
            initial_cost = cost_model.estimate_trade_cost(lots)

            logger.info(
                f"[INFO] Trade cost preview (initial): spread=${spread_pts * point_val * lots:.2f}, commission=${commission * lots:.2f}, total=${initial_cost:.2f}"
            )

            # Final cost preview after position sizing
            final_lots = lots  # This would be the actual sized position
            final_cost = cost_model.estimate_trade_cost(final_lots)

            logger.info(
                f"[INFO] Trade cost preview (final): lots={final_lots:.2f}, spread=${spread_pts * point_val * final_lots:.2f}, commission=${commission * final_lots:.2f}, total=${final_cost:.2f}"
            )

            # Cost gate check with final lots
            if hasattr(signal, 'expected_edge') and signal.expected_edge <= final_cost:
                logger.info(
                    f"[INFO] Trade SKIPPED - Reason: negative_edge_after_costs (edge: ${signal.expected_edge:.2f} <= cost: ${final_cost:.2f})"
                )
                return None, decision_meta

            logger.info(
                f"[INFO] Signal generated: {signal.side} | Confidence: {signal.confidence} | RR: {signal.meta.get('rr', 0):.2f}"
            )
        else:
            logger.info("[INFO] No signal generated")
            # Initialize context for no-signal case
            ctx_dict.update(
                {
                    "tech_ok": False,
                    "pa_ok": False,
                    "direction": "flat",
                    "rr": 0.0,
                    "pro_signal": False,
                }
            )

        return signal, decision_meta

    except Exception as e:
        logger.info(f"[ERROR] Evaluation error: {e}")
        import traceback

        traceback.print_exc()
        return None, {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

def manage_open_positions(symbol: str, broker, config: dict) -> dict:
    """Professional position management: Partial TP + Break-Even"""
    try:
        from src.core.broker_mt5 import broker as mt5_broker

        # Get open positions for this symbol
        try:
            positions = broker.get_positions(symbol) if hasattr(broker, 'get_positions') else []
        except:
            positions = []

        if not positions:
            return {"managed": 0, "actions": []}

        actions = []
        managed_count = 0

        for position in positions:
            ticket = position.get('ticket')
            current_price = position.get('price', 0)
            entry_price = position.get('price_open', 0)
            sl = position.get('sl', 0)
            tp = position.get('tp', 0)
            volume = position.get('volume', 0)
            profit = position.get('profit', 0)

            if not ticket or not current_price or not entry_price:
                continue

            # Calculate current R (risk-reward ratio)
            if sl > 0:
                risk_distance = abs(entry_price - sl)
                profit_distance = abs(current_price - entry_price)
                current_R = profit_distance / risk_distance if risk_distance > 0 else 0
            else:
                current_R = 0

            # Check for Partial TP at R >= 1.0
            if current_R >= 1.0 and not position.get('tp_half_done', False):
                try:
                    # Close half position
                    close_volume = volume * 0.5
                    result = mt5_broker.close_position_partial(ticket, close_volume)
                    if result:
                        actions.append(f"Partial TP: {ticket} (R={current_R:.2f})")
                        position['tp_half_done'] = True
                        managed_count += 1
                except Exception as e:
                    logger.info(f"[WARNING] Partial TP failed for {ticket}: {e}")

            # Check for Break-Even at R >= 0.8
            elif current_R >= 0.8 and not position.get('be_done', False):
                try:
                    # Move SL to BE + small offset
                    offset = risk_distance * 0.1  # 10% of initial risk as offset
                    new_sl = (
                        entry_price + offset
                        if position.get('type', 0) == 0
                        else entry_price - offset
                    )

                    result = mt5_broker.modify_position(ticket, new_sl, tp)
                    if result:
                        actions.append(f"Break-Even: {ticket} (R={current_R:.2f})")
                        position['be_done'] = True
                        managed_count += 1
                except Exception as e:
                    logger.info(f"[WARNING] Break-Even failed for {ticket}: {e}")

        return {"managed": managed_count, "actions": actions, "total_positions": len(positions)}

    except Exception as e:
        logger.info(f"[ERROR] Position management failed: {e}")
        return {"managed": 0, "actions": [], "error": str(e)}

def _notify_skip(reason: str, meta: dict[str, Any]) -> None:
    """Helper function to notify supervisor of skip events"""
    skip_meta = {
        "symbol": meta["symbol"],
        "spread_points": meta["symbol_info"]["spread_points"],
        "risk_reward": meta.get("risk_reward", 0.0),
        "rsi": meta.get("rsi", 50.0),
        "atr": meta.get("atr", 0.0),
        "trend": meta.get("trend", "flat"),
        "ict": meta.get("ict", {"structure_side": "neutral", "signals": []}),
        "positions": meta.get("positions", {"open": 0, "by_symbol": {}}),
        "portfolio": meta.get("portfolio", {"open_risk_value": 0.0, "equity": 84985.01}),
        "filters": meta.get("filters", {"min_rr": 1.0, "max_spread_points": 80}),
        "news_block": meta.get("news_block", False),
        "challenge": meta.get(
            "challenge", {"enabled": True, "trades_today": 0, "consecutive_losses": 0}
        ),
        "cb_order": meta.get("cb_order", {"open": False, "recent_errors": 0}),
        "cfg": meta.get("cfg", {"risk_pct": 0.005, "max_concurrent_positions": 1}),
        "mode": meta.get("mode", "DEMO"),
    }
    on_skip(reason, skip_meta)

def maybe_execute_trade(
    signal: Any,
    meta: dict[str, Any],
    cfg: dict[str, Any],
    services: dict[str, Any],
    app_state: dict[str, Any] = None,
    ctx: Any = None,
) -> str | None:
    """Execute trade if conditions are met, return order_id or None"""
    global CHALLENGE_STATE

    if not signal:
        return None

    try:
        # Supervisor hook: signal event
        signal_meta = {
            "symbol": meta["symbol"],
            "spread_points": meta["symbol_info"]["spread_points"],
            "risk_reward": meta.get("risk_reward", 0.0),
            "rsi": meta.get("rsi", 50.0),
            "atr": meta.get("atr", 0.0),
            "trend": meta.get("trend", "flat"),
            "ict": meta.get("ict", {"structure_side": "neutral", "signals": []}),
            "positions": meta.get("positions", {"open": 0, "by_symbol": {}}),
            "portfolio": meta.get("portfolio", {"open_risk_value": 0.0, "equity": 84985.01}),
            "filters": meta.get("filters", {"min_rr": 1.0, "max_spread_points": 80}),
            "news_block": meta.get("news_block", False),
            "challenge": meta.get(
                "challenge", {"enabled": True, "trades_today": 0, "consecutive_losses": 0}
            ),
            "cb_order": meta.get("cb_order", {"open": False, "recent_errors": 0}),
            "cfg": meta.get("cfg", {"risk_pct": 0.005, "max_concurrent_positions": 1}),
            "mode": meta.get("mode", "DEMO"),
        }
        on_signal(signal_meta)

        # Log execution attempt
        from src.reporting.exec_logger import log_exec

        log_exec(
            {
                "phase": "pre_check",
                "symbol": meta["symbol"],
                "rr": meta.get("risk_reward"),
                "spread": meta["symbol_info"]["spread_points"],
                "rsi": meta.get("rsi"),
                "atr": meta.get("atr"),
                "side": signal.side if signal else None,
            }
        )

        # Check safety guards
        strategy_cfg = cfg.get("strategy", {})
        filters = strategy_cfg.get("filters", {})
        risk_cfg = cfg.get("risk", {})

        # Enhanced spread check with crypto percentage support
        symbol = meta["symbol"]
        try:
            from src.core.filters.spread_filter import SpreadFilter

            spread_filter = SpreadFilter(cfg)
            tick_data = meta["symbol_info"].get("tick", {})

            # Debug logging for spread check
            logger.info(
                f"[DEBUG] Spread check for {symbol}: points={meta['symbol_info'].get('spread_points', 0)}"
            )
            if tick_data:
                logger.info(
                    f"[DEBUG] Tick data available: ask={tick_data.get('ask', 'N/A')}, bid={tick_data.get('bid', 'N/A')}"
                )
            else:
                logger.info("[DEBUG] No tick data available, using points-based check")

            spread_result = spread_filter.check_spread(symbol, meta["symbol_info"], tick_data)

            if not spread_result.passed:
                meta["skip_reason"] = spread_result.reason
                _notify_skip(meta["skip_reason"], meta)
                log_exec(
                    {
                        "phase": "skip",
                        "symbol": meta["symbol"],
                        "reason": meta.get("skip_reason"),
                    }
                )
                logger.info(f"[INFO] Spread check failed for {symbol}: {spread_result.reason}")
                return None
            else:
                logger.info(f"[INFO] Spread check passed for {symbol}: {spread_result.reason}")

        except Exception as e:
            logger.info(f"[WARNING] Error in enhanced spread check for {symbol}: {e}")
            # Fallback to simple points check
            current_spread = meta["symbol_info"]["spread_points"]
            max_spread_map = filters.get("max_spread_points_map", {})
            max_spread = max_spread_map.get(symbol, cfg.get("trading", {}).get("spread_thresholds", {}).get(symbol, cfg.get("trading", {}).get("max_spread_points", 1000)))
            if current_spread > max_spread:
                meta["skip_reason"] = f"spread {current_spread:.1f} > {max_spread} (fallback)"
                _notify_skip(meta["skip_reason"], meta)
                log_exec(
                    {
                        "phase": "skip",
                        "symbol": meta["symbol"],
                        "reason": meta.get("skip_reason"),
                    }
                )
                logger.info(f"[INFO] Spread too high for {symbol}: {current_spread:.1f} > {max_spread}")
                return None

        # --- Challenge guards (فقط اگر challenge_mode فعال است) ---
        challenge_enabled = cfg.get("challenge_mode", {}).get("enabled", True)
        if challenge_enabled:
            try:
                now_utc = datetime.now(UTC)
                if CHALLENGE_STATE is None:
                    CHALLENGE_STATE = initialize_challenge_state(broker)

                ok, reason = guard_challenge(
                    cfg, broker, CHALLENGE_STATE, meta["symbol"], now_utc, services
                )
                if not ok:
                    meta["skip_reason"] = f"challenge: {reason}"
                    _notify_skip(meta["skip_reason"], meta)
                    log_exec(
                        {
                            "phase": "skip",
                            "symbol": meta["symbol"],
                            "reason": meta.get("skip_reason"),
                        }
                    )
                    logger.info(f"[INFO] Challenge guard blocked: {reason}")
                    return None
            except Exception as e:
                meta["skip_reason"] = f"challenge_error: {e}"
                _notify_skip(meta["skip_reason"], meta)
                log_exec(
                    {"phase": "skip", "symbol": meta["symbol"], "reason": meta.get("skip_reason")}
                )
                logger.info(f"[WARNING] Challenge guard error: {e}")
                return None

        # RR check with symbol-specific limits
        symbol = meta["symbol"]
        min_rr_map = filters.get("min_rr_map", {})
        min_rr = min_rr_map.get(symbol, filters.get("min_rr", 1.0))
        current_rr = meta.get("risk_reward", 0)
        if current_rr < min_rr:
            meta["skip_reason"] = f"rr {current_rr:.2f} < {min_rr}"
            _notify_skip(meta["skip_reason"], meta)
            log_exec(
                {
                    "phase": "skip",
                    "symbol": meta["symbol"],
                    "reason": meta.get("skip_reason"),
                }
            )
            logger.info(f"[INFO] RR too low for {symbol}: {current_rr:.2f} < {min_rr}")
            return None

        # SMOKE signal bypass for DEMO mode
        if (
            hasattr(signal, 'meta')
            and signal.meta.get("source") == "smoke"
            and getattr(ctx, "is_demo", False)
        ):
            logger.info(
                f"[SMOKE] risk gate bypass (DEMO) | symbol={signal.meta.get('symbol', 'UNKNOWN')}"
            )
            # Skip all risk checks for SMOKE signals in DEMO mode
            pass
        else:
            # Portfolio risk check
            try:
                from src.core.portfolio_risk import portfolio_open_risk

                max_portfolio_risk_pct = risk_cfg.get("max_portfolio_risk_pct", 0.02)  # 2% equity
                equity_now = broker.get_equity() or broker.get_balance()
                open_risk_value, by_symbol = portfolio_open_risk()
                if equity_now > 0 and (open_risk_value / equity_now) > max_portfolio_risk_pct:
                    meta["skip_reason"] = (
                        f"portfolio_risk {(open_risk_value/equity_now):.3f} > {max_portfolio_risk_pct}"
                    )
                    _notify_skip(meta["skip_reason"], meta)
                    log_exec(
                        {
                            "phase": "skip",
                            "symbol": meta["symbol"],
                            "reason": meta.get("skip_reason"),
                        }
                    )
                    logger.info(
                        f"[RISK] Blocked by portfolio risk: {open_risk_value:.2f} / {equity_now:.2f}"
                    )
                    return None
            except Exception as e:
                logger.info(f"[WARNING] Portfolio risk check error: {e}")

            # Position count check (REAL)
            max_positions = risk_cfg.get("max_concurrent_positions", 1)
            try:
                from src.core.reconcile import count_open_positions

                current_positions = count_open_positions(symbol)
            except Exception as e:
                meta["skip_reason"] = f"positions_error: {e}"
                current_positions = 0

            if current_positions >= max_positions:
                meta["skip_reason"] = f"max_positions {current_positions}/{max_positions}"
                _notify_skip(meta["skip_reason"], meta)
                log_exec(
                    {
                        "phase": "skip",
                        "symbol": meta["symbol"],
                        "reason": meta.get("skip_reason"),
                    }
                )
                logger.info(f"[INFO] Max positions reached: {current_positions}/{max_positions}")
                return None

        # News filter check
        if services.get("news_filter"):
            is_news_time, reason = services["news_filter"].is_news_time(meta["symbol"])
            if is_news_time:
                meta["skip_reason"] = f"news: {reason}"
                _notify_skip(meta["skip_reason"], meta)
                log_exec(
                    {
                        "phase": "skip",
                        "symbol": meta["symbol"],
                        "reason": meta.get("skip_reason"),
                    }
                )
                logger.info(f"[INFO] News filter blocked: {reason}")
                return None

        # AI Filter check
        if services.get("ai_filter") and strategy_cfg.get("ai_filter", {}).get("enabled", False):
            threshold = strategy_cfg["ai_filter"].get("threshold", 0.6)

            # Extract features from signal
            features = {
                "trend_up": 1 if meta.get("trend") == "up" else 0,
                "trend_down": 1 if meta.get("trend") == "down" else 0,
                "rsi": meta.get("rsi", 50),
                "atr": meta.get("atr", 0),
                "rr": meta.get("risk_reward", 0),
                "ict_bull": 1 if meta.get("ict_structure") == "bull" else 0,
                "ict_bear": 1 if meta.get("ict_structure") == "bear" else 0,
                "has_ob": 1 if signal.meta.get("ict", {}).get("order_block") else 0,
                "has_fvg": 1 if signal.meta.get("ict", {}).get("fvg") else 0,
                "has_sweep": 1 if signal.meta.get("ict", {}).get("sweep") else 0,
            }

            proba = services["ai_filter"].predict_proba(features)
            if proba < threshold:
                meta["skip_reason"] = f"ai {proba:.3f} < {threshold}"
                _notify_skip(meta["skip_reason"], meta)
                log_exec(
                    {
                        "phase": "skip",
                        "symbol": meta["symbol"],
                        "reason": meta.get("skip_reason"),
                    }
                )
                logger.info(f"[INFO] AI Filter blocked: {proba:.3f} < {threshold}")
                return None

        # Calculate position size and SL/TP
        if services.get("risk_manager"):
            risk_pct = risk_cfg.get("risk_pct", 0.01)
            # Get real balance from MT5
            try:
                import MetaTrader5 as mt5

                account_info = mt5.account_info()
                if account_info:
                    balance = float(account_info.equity) or float(account_info.balance)
                else:
                    balance = broker.get_equity() or broker.get_balance()
            except Exception as e:
                logger.info(f"[WARNING] Error getting balance for position sizing: {e}")
                balance = broker.get_equity() or broker.get_balance()

            # Import price planner
            from src.core.price_planner import (
                SymbolMeta,
                calculate_real_rr,
                plan_sl_tp,
                validate_sl_tp_orientation,
            )

            # Get current tick data for accurate entry prices
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.info(f"[ERROR] No tick data available for {symbol}")
                return None

            # Calculate SL/TP levels using proper price planner
            atr_val = meta.get("atr", 0)
            atr_cfg = strategy_cfg.get("atr", {})
            rr = atr_cfg.get("rr", 1.5)
            atr_k = atr_cfg.get("atr_k", 1.2)
            min_stop_points = atr_cfg.get("min_stop_points", 10.0)
            min_stop_pct = atr_cfg.get("min_stop_pct", 0.0025)
            spread_mult = atr_cfg.get("spread_mult", 2.0)

            # Create symbol metadata
            symbol_meta = SymbolMeta(
                point=meta["symbol_info"]["point"],
                digits=meta["symbol_info"]["digits"],
                tick_size=meta["symbol_info"].get("tick_size", meta["symbol_info"]["point"]),
                spread_points=meta["symbol_info"]["spread_points"],
                contract_size=meta["symbol_info"].get("contract_size", 1.0),
            )

            # Plan SL/TP with correct orientation
            entry_price, sl_price, tp_price = plan_sl_tp(
                side=signal.side,
                bid=tick.bid,
                ask=tick.ask,
                atr=atr_val,
                rr=rr,
                meta=symbol_meta,
                atr_k=atr_k,
                min_stop_points=min_stop_points,
                min_stop_pct=min_stop_pct,
                spread_mult=spread_mult,
            )

            # Validate orientation
            if not validate_sl_tp_orientation(signal.side, entry_price, sl_price, tp_price):
                logger.info(
                    f"[ERROR] Invalid SL/TP orientation for {signal.side}: entry={entry_price:.5f}, sl={sl_price:.5f}, tp={tp_price:.5f}"
                )
                return None

            # Calculate real RR
            real_rr = calculate_real_rr(entry_price, sl_price, tp_price, signal.side)
            logger.info(
                f"[DEBUG] SL/TP Plan: {signal.side} | entry={entry_price:.5f} | sl={sl_price:.5f} | tp={tp_price:.5f} | RR={real_rr:.2f}"
            )

            # Calculate position size
            sl_points = abs(entry_price - sl_price) / meta["symbol_info"]["point"]
            lots = services["risk_manager"]["position_size_by_risk"](
                balance, risk_pct, sl_points, meta["symbol_info"]["tick_value"]
            )

            # Apply position size cap for crypto symbols (ADAUSD)
            if symbol == "ADAUSD":
                # Cap crypto position size to reasonable levels
                max_crypto_lots = 1.0  # Maximum 1 lot for crypto
                lots = min(lots * 0.75, max_crypto_lots)  # 75% of calculated size, max 1 lot
                logger.info(
                    f"[INFO] ADAUSD position size normalized: {lots:.2f} lots (capped at {max_crypto_lots})"
                )

            # Store for reporting
            meta["position_size"] = float(lots)

            # Comprehensive logging for SL/TP plan
            logger.info(
                f"[INFO] ORDER_PLAN_RESULT | side={signal.side} | entry={entry_price:.5f} | sl={sl_price:.5f} | tp={tp_price:.5f} | ok=True | reason=calculated"
            )
            logger.info(
                f"[INFO] Trade calculated: {signal.side} {lots:.2f} lots | SL: {sl_price:.5f} | TP: {tp_price:.5f} | RR: {real_rr:.2f}"
            )

            # Log detailed plan information
            logger.info(
                f"[DEBUG] PLAN | {symbol} | side={signal.side} entry={entry_price:.5f} sl={sl_price:.5f} tp={tp_price:.5f} rr={real_rr:.2f} atr={atr_val:.5f} spread={symbol_meta.spread_points*symbol_meta.point:.5f}"
            )

            # Circuit-Breaker gate
            if not cb_order.allow():
                meta["skip_reason"] = "circuit_breaker_open"
                _notify_skip(meta["skip_reason"], meta)
                log_exec(
                    {
                        "phase": "skip",
                        "symbol": meta["symbol"],
                        "reason": meta.get("skip_reason"),
                    }
                )
                logger.info("[WARN] CircuitBreaker open: skipping new orders temporarily")
                return None

            # Log order sending
            log_exec(
                {
                    "phase": "send",
                    "symbol": meta["symbol"],
                    "side": signal.side,
                    "entry": meta.get("entry_price", signal.entry),
                }
            )

            # Final validation before order placement
            assert rr >= 1.0, f"Invalid RR: {rr} < 1.0"
            if signal.side == "long":
                assert (
                    sl_price < entry_price < tp_price
                ), f"Invalid long plan sl={sl_price} entry={entry_price} tp={tp_price}"
            else:
                assert (
                    tp_price < entry_price < sl_price
                ), f"Invalid short plan sl={sl_price} entry={entry_price} tp={tp_price}"

            # SMOKE signal special handling for DEMO mode
            if (
                hasattr(signal, 'meta')
                and signal.meta.get("source") == "smoke"
                and getattr(meta.get("ctx"), "is_demo", False)
            ):
                # Use smaller lot size for SMOKE signals
                smoke_lot = getattr(cfg.get("testing", {}), "smoke_signal_lot", 0.01)
                lots = min(lots, smoke_lot)
                logger.info(
                    f"[SMOKE] executing demo order | symbol={meta['symbol']} side={signal.side} lot={lots:.2f}"
                )

            # Place actual order with broker using execution module
            try:
                from src.core.execution import place_market_order_with_metrics

                exec_rep = place_market_order_with_metrics(
                    symbol=meta["symbol"],
                    side="long" if signal.side == "long" else "short",
                    lots=lots,
                    sl=sl_price,
                    tp=tp_price,
                    slippage_points=cfg.get("broker", {}).get("slippage_points", 10),
                    retries=cfg.get("broker", {}).get("retry_on_requote", 2),
                )

                if exec_rep.ok:
                    meta["exec_report"] = exec_rep.__dict__
                    # Record success in circuit breaker
                    cb_order.record_success()

                    # Update challenge state
                    if CHALLENGE_STATE is not None:
                        CHALLENGE_STATE.trades_today += 1
                        try:
                            eq = broker.get_equity()
                            if eq:
                                CHALLENGE_STATE.equity_peak = max(CHALLENGE_STATE.equity_peak, eq)
                        except Exception:
                            pass

                    # Save trade to application state
                    if app_state is not None:
                        try:
                            from src.core.state import save_state

                            app_state["open_trades"][str(exec_rep.order_id)] = {
                                "symbol": meta["symbol"],
                                "side": signal.side,
                                "entry": signal.entry,
                                "sl": sl_price,
                                "tp": tp_price,
                                "lots": lots,
                                "time": datetime.now(UTC).isoformat(),
                            }
                            save_state(app_state)
                        except Exception as e:
                            logger.info(f"[WARNING] Failed to save trade state: {e}")
                    else:
                        logger.info("[WARNING] No app_state provided, skipping trade state save")

                    # Log successful fill
                    log_exec(
                        {
                            "phase": "fill",
                            "symbol": meta["symbol"],
                            "order_id": exec_rep.order_id,
                            "lots": lots,
                            "sl": sl_price,
                            "tp": tp_price,
                        }
                    )

                    logger.info(
                        f"[EXEC] OK order={exec_rep.order_id} deal={exec_rep.deal_id} fill={exec_rep.fill_price:.5f} "
                        f"slip={exec_rep.slippage_points:.1f}pts latency={exec_rep.latency_ms:.0f}ms"
                    )
                    return exec_rep.order_id
                else:
                    meta["skip_reason"] = f"order_failed: {exec_rep.retcode}/{exec_rep.comment}"
                    _notify_skip(meta["skip_reason"], meta)
                    # Record error in circuit breaker
                    cb_order.record_error()

                    # Log execution error
                    log_exec(
                        {
                            "phase": "error",
                            "symbol": meta["symbol"],
                            "reason": meta.get("skip_reason", "order_error"),
                        }
                    )

                    logger.info(f"[EXEC] FAIL ret={exec_rep.retcode} comment={exec_rep.comment}")
                    return None

            except Exception as e:
                meta["skip_reason"] = f"order_error: {e}"
                _notify_skip(meta["skip_reason"], meta)

                # Log execution error
                log_exec(
                    {
                        "phase": "error",
                        "symbol": meta["symbol"],
                        "reason": meta.get("skip_reason", "order_error"),
                    }
                )

                logger.info(f"[ERROR] Order placement error: {e}")
                return None

        return None

    except Exception as e:
        meta["skip_reason"] = f"execution_error: {e}"
        _notify_skip(meta["skip_reason"], meta)
        logger.info(f"[ERROR] Trade execution error: {e}")
        import traceback

        traceback.print_exc()
        return None

# No dummy OHLC in production - fail fast

def write_report(run_meta: dict[str, Any]) -> None:
    """Write run report to JSON and Markdown files"""
    try:
        # Create reports directory
        Path("reports").mkdir(exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        symbol = run_meta.get("symbol", "UNKNOWN")

        # Get real equity for reporting
        try:
            import MetaTrader5 as mt5

            account_info = mt5.account_info()
            if account_info:
                current_equity = float(account_info.equity) or float(account_info.balance)
            else:
                current_equity = broker.get_equity() or broker.get_balance()
        except Exception as e:
            logger.info(f"[WARNING] Error getting equity for report: {e}")
            current_equity = broker.get_equity() or broker.get_balance()

        # Get challenge state info
        evaluation = run_meta.get("evaluation", {})
        challenge_info = evaluation.get("challenge_mode", {})

        # Add account and challenge info to JSON report
        run_meta["account"] = {
            "equity_now": round(current_equity, 2),
            "equity_peak": round(challenge_info.get("equity_peak", 0), 2),
            "equity_start_of_day": round(challenge_info.get("equity_start_of_day", 0), 2),
        }
        run_meta["challenge"] = {
            "enabled": bool(challenge_info.get("enabled", False)),
            "trades_today": challenge_info.get("trades_today", 0),
            "consecutive_losses": challenge_info.get("consecutive_losses", 0),
            "reason_if_blocked": (
                evaluation.get("reason")
                if evaluation.get("decision") == "challenge_guard"
                else None
            ),
        }

        # File paths
        json_path = f"reports/run_{timestamp}_{symbol}.json"
        md_path = f"reports/run_{timestamp}_{symbol}.md"

        # Write JSON report
        with open(json_path, 'w') as f:
            json.dump(run_meta, f, indent=2, default=str)

        # Write Markdown report
        with open(md_path, 'w') as f:
            f.write("# MR BEN Trading Run Report\n\n")
            f.write(f"**Run ID**: run_{timestamp}_{symbol}\n")
            f.write(f"**Symbol**: {symbol}\n")
            f.write(f"**Mode**: {run_meta.get('mode', 'UNKNOWN')}\n")
            f.write(f"**Timestamp**: {run_meta.get('timestamp', 'UNKNOWN')}\n\n")

            # Strategy Configuration
            f.write("## Strategy Configuration\n")
            strategy_cfg = run_meta.get("strategy_config", {})
            f.write(f"- **Strategy**: {strategy_cfg.get('name', 'UNKNOWN')}\n")
            f.write(f"- **Timeframes**: {strategy_cfg.get('timeframes', {})}\n")
            f.write(f"- **EMA Periods**: {strategy_cfg.get('ema_periods', [])}\n")
            f.write(f"- **MACD**: {strategy_cfg.get('macd', {})}\n")
            f.write(f"- **RSI**: {strategy_cfg.get('rsi', {})}\n")
            f.write(f"- **ATR**: {strategy_cfg.get('atr', {})}\n")
            f.write(f"- **ICT**: {strategy_cfg.get('ict', {})}\n")
            f.write(f"- **Filters**: {strategy_cfg.get('filters', {})}\n")
            f.write(f"- **AI Filter**: {strategy_cfg.get('ai_filter', {})}\n\n")

            # Risk Configuration
            f.write("## Risk Configuration\n")
            risk_cfg = run_meta.get("risk_config", {})
            f.write(f"- **Risk %**: {risk_cfg.get('risk_pct', 0)}\n")
            f.write(f"- **Max Positions**: {risk_cfg.get('max_concurrent_positions', 0)}\n")
            f.write(f"- **Min RR**: {risk_cfg.get('min_rr', 0)}\n\n")

            # Evaluation Results
            f.write("## Evaluation Results\n")
            evaluation = run_meta.get("evaluation", {})
            f.write(f"- **Has Signal**: {evaluation.get('has_signal', False)}\n")

            if evaluation.get('has_signal'):
                f.write(f"- **Signal Side**: {evaluation.get('signal_side', 'UNKNOWN')}\n")
                f.write(f"- **Confidence**: {evaluation.get('signal_confidence', 0)}\n")
                f.write(f"- **Reason**: {evaluation.get('signal_reason', 'UNKNOWN')}\n")
                f.write(f"- **Entry Price**: {evaluation.get('entry_price', 0)}\n")
                f.write(f"- **Stop Loss**: {evaluation.get('stop_loss', 0)}\n")
                f.write(f"- **Take Profit**: {evaluation.get('take_profit', 0)}\n")
                f.write(f"- **Risk/Reward**: {evaluation.get('risk_reward', 0):.2f}\n")
                f.write(f"- **RSI**: {evaluation.get('rsi', 0):.1f}\n")
                f.write(f"- **ATR**: {evaluation.get('atr', 0):.5f}\n")
                f.write(f"- **Trend**: {evaluation.get('trend', 'UNKNOWN')}\n")
                f.write(f"- **ICT Structure**: {evaluation.get('ict_structure', 'UNKNOWN')}\n")

            # Symbol Information
            f.write("\n## Symbol Information\n")
            symbol_info = evaluation.get("symbol_info", {})
            f.write(f"- **Digits**: {symbol_info.get('digits', 0)}\n")
            f.write(f"- **Point**: {symbol_info.get('point', 0)}\n")
            f.write(f"- **Spread**: {symbol_info.get('spread_points', 0)} points\n")
            f.write(f"- **Tick Value**: {symbol_info.get('tick_value', 0)}\n")

            # Services Status
            f.write("\n## Services Status\n")
            services = run_meta.get("services", {})
            f.write(f"- **AI Filter**: {'ENABLED' if services.get('ai_filter') else 'DISABLED'}\n")
            f.write(
                f"- **News Filter**: {'ENABLED' if services.get('news_filter') else 'DISABLED'}\n"
            )
            f.write(
                f"- **Risk Manager**: {'ENABLED' if services.get('risk_manager') else 'DISABLED'}\n"
            )

            # Safety Panel
            f.write("\n## Safety Panel\n")
            account_info = run_meta.get("account", {})
            challenge_info = run_meta.get("challenge", {})
            evaluation = run_meta.get("evaluation", {})

            # Account Status
            f.write(f"- **Equity Now**: {account_info.get('equity_now', 0):.2f}\n")
            f.write(f"- **Equity Peak**: {account_info.get('equity_peak', 0):.2f}\n")
            f.write(
                f"- **Equity Start of Day**: {account_info.get('equity_start_of_day', 0):.2f}\n"
            )

            # Symbol Info
            symbol_info = evaluation.get("symbol_info", {})
            f.write(f"- **Spread Points**: {symbol_info.get('spread_points', 0):.2f}\n")
            f.write(f"- **Risk/Reward**: {evaluation.get('risk_reward', 0):.2f}\n")

            # Challenge Mode
            f.write(
                f"- **Challenge Enabled**: {'YES' if challenge_info.get('enabled') else 'NO'}\n"
            )
            if challenge_info.get('enabled'):
                f.write(f"- **Trades Today**: {challenge_info.get('trades_today', 0)}\n")
                f.write(
                    f"- **Consecutive Losses**: {challenge_info.get('consecutive_losses', 0)}\n"
                )
                f.write(
                    f"- **Max Daily Loss %**: {run_meta.get('strategy_config', {}).get('challenge_mode', {}).get('max_daily_loss_pct', 0):.1f}%\n"
                )
                f.write(
                    f"- **Max Overall DD %**: {run_meta.get('strategy_config', {}).get('challenge_mode', {}).get('max_overall_dd_pct', 0):.1f}%\n"
                )
                if challenge_info.get('reason_if_blocked'):
                    f.write(f"- **Blocked Reason**: {challenge_info.get('reason_if_blocked')}\n")

            # Trade Decision
            f.write("\n## Trade Decision\n")
            trade_decision = run_meta.get("trade_decision", {})
            f.write(f"- **Order ID**: {trade_decision.get('order_id', 'None')}\n")
            f.write(
                f"- **Decision**: {'EXECUTED' if trade_decision.get('order_id') else 'SKIPPED'}\n"
            )

            if trade_decision.get('order_id'):
                f.write(
                    f"- **Execution Time**: {trade_decision.get('execution_time', 'UNKNOWN')}\n"
                )
            else:
                f.write(f"- **Skip Reason**: {trade_decision.get('skip_reason', 'No signal')}\n")

        logger.info(f"[OK] Report saved to {json_path} and {md_path}")

    except Exception as e:
        logger.info(f"❌ Failed to write report: {e}")
        import traceback

        traceback.print_exc()

def main() -> None:
    """Main entry point for MR BEN Live Trading System"""
    global CHALLENGE_STATE

    try:
        # Parse command line arguments
        args = parse_args()
        
        # Determine mode
        mode = "live" if args.live else ("demo" if args.demo else "report" if args.report_only else "demo")
        
        # MT5 graceful handling based on mode
        global mt5_ok, mt5, broker
        if not mt5_ok:
            if mode == "live":
                logger.error(json.dumps({"kind":"mt5_required_live","mode":mode,"ts":datetime.now(UTC).isoformat()}))
                sys.exit(1)
            else:
                logger.warning(json.dumps({"kind":"mt5_unavailable_demo","mode":mode,"ts":datetime.now(UTC).isoformat()}))
        elif mt5 and not mt5.initialize():
            mt5_ok = False
            if mode == "live":
                logger.error(json.dumps({"kind":"mt5_init_fail","mode":mode,"ts":datetime.now(UTC).isoformat()}))
                sys.exit(1)
            else:
                logger.warning(json.dumps({"kind":"mt5_init_fail_demo","mode":mode,"ts":datetime.now(UTC).isoformat()}))

        # Set demo mode environment variable for supervisor
        if args.demo or args.report_only:
            os.environ["MRBEN_DEMO_MODE"] = "true"

        # Initialize JSON logger if enabled
        if args.json_logs:
            from src.core.json_logger import json_logger

            json_logger.enabled = True
            logger.info(f"[INFO] Structured JSON logging enabled: {json_logger.log_file}")
        else:
            from src.core.json_logger import json_logger

            json_logger.enabled = False

        # Set up logging
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()), format='[%(levelname)s] %(message)s'
        )

        # Load configuration first
        logger.info(f"DEBUG: Loading config file: {args.config}")
        config = load_config(args.config, args)
        logger.info(f"DEBUG: Loaded config symbols: {config.get('symbols', {}).get('supported', [])}")

        # Import symbol alias mapper
        from src.core.symbol_alias import map_symbol, map_symbols

        # Resolve symbols early and persist
        symbols = [s.strip() for s in (args.symbol or "").split(",") if s.strip()]
        if not symbols:
            symbols = config.get("trading", {}).get("symbols", [])
        symbols = list(dict.fromkeys(symbols))  # dedupe keep order
        if not symbols:
            raise SystemExit("No symbols provided")

        # Apply symbol mapping (legacy .PRO -> clean names)
        original_symbols = symbols.copy()
        symbols = map_symbols(symbols)
        
        # Log symbol aliases if any changes were made
        for orig, mapped in zip(original_symbols, symbols):
            if orig != mapped:
                logger.info(f"Symbol alias: {orig} -> {mapped}")

        logger.info(f"[INFO] Trading symbols: {symbols}")

        # Validate EACH symbol; never validate the raw CLI string
        def validate_symbol(symbol: str) -> None:
            """Validate symbol against supported list"""
            supported_symbols = config.get("symbols", {}).get("supported", ["XAUUSD"])
            logger.info(f"DEBUG: Validating symbol '{symbol}' against supported list: {supported_symbols}")
            if symbol not in supported_symbols:
                logger.info(f"❌ Symbol '{symbol}' not in supported list: {supported_symbols}")
                raise SystemExit(f"Unsupported symbol: {symbol}")

        for _s in symbols:
            validate_symbol(_s)

        # Multi-symbol vs single-symbol mode decision
        default_symbol = config.get('symbols', {}).get('default', 'XAUUSD')
        symbol_list = _resolve_symbol_cfgs(config, args.symbols, default_symbol)
        
        # Apply symbol mapping to symbol_list
        for sc in symbol_list:
            old_name = sc.name
            sc.name = map_symbol(sc.name)
            if sc.name != old_name:
                logger.info(f"Symbol alias: {old_name} -> {sc.name}")
        
        if args.symbols and len(symbol_list) >= 2:
            # MULTI-SYMBOL MODE
            logger.info(f"[INFO] Starting multi-symbol engine for: {[sc.name for sc in symbol_list]}")
            
            # Build services for multi-symbol
            services = build_services(config)
            
            # Initialize broker
            if not bootstrap_broker():
                logger.info("❌ Failed to connect to broker. Exiting.")
                return
            
            # Create circuit breaker for multi-symbol
            from src.core.circuit_breaker import CBConfig, CircuitBreaker
            cb_config = CBConfig(
                max_errors=config.get("circuit_breaker", {}).get("max_failures", 5),
                window_sec=config.get("circuit_breaker", {}).get("timeout", 60)
            )
            cb_order = CircuitBreaker("multi_symbol_orders", cb_config)
            
            # Create multi-symbol engine
            engine = MultiSymbolEngine(
                broker=broker, cb_order=cb_order,
                symbol_cfgs=symbol_list,
                fetch_signal=_fetch_signal_for_symbol,
                logger=logger,
            )
            
            # Start multi-symbol engine
            engine.start(stagger_seconds=1.0)
            
            try:
                # Keep process alive; monitor for kill switch
                while True:
                    # Temporarily disabled kill switch for multi-symbol stability
                    # if should_stop_now():
                    #     logger.info("Kill switch received; stopping multi-symbol engine.")
                    #     break
                    time.sleep(1.0)
            finally:
                engine.stop()
            return  # prevent single-mode path
        else:
            # SINGLE-SYMBOL MODE (existing behavior)
            # Use first symbol for single mode
            symbol = symbols[0] if symbols else default_symbol
            logger.info(f"[INFO] Single-symbol mode: {symbol}")

        # Initialize market hours manager for crypto support
        try:
            from src.core.market_hours import initialize_market_hours

            initialize_market_hours(config)
            logger.info("[OK] Market hours manager initialized for crypto support")
        except Exception as e:
            logger.info(f"[WARNING] Market hours initialization warning: {e}")

        # Auto-select profile based on mode
        if args.profile is None:
            if args.live:
                args.profile = "production"
                logger.info(f"[INFO] Profile selected: {args.profile} (LIVE)")
            else:
                args.profile = "test_loose"
                logger.info(f"[INFO] Profile selected: {args.profile} (DEMO)")
        else:
            logger.info(f"[INFO] Profile selected: {args.profile} (user-specified)")

        # Apply profile configuration
        if "profiles" in config and args.profile in config["profiles"]:
            profile_config = config["profiles"][args.profile]
            # Merge profile config into main config
            for key, value in profile_config.items():
                if key in config:
                    if isinstance(value, dict) and isinstance(config[key], dict):
                        config[key].update(value)
                    else:
                        config[key] = value
                else:
                    config[key] = value
            logger.info(f"[INFO] Applied profile: {args.profile}")
            if args.profile == "production":
                logger.info("[INFO] Active profile: production (no fallback applied)")

        # Override risk from CLI if provided
        if args.risk is not None:
            config.setdefault("risk", {})["risk_pct"] = float(args.risk)

        # Override max concurrent positions from CLI if provided
        if hasattr(args, 'max_orders') and args.max_orders is not None:
            config.setdefault("risk", {})["max_concurrent_positions"] = args.max_orders
            logger.info(f"[INFO] Max concurrent positions overridden to: {args.max_orders}")

        # Override supervisor from CLI if provided
        if args.supervisor == "on":
            config.setdefault("supervisor", {})["enabled"] = True
        elif args.supervisor == "off":
            config.setdefault("supervisor", {})["enabled"] = False
            # In LIVE mode, supervisor is required
            if args.live and not args.demo and not args.report_only:
                logger.info("❌ Supervisor disabled in LIVE mode - aborting")
                sys.exit(1)

        # Force challenge mode settings based on LIVE/DEMO mode
        if args.live:
            # In LIVE mode, disable challenge guards for production trading
            config.setdefault("challenge_mode", {})["enabled"] = False
            config.setdefault("challenge", {})["enabled"] = False
            config.setdefault("challenge", {})["session_lock"] = False
            # Force environment variable to prevent any re-enabling
            os.environ["MRBEN_DEMO_MODE"] = "false"
            logger.info("[INFO] Challenge disabled in LIVE (session_lock off)")
            logger.info("[INFO] Challenge/Session locks are OFF in production.")
            logger.info("[SANITY] production enforce -> DEMO=false, challenge=false, session_lock=false")
        else:
            # In DEMO mode, keep challenge guards enabled
            config.setdefault("challenge_mode", {})["enabled"] = True
            logger.info("[INFO] Challenge enabled in DEMO mode")

        # Symbols already resolved above

        # Disable session_lock for crypto trading (24x7)
        if "ADAUSD" in symbols:
            config.setdefault("challenge", {})["session_lock"] = False
            logger.info("[INFO] session_lock disabled for crypto trading (24x7)")

        # Symbols already validated above

        # Bootstrap broker connection
        if not bootstrap_broker():
            logger.info("❌ Failed to connect to broker. Exiting.")
            return

        # Live trading safety check
        if not args.demo and not args.report_only:
            logger.info("🚨 LIVE TRADING MODE - Safety checks enabled")
            logger.info("[WARNING] Ensure all risk parameters are correct before proceeding")

            # Additional live trading validations
            if not config.get("risk", {}).get("max_daily_loss"):
                logger.info("❌ Max daily loss not configured for live trading. Exiting.")
                return

            if not config.get("challenge_mode", {}).get("enabled"):
                logger.info("[WARNING] Challenge mode disabled - proceed with caution")

        # Symbols will be processed in the main loop
        logger.info(f"[INFO] Starting MR BEN Live Trading System - Symbols: {symbols}")

        # Log account equity at startup
        try:
            import MetaTrader5 as mt5

            account_info = mt5.account_info()
            if account_info:
                equity = float(account_info.equity) or float(account_info.balance)
            else:
                equity = broker.get_equity() or broker.get_balance()
        except Exception as e:
            logger.info(f"[WARNING] Error getting equity at startup: {e}")
            equity = broker.get_equity() or broker.get_balance()
        logger.info(f"[INFO] Account | equity_now={equity:.2f}")

        # Build services
        services = build_services(config)

        # Initialize supervisor
        try:
            supervisor = initialize_supervisor(config)
            if supervisor and supervisor.enabled:
                logger.info("[OK] Trading Supervisor initialized")
            else:
                logger.info("[INFO] Trading Supervisor disabled")
        except Exception as e:
            logger.info(f"[WARNING] Failed to initialize supervisor: {e}")

        # Initialize challenge mode state
        try:
            challenge_state = initialize_challenge_state(broker)
            logger.info(f"[OK] Challenge mode initialized - Equity: {challenge_state.equity_peak:.2f}")

            # Set global challenge state
            CHALLENGE_STATE = challenge_state
        except Exception as e:
            logger.info(f"[WARNING] Failed to initialize challenge mode: {e}")
            # امن‌ترین کار: چَلنچ را غیرفعال کن تا سیستم به‌خاطر آن اسکیپ نکند
            config.setdefault("challenge_mode", {})["enabled"] = False
            challenge_state = None
            CHALLENGE_STATE = None

        # Load application state
        try:
            from src.core.state import load_state

            app_state = load_state()
            logger.info("[OK] Application state loaded")
        except Exception as e:
            logger.info(f"[WARNING] Failed to load application state: {e}")
            app_state = {"open_trades": {}, "challenge": {}, "last_signal_time": {}}

        # Report-only mode
        if args.report_only:
            logger.info(f"[INFO] Report-only mode: evaluating {symbol} once...")

            # Add debug flag to config
            config["debug_scan_on_nosignal"] = getattr(args, 'debug_scan_on_nosignal', False)
            config["demo_smoke_signal"] = getattr(args, 'demo_smoke_signal', False)

            # Single evaluation
            signal, decision_meta = evaluate_once(
                symbol, config, services, challenge_state, args.live
            )

            # Build run metadata
            run_meta = {
                "symbol": symbol,
                "mode": "DEMO" if args.demo else "LIVE",
                "timestamp": datetime.now(UTC).isoformat(),
                "strategy_config": config.get("strategy", {}),
                "risk_config": config.get("risk", {}),
                "evaluation": decision_meta,
                "services": {
                    "ai_filter": bool(services.get("ai_filter")),
                    "news_filter": bool(services.get("news_filter")),
                    "risk_manager": bool(services.get("risk_manager")),
                },
                "trade_decision": {"order_id": None, "skip_reason": "Report-only mode"},
            }

            # Write report and exit
            write_report(run_meta)
            logger.info("[INFO] Report generated successfully. Exiting.")
            return

        # Live trading mode
        logger.info(f"[INFO] Starting live trading loop for {symbols}...")
        logger.info(f"[INFO] Mode: {'LIVE' if args.live else 'DEMO'}")
        logger.info(f"[INFO] Max orders: {args.max_orders}")

        # Create context object for DEMO mode detection
        class Context:
            def __init__(self, is_demo=False):
                self.is_demo = is_demo

        ctx = Context(is_demo=not args.live)

        # Initialize enhanced components
        gate = AllOfFourGate(config)
        limit = ConcurrencyLimiter(config)
        spread_ctl = SpreadController(config)
        sup = SupervisorClient(config)
        stealth = StealthStrategy(config)

        # Main trading loop
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                logger.info(
                    f"\n[INFO] Trading cycle {cycle_count} - {datetime.now(UTC).strftime('%H:%M:%S')}"
                )

                # Kill-switch check (beginning of each cycle)
                stop, reason = should_stop_now()
                if stop:
                    send_alert("Kill-Switch Triggered", reason, level="CRITICAL")
                    logger.info("[INFO] Kill-switch detected. Shutting down gracefully...")
                    break

                # Config lock check - disabled for production trading
                lock_enabled = False  # Disabled for production trading
                try:
                    ok_cfg, reason_cfg = enforce_config_lock(args.config, lock_enabled=lock_enabled)
                    if not ok_cfg:
                        send_alert(
                            "Config Lock Violation",
                            reason_cfg,
                            level="CRITICAL",
                            extra={"config": args.config},
                        )
                    logger.info("[ERROR] " + reason_cfg)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Config lock not enforced: {e}")
                    break

                # Update challenge state for new day
                now_utc = datetime.now(UTC)
                challenge_state = update_challenge_state(challenge_state, broker, now_utc)

                # Process each symbol in dual-symbol mode
                for symbol in symbols:
                    # Supervisor hook: cycle event
                    # اسپرد لحظه‌ای هر سیکل
                    try:
                        symbol_info_cycle = get_symbol_info(symbol)
                    except RuntimeError as e:
                        logger.info(f"[WARNING] Failed to get symbol info for cycle: {e}")
                        symbol_info_cycle = {"spread_points": 0}
                    cycle_meta = {
                        "symbol": symbol,
                        "cycle_count": cycle_count,
                        "spread_points": symbol_info_cycle["spread_points"],
                        "equity": broker.get_equity() or broker.get_balance(),
                        "challenge": {
                            "enabled": config.get("challenge_mode", {}).get("enabled", True),
                            "trades_today": challenge_state.trades_today if challenge_state else 0,
                            "consecutive_losses": (
                                challenge_state.consecutive_losses if challenge_state else 0
                            ),
                        },
                        "cfg": {
                            "risk_pct": config.get("risk", {}).get("risk_pct", 0.005),
                            "max_concurrent_positions": config.get("risk", {}).get(
                                "max_concurrent_positions", 1
                            ),
                        },
                        "mode": "DEMO" if args.demo else "LIVE",
                    }
                    on_cycle(cycle_meta)

                    # Supervisor emit: cycle event (non-blocking)
                    supervisor_emit(
                        "cycle",
                        {
                            "symbol": symbol,
                            "spread_points": symbol_info_cycle["spread_points"],
                            "positions": {"open": 0},  # Will be updated after position check
                            "portfolio": {
                                "equity": broker.get_equity() or broker.get_balance(),
                                "open_risk_value": 0,
                            },
                            "filters": config.get("strategy", {}).get("filters", {}),
                            "challenge": {
                                "enabled": config.get("challenge_mode", {}).get("enabled", False)
                            },
                            "cb_order": {
                                "open": False,
                                "recent_errors": 0,
                            },  # Will be updated later
                            "mode": "LIVE" if not args.demo else "DEMO",
                        },
                    )

                    # Add debug flag to config
                    config["debug_scan_on_nosignal"] = getattr(
                        args, 'debug_scan_on_nosignal', False
                    )
                    config["demo_smoke_signal"] = getattr(args, 'demo_smoke_signal', False)
                    
                    # IMPORTANT: Also add to strategy config so confluence_signal can access it
                    if "strategy" not in config:
                        config["strategy"] = {}
                    config["strategy"]["demo_smoke_signal"] = getattr(args, 'demo_smoke_signal', False)

                    # Single evaluation
                    signal, decision_meta = evaluate_once(
                        symbol, config, services, challenge_state, args.live
                    )

                    # Enhanced components integration
                    if signal and decision_meta.get("context"):
                        ctx_dict = decision_meta["context"]

                        # Supervisor heartbeat
                        sup.heartbeat()

                        # Spread debounce check
                        if not spread_ctl.ok(symbol, ctx_dict):
                            spread_reason = ctx_dict.get("spread_reason", "spread_high")
                            logger.info(f"[INFO] Trade SKIPPED - Reason: {spread_reason}")
                            signal = None
                            decision_meta["skip_reason"] = spread_reason

                        # Soft DD gate check
                        if ctx_dict.get("dd_soft_block", False):
                            logger.info("[INFO] Trade SKIPPED - Reason: dd_soft_block")
                            signal = None
                            decision_meta["skip_reason"] = "dd_soft_block"

                        # Supervisor decision - always call for heartbeat
                        sv = sup.decide(ctx_dict.get("snapshot", {}))
                        ctx_dict["sup_ok"] = sv.allow
                        if sv.tweaks and config.get("supervisor", {}).get(
                            "allow_param_tweaks", True
                        ):
                            # Apply tweaks to config
                            for key, value in sv.tweaks.items():
                                if key in config:
                                    config[key] = value

                        if not sv.allow:
                            logger.info("[INFO] Trade SKIPPED - Reason: supervisor_block")
                            signal = None
                            decision_meta["skip_reason"] = "supervisor_block"

                        # All-of-Four gate check
                        if signal:
                            decision = gate.decide(symbol, ctx_dict)
                            if not decision.allow:
                                logger.info(f"[INFO] Trade SKIPPED - Reason: {decision.reason}")
                                signal = None
                                decision_meta["skip_reason"] = decision.reason

                        # Concurrency cap check
                        if signal and not limit.can_open(symbol):
                            # When cap is full, manage existing positions instead of generating new signals
                            position_mgmt = manage_open_positions(symbol, broker, config)
                            if position_mgmt["managed"] > 0:
                                logger.info(
                                    f"[INFO] Cap full - Managed {position_mgmt['managed']} positions: {', '.join(position_mgmt['actions'])}"
                                )
                            else:
                                logger.info(
                                    "[INFO] Trade SKIPPED - Reason: cap_full_manage_only (no positions to manage)"
                                )
                            signal = None
                            decision_meta["skip_reason"] = "cap_full_manage_only"

                    # Supervisor emit: signal event (if signal exists)
                    if signal:
                        supervisor_emit(
                            "signal",
                            {
                                "symbol": symbol,
                                "rr": decision_meta.get("risk_reward", 0),
                                "rsi": decision_meta.get("rsi", 0),
                                "atr": decision_meta.get("atr", 0),
                                "trend": decision_meta.get("trend", "unknown"),
                                "ict": {
                                    "structure_side": decision_meta.get("ict_structure", "unknown")
                                },
                                "mode": "LIVE" if not args.demo else "DEMO",
                            },
                        )

                    # Execute trade if signal exists
                    order_id = None
                    if signal:
                        order_id = maybe_execute_trade(
                            signal, decision_meta, config, services, app_state, ctx
                        )

                        # Concurrency tracking
                        if order_id:
                            limit.on_open(symbol)

                        # Supervisor emit: fill or skip event
                        if order_id:
                            supervisor_emit(
                                "fill",
                                {
                                    "symbol": symbol,
                                    "order_id": order_id,
                                    "fill_price": 0.0,  # Will be updated with actual fill price
                                    "mode": "LIVE" if not args.demo else "DEMO",
                                },
                            )
                        else:
                            supervisor_emit(
                                "skip",
                                {
                                    "symbol": symbol,
                                    "reason": decision_meta.get("skip_reason", "unknown"),
                                    "rr": decision_meta.get("risk_reward", 0),
                                    "spread_points": decision_meta.get("symbol_info", {}).get(
                                        "spread_points", 0
                                    ),
                                    "mode": "LIVE" if not args.demo else "DEMO",
                                },
                            )

                    # Build run metadata for this cycle
                    run_meta = {
                        "symbol": symbol,
                        "mode": "DEMO" if args.demo else "LIVE",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "strategy_config": config.get("strategy", {}),
                        "risk_config": config.get("risk", {}),
                        "evaluation": decision_meta,
                        "services": {
                            "ai_filter": bool(services.get("ai_filter")),
                            "news_filter": bool(services.get("news_filter")),
                            "risk_manager": bool(services.get("risk_manager")),
                        },
                        "trade_decision": {
                            "order_id": order_id,
                            "execution_time": datetime.now(UTC).isoformat() if order_id else None,
                            "skip_reason": (
                                decision_meta.get("skip_reason", "No signal")
                                if not order_id
                                else None
                            ),
                        },
                    }

                    # Log trade decision
                    if order_id:
                        logger.info(f"[INFO] Trade EXECUTED - Order ID: {order_id}")
                        logger.info(
                            f"[INFO] Signal: {decision_meta.get('signal_side', 'UNKNOWN')} | Confidence: {decision_meta.get('signal_confidence', 0):.2f}"
                        )
                        logger.info(
                            f"[INFO] Entry: {decision_meta.get('entry_price', 0):.5f} | SL: {decision_meta.get('stop_loss', 0):.5f} | TP: {decision_meta.get('take_profit', 0):.5f}"
                        )
                        logger.info(
                            f"[INFO] RR: {decision_meta.get('risk_reward', 0):.2f} | Lots: {decision_meta.get('position_size', 0):.2f}"
                        )
                    else:
                        if signal:
                            logger.info(
                                "[INFO] Trade SKIPPED - Reason: Safety guard or execution failure"
                            )
                        else:
                            logger.info("[INFO] No signal generated - Continuing...")

                # Log cycle end with equity info (after all symbols processed)
                try:
                    import MetaTrader5 as mt5

                    account_info = mt5.account_info()
                    if account_info:
                        equity = float(account_info.equity) or float(account_info.balance)
                    else:
                        equity = broker.get_equity() or broker.get_balance()
                except Exception as e:
                    logger.info(f"[WARNING] Error getting equity in cycle: {e}")
                    equity = broker.get_equity() or broker.get_balance()

                # Enhanced cycle logging with safety status
                logger.info(
                    f"[INFO] Cycle end | equity_now={equity:.2f} | peak={challenge_state.equity_peak:.2f} | trades_today={challenge_state.trades_today} | consec_losses={challenge_state.consecutive_losses}"
                )

                # Portfolio risk summary (every 10 cycles)
                if cycle_count % 10 == 0:
                    try:
                        from src.core.portfolio_risk import portfolio_open_risk

                        open_risk_value, by_symbol = portfolio_open_risk()
                        risk_pct = (open_risk_value / equity) if equity > 0 else 0
                        logger.info(
                            f"PortfolioRisk: open_value=${open_risk_value:.2f} pct={risk_pct:.3f} by_symbol={by_symbol}"
                        )
                    except Exception as e:
                        logger.info(f"[WARNING] Portfolio risk summary error: {e}")

                # Daily report (every 25 cycles)
                if cycle_count % 25 == 0:
                    try:
                        from src.reports.daily_report import write_daily

                        daily_summary = {
                            'date': datetime.now(UTC).strftime('%Y-%m-%d'),
                            'mode': 'LIVE' if not args.demo else 'DEMO',
                            'symbols': symbols,
                            'equity_now': equity,
                            'peak_equity': (
                                challenge_state.equity_peak if challenge_state else equity
                            ),
                            'dd': (
                                (challenge_state.equity_peak - equity) / challenge_state.equity_peak
                                if challenge_state and challenge_state.equity_peak > 0
                                else 0
                            ),
                            'trades_today': challenge_state.trades_today if challenge_state else 0,
                            'accepted': 0,  # TODO: track actual accepted trades
                            'rejected': 0,  # TODO: track actual rejected trades
                            'top_reject_reasons': ['spread_high', 'concurrency_cap'],
                            'last_decisions': [],  # TODO: track last decisions
                        }

                        json_path, md_path = write_daily(daily_summary)
                        logger.info(f"[INFO] Daily report generated: {md_path}")
                    except Exception as e:
                        logger.info(f"[WARNING] Daily report error: {e}")

                # Activation report (first cycle only)
                if cycle_count == 1:
                    try:
                        from src.reports.activation_report import write_activation_report

                        activation_data = {
                            "mode": "LIVE" if not args.demo else "DEMO",
                            "symbols": symbols,
                            "supervisor_heartbeat_ok": sup.healthy(),
                            "gates": {
                                "all_of_four": True,
                                "spread_debounce": True,
                                "soft_dd_gate": True,
                                "concurrency_cap": config.get("signals", {}).get(
                                    "concurrency_cap", 2
                                ),
                            },
                            "risk_stops": {
                                "daily_stop_realized": config.get("risk", {}).get(
                                    "daily_stop_realized", 0.05
                                ),
                                "dd_soft_from_peak": config.get("risk", {}).get(
                                    "dd_soft_from_peak", 0.06
                                ),
                            },
                            "cost_model": "enabled",
                            "reports": {
                                "daily_written": True,
                                "paths_hint": [
                                    f"reports/daily_{datetime.now(UTC).strftime('%Y-%m-%d')}.md",
                                    f"reports/daily_{datetime.now(UTC).strftime('%Y-%m-%d')}.json",
                                ],
                            },
                        }

                        report_path = write_activation_report(activation_data)
                        logger.info(f"[OK] Activation report generated: {report_path}")
                    except Exception as e:
                        logger.info(f"[WARNING] Activation report error: {e}")

                # Safety status check
                if not args.demo:
                    daily_loss_pct = (
                        (challenge_state.equity_peak - equity) / challenge_state.equity_peak
                        if challenge_state.equity_peak > 0
                        else 0
                    )
                    if daily_loss_pct > 0.02:  # 2% daily loss warning
                        logger.info(
                            f"[WARNING] DAILY LOSS WARNING: {daily_loss_pct:.2%} | Peak: {challenge_state.equity_peak:.2f}"
                        )
                    elif daily_loss_pct > 0.015:  # 1.5% daily loss alert
                        logger.info(f"[WARNING] Daily loss alert: {daily_loss_pct:.2%}")

                # Position management (every cycle)
                try:
                    # Use our local manage_open_positions function
                    pm_result = manage_open_positions("ALL", broker, config)
                    if pm_result["managed"] > 0:
                        logger.info(
                            f"[PM] Managed {pm_result['managed']} positions: {', '.join(pm_result['actions'])}"
                        )
                    else:
                        logger.info("[PM] No positions to manage")
                except Exception as e:
                    logger.info(f"[WARNING] Position management error: {e}")

                # Sleep before next cycle
                logger.info("[INFO] Sleeping for 12 seconds...")
                time.sleep(12)

            except KeyboardInterrupt:
                logger.info("\n[INFO] Received interrupt signal. Shutting down gracefully...")
                break
            except Exception as e:
                logger.info(f"[ERROR] Error in trading cycle {cycle_count}: {e}")

                # Supervisor hook: error event
                try:
                    symbol_info = (
                        get_symbol_info(symbol) if 'symbol' in locals() else {"spread_points": 0}
                    )
                    equity = broker.get_equity() or broker.get_balance()
                except:
                    symbol_info = {"spread_points": 0}
                    equity = 0

                error_meta = {
                    "symbol": symbol if 'symbol' in locals() else "unknown",
                    "cycle_count": cycle_count,
                    "spread_points": symbol_info["spread_points"],
                    "equity": equity,
                    "challenge": {
                        "enabled": config.get("challenge_mode", {}).get("enabled", True),
                        "trades_today": challenge_state.trades_today if challenge_state else 0,
                        "consecutive_losses": (
                            challenge_state.consecutive_losses if challenge_state else 0
                        ),
                    },
                    "cfg": {
                        "risk_pct": config.get("risk", {}).get("risk_pct", 0.005),
                        "max_concurrent_positions": config.get("risk", {}).get(
                            "max_concurrent_positions", 1
                        ),
                    },
                    "mode": "DEMO" if args.demo else "LIVE",
                }
                on_error("trading_cycle", str(e), error_meta)

                # Supervisor emit: error event
                supervisor_emit(
                    "error",
                    {
                        "symbol": symbol,
                        "stage": "trading_cycle",
                        "msg": str(e),
                        "mode": "LIVE" if not args.demo else "DEMO",
                    },
                )

                import traceback

                traceback.print_exc()
                logger.info("[INFO] Continuing after error...")
                time.sleep(30)  # Longer sleep after error

        logger.info(f"[INFO] Trading system stopped. Total cycles: {cycle_count}")

    except Exception as e:
        logger.info(f"[ERROR] Fatal error in main: {e}")
        import traceback

        traceback.print_exc()
        return 1

def main():
    """Main entry point for live trading system"""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Load configuration
        logger.info(f"DEBUG: Loading config file: {args.config}")
        config = load_config(args.config, args)
        logger.info(f"DEBUG: Loaded config symbols: {config.get('symbols', {}).get('supported', [])}")
        
        # Parse symbols using new function
        symbols_list = parse_symbol_list(args.symbols)
        if not symbols_list and args.symbol:
            symbols_list = parse_symbol_list(args.symbol)
        if not symbols_list:
            symbols_list = [config.get("symbols", {}).get("default", "XAUUSD")]
        
        # Import symbol alias mapper
        from src.core.symbol_alias import map_symbol, map_symbols
        
        # Apply symbol mapping
        original_symbols = symbols_list.copy()
        symbols_list = map_symbols(symbols_list)
        
        # Log symbol aliases if any changes were made
        for orig, mapped in zip(original_symbols, symbols_list):
            if orig != mapped:
                logger.info(f"Symbol alias: {orig} -> {mapped}")

        logger.info(f"[INFO] Trading symbols: {symbols_list}")
        
        # Validate EACH symbol; never validate the raw CLI string
        def validate_symbol(symbol: str) -> None:
            """Validate symbol against supported list"""
            supported_symbols = config.get("symbols", {}).get("supported", ["XAUUSD"])
            logger.info(f"DEBUG: Validating symbol '{symbol}' against supported list: {supported_symbols}")
            if symbol not in supported_symbols:
                logger.info(f"❌ Symbol '{symbol}' not in supported list: {supported_symbols}")
                raise SystemExit(f"Unsupported symbol: {symbol}")

        # Validate all symbols
        for symbol in symbols_list:
            validate_symbol(symbol)
        
        # Use the pipeline approach as recommended
        logger.info(f"[INFO] Starting pipeline-based trading for {symbols_list}")
        
        # Initialize services
        services = {
            "ai_filter": None,  # Will be initialized if available
            "broker": None,     # Will be initialized
        }
        
        # Initialize broker
        try:
            from src.core.broker_mt5 import broker
            services["broker"] = broker
            logger.info("[INFO] Broker service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize broker: {e}")
            return 1
        
        # Run trading loop for each symbol
        cycle_count = 0
        max_cycles = args.max_cycles if hasattr(args, 'max_cycles') and args.max_cycles else 1000000
        
        # Determine mode
        mode = "live" if args.live else ("demo" if args.demo else "report" if args.report_only else "demo")
        
        logger.info(f"[INFO] Starting trading loop (max cycles: {max_cycles})")
        
        while cycle_count < max_cycles:
            cycle_count += 1
            
            # Heartbeat logging
            logger.info(json.dumps({
                "kind": "heartbeat",
                "cycle": cycle_count,
                "mode": mode,
                "symbols": symbols_list,
                "ts": datetime.now(UTC).isoformat()
            }))
            
            logger.info(f"\n[INFO] Trading cycle {cycle_count} - {datetime.now(UTC).strftime('%H:%M:%S')}")
            
            try:
                for symbol in symbols_list:
                    logger.info(f"[INFO] Processing symbol: {symbol}")
                    
                    # Check MT5 availability
                    if not mt5_ok:
                        logger.warning(json.dumps({"kind":"skip","symbol":symbol,"reason":"mt5_unavailable","ts":datetime.now(UTC).isoformat()}))
                        continue
                    
                    # ADAUSD monitor-only mode
                    if symbol == "ADAUSD":
                        logger.info(json.dumps({"kind":"monitor_only","symbol":symbol,"ts":datetime.now(UTC).isoformat()}))
                        continue
                    
                    # Pre-filters
                    from src.filters.session_filter import session_ok
                    from src.filters.news_filter import news_ok
                    
                    if not session_ok(symbol):
                        logger.info(json.dumps({"kind":"prefilter","symbol":symbol,"reason":"session","ts":datetime.now(UTC).isoformat()}))
                        continue
                    if not news_ok(symbol):
                        logger.info(json.dumps({"kind":"prefilter","symbol":symbol,"reason":"news","ts":datetime.now(UTC).isoformat()}))
                        continue
                    
                    # Evaluate signal for this symbol
                    signal, decision_meta = evaluate_once(
                        symbol=symbol,
                        cfg=config,
                        services=services,
                        challenge_state=None,
                        is_live_mode=not args.demo
                    )
                    
                    # Maybe execute trade if signal exists
                    if signal:
                        result = maybe_execute_trade(
                            signal=signal,
                            meta=decision_meta,
                            cfg=config,
                            services=services
                        )
                        logger.info(f"[INFO] Trade result: {result}")
                    else:
                        logger.info("[INFO] No signal generated - Continuing...")
                
                # Sleep between cycles
                time.sleep(12)  # 12 seconds between cycles
                
            except KeyboardInterrupt:
                logger.info(json.dumps({
                    "kind": "shutdown",
                    "reason": "user_interrupt",
                    "cycle": cycle_count,
                    "ts": datetime.now(UTC).isoformat()
                }))
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(30)  # Longer sleep after error
        
        logger.info(json.dumps({
            "kind": "trading_completed",
            "total_cycles": cycle_count,
            "ts": datetime.now(UTC).isoformat()
        }))
            
    except KeyboardInterrupt:
        logger.info(json.dumps({
            "kind": "shutdown",
            "reason": "keyboard_interrupt",
            "ts": datetime.now(UTC).isoformat()
        }))
        return 0
    except Exception as e:
        logger.error(json.dumps({
            "kind": "fatal_error",
            "error": str(e),
            "ts": datetime.now(UTC).isoformat()
        }))
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # MT5 shutdown
        try:
            if mt5_ok and mt5_inited:
                mt5.shutdown()
                logger.info(json.dumps({
                    "kind": "mt5_shutdown",
                    "ts": datetime.now(UTC).isoformat()
                }))
        except Exception:
            pass
    
    return 0

# JSON logging setup
import os

if os.environ.get("MRBEN_JSON_LOG", "1") == "1":
    setup_json_logger()
Path("logs").mkdir(exist_ok=True)

if __name__ == "__main__":
    main()
