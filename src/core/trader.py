"""
Main trading system orchestrator for MR BEN Trading System.
"""

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

# Global MT5 availability flag
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# Import local modules
try:
    # Try absolute imports from src root
    from ai.system import MRBENAdvancedAISystem
    from config.settings import MT5Config
    from core.exceptions import TradingSystemError
    from core.metrics import PerformanceMetrics
    from data.manager import MT5DataManager
    from execution.executor import EnhancedTradeExecutor
    from indicators.atr import compute_atr
    from risk.manager import EnhancedRiskManager
    from risk_manager.atr_sl_tp import calc_sltp_from_atr
    from signals.multi_tf_rsi_macd import analyze_multi_tf_rsi_macd
    from utils.helpers import (
        _apply_soft_gate,
        _rolling_atr,
        _swing_extrema,
        enforce_min_distance_and_round,
    )
    from utils.memory import cleanup_memory, log_memory_usage
    from utils.position_management import (
        _count_open_positions,
        _get_open_positions,
        _prune_trailing_registry,
    )
except ImportError:
    # Simplified fallback - just import what we can
    MRBENAdvancedAISystem = None
    MT5Config = None
    MT5DataManager = None
    EnhancedTradeExecutor = None
    EnhancedRiskManager = None
    _apply_soft_gate = None
    _rolling_atr = None
    _swing_extrema = None
    enforce_min_distance_and_round = None
    cleanup_memory = None
    log_memory_usage = None
    _count_open_positions = None
    _get_open_positions = None
    _prune_trailing_registry = None
    TradingSystemError = None
    PerformanceMetrics = None

# Bot Version
BOT_VERSION = "4.1.0"


class MT5LiveTrader:
    """Main trading system orchestrator."""

    def __init__(self):
        """Initialize the trading system."""
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.startup_time = time.time()

        # Initialize config with error handling
        try:
            self.config = MT5Config()
        except Exception as e:
            print(f"❌ Failed to initialize config: {e}")
            raise

        # Setup logging
        self._setup_logging()

        # Initialize basic state variables first
        self._initialize_basic_state()

        # Log initial memory usage
        log_memory_usage(self.logger, "Initial memory usage")

        # Initialize components with error handling
        self._initialize_components()

        # Initialize component-dependent state variables
        self._initialize_component_state()

        # Log startup completion
        startup_time = time.time() - self.startup_time
        self.logger.info(f"🚀 MT5LiveTrader initialized in {startup_time:.2f}s")
        log_memory_usage(self.logger, "Post-initialization memory usage")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        # Dedicated logger for SL/TP decisions with more specific naming
        self.sltp_logger = logging.getLogger("core.trader.sltp")
        self.logger.setLevel(getattr(logging, self.config.LOG_LEVEL.upper(), logging.INFO))

        # Logging root
        os.makedirs("logs", exist_ok=True)
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=getattr(logging, self.config.LOG_LEVEL.upper(), logging.INFO),
                format='[%(asctime)s][%(levelname)s] %(message)s',
            )

        # Add rotating file handler
        try:
            from logging.handlers import RotatingFileHandler

            os.makedirs(os.path.dirname(self.config.LOG_FILE), exist_ok=True)
            fh = RotatingFileHandler(
                self.config.LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding='utf-8'
            )
            fh.setLevel(getattr(logging, self.config.LOG_LEVEL.upper(), logging.INFO))
            fh.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s'))

            # Check if similar handler already exists
            root = logging.getLogger()
            exists = any(
                isinstance(h, logging.handlers.RotatingFileHandler)
                and getattr(h, 'baseFilename', '') == os.path.abspath(self.config.LOG_FILE)
                for h in root.handlers
            )
            if not exists:
                root.addHandler(fh)
        except Exception as e:
            self.logger.warning(f"File logging setup failed: {e}")

    def _initialize_components(self) -> None:
        """Initialize all system components with error handling."""
        try:
            # Risk Manager
            self.risk_manager = EnhancedRiskManager(
                base_risk=self.config.BASE_RISK,
                min_lot=self.config.MIN_LOT,
                max_lot=self.config.MAX_LOT,
                max_open_trades=self.config.MAX_OPEN_TRADES,
                base_confidence_threshold=0.35,
                tf_minutes=self.config.TIMEFRAME_MIN,
            )
            self.logger.info("✅ Risk Manager initialized")

            # Trade Executor
            self.trade_executor = EnhancedTradeExecutor(self.risk_manager)
            self.logger.info("✅ Trade Executor initialized")

            # Data Manager
            self.data_manager = MT5DataManager(self.config.SYMBOL, self.config.TIMEFRAME_MIN)
            self.logger.info("✅ Data Manager initialized")

            # MT5 connect for trading
            self.mt5_connected = self._initialize_mt5_for_trading()
            if self.mt5_connected:
                self.logger.info("✅ MT5 connection established")
            else:
                self.logger.warning("⚠️ MT5 connection failed - running in demo mode")

            # AI system
            self.ai_system = MRBENAdvancedAISystem()
            self.logger.info("🤖 Initializing AI System...")
            self.ai_system.load_models()
            self.logger.info(
                f"✅ AI System loaded. Available models: {list(self.ai_system.models.keys())}"
            )

            # Conformal Gate (optional)
            self.conformal = None
            try:
                from utils.conformal import ConformalGate

                self.conformal = ConformalGate("models/meta_filter.joblib", "models/conformal.json")
                self.logger.info("✅ Conformal gate loaded.")
            except Exception as e:
                self.logger.warning(f"⚠️ Conformal not available: {e}")

            # Event Logger (optional)
            try:
                from telemetry import EventLogger

                os.makedirs("data", exist_ok=True)
                self.ev = EventLogger(
                    path="data/events.jsonl", run_id=self.run_id, symbol=self.config.SYMBOL
                )
                self.logger.info("✅ Event Logger initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Event Logger not available: {e}")
                self.ev = None

            # Config hash for tracking changes
            self.config_hash = hashlib.sha256(
                json.dumps(self.config.config_data, sort_keys=True).encode()
            ).hexdigest()[:12]
            self.logger.info(f"BOT_VERSION={BOT_VERSION} CONFIG_HASH={self.config_hash}")

        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise

    def _initialize_basic_state(self) -> None:
        """Initialize basic state variables that don't depend on components."""
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.running = False
        self.consecutive_signals = 0
        self.last_signal = 0
        self.last_trailing_update = datetime.now()
        self.trailing_update_interval = 15  # sec - faster response for trailing
        self.start_balance: float | None = None

        # Bar-gate state
        self.last_bar_time = None

        # Trailing registry for position management
        self.trailing_registry = {}
        self.trailing_step = 0.5  # Default trailing step in price units

        # Chandelier trailing engine (optional)
        try:
            from utils.trailing import ChandelierTrailing, TrailParams

            self.trailing_engine = ChandelierTrailing(TrailParams(k_atr=1.5, min_step=0.2))
        except ImportError:
            self.trailing_engine = None
            self.logger.warning("⚠️ Chandelier trailing not available")

        # MFE Logger (optional)
        try:
            from telemetry import MFELogger

            self.mfe_logger = MFELogger("data/mfe_tick_data.jsonl")
        except ImportError:
            self.mfe_logger = None
            self.logger.warning("⚠️ MFE Logger not available")

        # Re-entry window system
        self.reentry_window_until = None
        self._prev_open_count = 0

        # Flat-state detection and reset system
        self._was_open = 0
        self.cooldown_until = None
        self.reentry_lock_until = None
        self.last_trade_side = None  # +1 buy / -1 sell / None
        self.last_fill_time = None

        # Warm-up guard
        self.start_time = datetime.now()

        # In-bar evaluation control
        self._last_inbar_eval = None

        # Spread threshold
        self.max_spread_points = int(self.config.MAX_SPREAD_POINTS)

    def _initialize_component_state(self) -> None:
        """Initialize state variables that depend on components."""
        # Dynamic SL/TP parameters (optimized for achievable TPs)
        rconf = self.config.config_data.get("risk", {})
        self.sl_mult_base = float(rconf.get("sl_atr_multiplier", 1.6))
        self.tp_mult_base = min(float(rconf.get("tp_atr_multiplier", 3.0)), 2.2)

        # ATR-based SL/TP parameters
        self.atr_period = int(rconf.get("atr_period", 14))
        self.rr = float(rconf.get("rr", 1.5))
        self.sl_k = float(rconf.get("sl_k", 1.0))
        self.tp_k = float(rconf.get("tp_k", 1.5))
        self.fallback_sl_pct = float(rconf.get("fallback_sl_pct", 0.005))
        self.fallback_tp_pct = float(rconf.get("fallback_tp_pct", 0.0075))

        # Multi-timeframe RSI/MACD parameters
        mtf_conf = self.config.config_data.get("multi_tf", {})
        self.mtf_timeframes = list(mtf_conf.get("timeframes", ["M5", "M15", "H1"]))
        self.mtf_rsi_period = int(mtf_conf.get("rsi_period", 14))
        self.mtf_macd_fast = int(mtf_conf.get("macd_fast", 12))
        self.mtf_macd_slow = int(mtf_conf.get("macd_slow", 26))
        self.mtf_macd_signal = int(mtf_conf.get("macd_signal", 9))
        self.mtf_rsi_overbought = int(mtf_conf.get("rsi_overbought", 70))
        self.mtf_rsi_oversold = int(mtf_conf.get("rsi_oversold", 30))
        self.mtf_min_agreement = int(mtf_conf.get("min_agreement", 2))

        self.conf_min = float(self.risk_manager.base_confidence_threshold)
        self.conf_max = 0.90
        self.k_sl = 0.35  # SL adjustment coefficient based on confidence
        self.k_tp = 0.20  # Reduced from 0.50 to 0.20
        self.swing_lookback = int(
            self.config.config_data.get("advanced", {}).get("swing_lookback", 12)
        )
        self.max_spread_atr_frac = float(
            self.config.config_data.get("advanced", {}).get("dynamic_spread_atr_frac", 0.10)
        )

        # Spread execution parameters
        execution_cfg = self.config.config_data.get("execution", {})
        self.spread_eps = float(execution_cfg.get("spread_eps", 0.02))
        self.use_spread_ma = bool(execution_cfg.get("use_spread_ma", True))
        self.spread_ma_window = int(execution_cfg.get("spread_ma_window", 5))
        self.spread_hysteresis_factor = float(execution_cfg.get("spread_hysteresis_factor", 1.05))

        # Spread buffer for MA calculation
        self._spread_buf = []
        self._spread_last_ok = None

        # TP Policy for volume splitting and breakeven
        tp_policy_cfg = self.config.config_data.get("tp_policy", {})
        self.tp_policy = {
            "split": tp_policy_cfg.get("split", True),
            "tp1_r": tp_policy_cfg.get("tp1_r", 0.8),
            "tp2_r": tp_policy_cfg.get("tp2_r", 1.5),
            "tp1_share": tp_policy_cfg.get("tp1_share", 0.5),
            "breakeven_after_tp1": tp_policy_cfg.get("breakeven_after_tp1", True),
        }
        self.min_R_after_round = 1.2

        # Performance monitoring
        self.last_memory_check = time.time()
        self.memory_check_interval = 300  # Check memory every 5 minutes

    def _initialize_mt5_for_trading(self) -> bool:
        """Initialize MT5 connection for trading."""
        if not MT5_AVAILABLE:
            self.logger.info("MT5 not available - trading disabled.")
            return False

        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False

            if self.config.LOGIN and self.config.PASSWORD and self.config.SERVER:
                if not mt5.login(
                    login=self.config.LOGIN,
                    password=self.config.PASSWORD,
                    server=self.config.SERVER,
                ):
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False

            info = mt5.account_info()
            sym = mt5.symbol_info(self.config.SYMBOL)
            if not sym:
                self.logger.error(f"Symbol not found: {self.config.SYMBOL}")
                return False

            if not sym.visible:
                if not mt5.symbol_select(self.config.SYMBOL, True):
                    self.logger.error(f"Cannot select symbol {self.config.SYMBOL}")
                    return False

            self.logger.info(
                f"✅ MT5 connected | Account: {info.login} Balance: {info.balance:.2f} Equity: {info.equity:.2f}"
            )
            self.logger.info(
                f"Symbol: {self.config.SYMBOL} MinLot: {sym.volume_min} MaxLot: {sym.volume_max} Step: {sym.volume_step}"
            )
            return True
        except Exception as e:
            self.logger.error(f"MT5 connect error: {e}")
            return False

    def start(self) -> None:
        """Start the trading system."""
        print(f"🎯 MR BEN Live Trading System {BOT_VERSION}")
        print("=" * 60)

        if not self._validate_system():
            self.logger.error("System validation failed.")
            return

        if self.mt5_connected:
            acc = mt5.account_info()
            if acc:
                self.start_balance = acc.balance
                self.logger.info(
                    f"Account: {acc.login} | Balance: {acc.balance:.2f} | Equity: {acc.equity:.2f}"
                )

        self.logger.info(f"[STARTUP] MR BEN Live Trading System {BOT_VERSION} Starting...")
        self.logger.info(f"RUN_ID={self.run_id}")
        self.logger.info(
            f"CONFIG: SYMBOL={self.config.SYMBOL}, TF={self.config.TIMEFRAME_MIN}, BARS={self.config.BARS}, MAGIC={self.config.MAGIC}"
        )
        self.logger.info(
            f"RISK: base={self.config.BASE_RISK}, conf_thresh={self.risk_manager.base_confidence_threshold}, max_open={self.config.MAX_OPEN_TRADES}"
        )
        self.logger.info(
            f"LIMITS: daily_loss={self.config.MAX_DAILY_LOSS}, max_trades={self.config.MAX_TRADES_PER_DAY}, sessions={self.config.SESSIONS}, risk_based_vol={self.config.USE_RISK_BASED_VOLUME}"
        )

        # Emit start event if available
        if self.ev:
            try:
                acc = mt5.account_info() if self.mt5_connected else None
                self.ev.emit(
                    "bot_start",
                    account_login=int(acc.login) if acc else None,
                    balance=float(acc.balance) if acc else None,
                    equity=float(acc.equity) if acc else None,
                    config={
                        "tf": self.config.TIMEFRAME_MIN,
                        "max_open": self.config.MAX_OPEN_TRADES,
                    },
                )
            except Exception as e:
                self.logger.warning(f"Failed to emit start event: {e}")

        # Preflight check
        if not self._preflight_check():
            self.logger.error("Preflight failed. Not starting loop.")
            return

        # Kill-switch equity
        self.session_start_equity = None
        if self.mt5_connected:
            ai = mt5.account_info()
            if ai:
                self.session_start_equity = ai.equity
                self.logger.info(f"Session start equity: {self.session_start_equity:.2f}")

        self.running = True
        self.logger.info("🔁 Trading loop is running")
        t = threading.Thread(target=self._trading_loop, daemon=True)
        t.start()
        self.logger.info("[SUCCESS] Trading loop started")

        # Bootstrap trailing for existing positions
        self._bootstrap_trailing()

    def stop(self) -> None:
        """Stop the trading system with proper cleanup."""
        self.logger.info("🛑 Stopping trading system...")
        self.running = False

        try:
            # Cleanup data manager
            if hasattr(self, 'data_manager') and self.data_manager:
                self.data_manager.shutdown()
                self.logger.info("✅ Data manager shutdown complete")
        except Exception as e:
            self.logger.warning(f"Data manager shutdown failed: {e}")

        # Clean MT5 shutdown
        try:
            if MT5_AVAILABLE and mt5.initialize():
                mt5.shutdown()
                self.logger.info("✅ MT5 shutdown complete")
        except Exception as e:
            self.logger.warning(f"MT5 shutdown failed: {e}")

        # Cleanup event logger
        try:
            if hasattr(self, "ev") and self.ev:
                self.ev.close()
                self.logger.info("✅ Event logger closed")
        except Exception as e:
            self.logger.warning(f"Event logger close failed: {e}")

        # Final cleanup
        self._cleanup_resources()

        # Log final performance metrics
        if hasattr(self, 'metrics'):
            try:
                stats = self.metrics.get_stats()
                self.logger.info("📊 Final Performance Summary:")
                self.logger.info(f"   Total Runtime: {stats['uptime_seconds']:.0f}s")
                self.logger.info(f"   Total Cycles: {stats['cycle_count']}")
                self.logger.info(f"   Total Trades: {stats['total_trades']}")
                self.logger.info(f"   Error Rate: {stats['error_rate']:.3f}")
                self.logger.info(f"   Final Memory: {stats['memory_mb']:.1f} MB")
            except Exception as e:
                self.logger.warning(f"Failed to log final metrics: {e}")

        self.logger.info("[SUCCESS] Trading system stopped successfully")

    def _validate_system(self) -> bool:
        """Validate system components."""
        ok = True
        if self.data_manager is None:
            self.logger.error("❌ Data manager not ready")
            ok = False
        if self.risk_manager is None:
            self.logger.error("❌ Risk manager not ready")
            ok = False
        if self.trade_executor is None:
            self.logger.error("❌ Trade executor not ready")
            ok = False
        if self.ai_system is None:
            self.logger.error("❌ AI system not ready")
            ok = False
        return ok

    def _preflight_check(self) -> bool:
        """Perform preflight checks before starting."""
        try:
            ok = True
            if MT5_AVAILABLE and self.mt5_connected:
                sym = mt5.symbol_info(self.config.SYMBOL)
                if not sym:
                    self.logger.error(f"Preflight: symbol not found {self.config.SYMBOL}")
                    ok = False
                else:
                    tick = mt5.symbol_info_tick(self.config.SYMBOL)
                    if not tick:
                        self.logger.warning("Preflight: no tick available")
                    else:
                        spread_price = tick.ask - tick.bid
                        self.logger.info(
                            f"Preflight: spread={spread_price:.5f} ask/bid=({tick.ask:.2f}/{tick.bid:.2f})"
                        )
                    self.logger.info(
                        f"Preflight: stops_level={getattr(sym,'trade_stops_level',None)} freeze_level={getattr(sym,'trade_freeze_level',None)}"
                    )

            if not os.path.exists('models'):
                self.logger.warning("Preflight: models folder not found")

            os.makedirs(os.path.dirname(self.config.LOG_FILE), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.TRADE_LOG_PATH), exist_ok=True)
            return ok
        except Exception as e:
            self.logger.error(f"Preflight error: {e}")
            return False

    def _bootstrap_trailing(self) -> None:
        """Register existing open positions into trailing on startup/restart."""
        try:
            if not MT5_AVAILABLE:
                return

            open_pos = _get_open_positions(
                self.config.SYMBOL, self.config.MAGIC, self.trailing_registry
            )
            self.trailing_registry = {}
            tick = mt5.symbol_info_tick(self.config.SYMBOL)
            mid = ((tick.bid + tick.ask) / 2.0) if tick else None

            for ticket, p in open_pos.items():
                is_buy = p.type == mt5.POSITION_TYPE_BUY
                highest = (mid if mid is not None else p.price_open) if is_buy else float('inf')
                lowest = float('-inf') if is_buy else (mid if mid is not None else p.price_open)

                highest = p.price_open if is_buy else highest
                lowest = lowest if not is_buy else lowest

                self.trailing_registry[ticket] = {
                    "dir": "buy" if is_buy else "sell",
                    "entry": float(p.price_open),
                    "sl": float(p.sl) if p.sl else None,
                    "tp": float(p.tp) if p.tp else None,
                    "highest": highest,
                    "lowest": lowest,
                }

            if open_pos:
                self.logger.info(
                    f"🔗 Trailing bootstrap: registered {len(open_pos)} open positions."
                )
        except Exception as e:
            self.logger.error(f"Trailing bootstrap error: {e}")

    def _cleanup_resources(self) -> None:
        """Clean up system resources and memory."""
        try:
            # Clear trailing registry
            if hasattr(self, 'trailing_registry'):
                self.trailing_registry.clear()
                self.logger.info("🧹 Trailing registry cleared")

            # Clear any cached data
            if hasattr(self, 'data_manager') and self.data_manager:
                if hasattr(self.data_manager, 'current_data'):
                    self.data_manager.current_data = None

            # Force garbage collection
            cleanup_memory()

            # Log final memory usage
            log_memory_usage(self.logger, "Post-cleanup memory usage")

        except Exception as e:
            self.logger.warning(f"Resource cleanup failed: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'running') and self.running:
                self.stop()
        except Exception:
            pass

    def _trading_loop(self):
        """Main trading loop - orchestrates the entire trading process."""
        self.logger.info("🔄 Trading loop started")

        while self.running:
            try:
                cycle_start = time.time()

                # Memory management
                if time.time() - self.last_memory_check > self.memory_check_interval:
                    log_memory_usage(self.logger, "Periodic memory check")
                    self.last_memory_check = time.time()

                    # Force cleanup if needed
                    from utils.memory import force_cleanup_if_needed

                    if force_cleanup_if_needed(threshold_mb=1000.0, logger=self.logger):
                        self.logger.info("🧹 Memory cleanup performed")

                # Check if we should continue trading
                if not self._should_continue_trading():
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue

                # Get market data
                df = self.data_manager.get_latest_data(self.config.BARS)
                if df is None or len(df) < 50:
                    self.logger.warning("Insufficient market data, skipping cycle")
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue

                # Check spread
                spread_ok, spread_points, threshold = self._check_spread()
                if not spread_ok:
                    self.logger.info(
                        f"Spread too high: {spread_points:.1f} > {threshold:.1f}, waiting..."
                    )
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue

                # Get current tick for execution
                tick = self.data_manager.get_current_tick()
                if not tick:
                    self.logger.warning("No current tick available, skipping cycle")
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue

                # Check for new bar (bar-gate)
                if not self._is_new_bar(df):
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue

                # Get open positions
                open_positions = _get_open_positions(
                    self.config.SYMBOL, self.config.MAGIC, self.trailing_registry
                )
                open_count = len(open_positions)

                # Update trailing stops
                if time.time() - self.last_trailing_update > self.trailing_update_interval:
                    self._update_trailing_stops()
                    self.last_trailing_update = time.time()

                # Check if we can open new trades
                if not self._can_open_new_trade(open_count):
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue

                # Generate AI signal
                signal_data = self._generate_trading_signal(df, tick)
                if not signal_data or signal_data.get('signal') == 0:
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue

                # Execute trade if conditions are met
                if self._should_execute_trade(signal_data, open_count):
                    success = self._execute_trade(signal_data, df)
                    if success:
                        self.metrics.record_trade()
                        self.consecutive_signals = 0
                        self.last_signal = signal_data.get('signal', 0)
                    else:
                        self.logger.warning("Trade execution failed")

                # Record cycle metrics
                cycle_time = time.time() - cycle_start
                self.metrics.record_cycle(cycle_time)

                # Sleep between cycles
                time.sleep(self.config.SLEEP_SECONDS)

            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                self.metrics.record_error()
                time.sleep(self.config.RETRY_DELAY)

        self.logger.info("🔄 Trading loop stopped")

    def _should_continue_trading(self) -> bool:
        """Check if trading should continue based on various conditions."""
        try:
            # Check if system is running
            if not self.running:
                return False

            # Check session restrictions
            if not self._is_trading_session():
                return False

            # Check daily limits
            if self._daily_limits_exceeded():
                return False

            # Check equity protection
            if self._equity_protection_triggered():
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking trading conditions: {e}")
            return False

    def _is_trading_session(self) -> bool:
        """Check if current time is within trading sessions."""
        try:
            if not self.config.SESSIONS:
                return True

            # Get current time in configured timezone
            import pytz

            tz = pytz.timezone(self.config.SESSION_TZ)
            now = datetime.now(tz)
            current_hour = now.hour

            # Check if current hour falls within any trading session
            for session in self.config.SESSIONS:
                if session.lower() == "london" and 8 <= current_hour < 16:
                    return True
                elif session.lower() == "ny" and 16 <= current_hour < 24:
                    return True
                elif session.lower() == "asia" and 0 <= current_hour < 8:
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking trading session: {e}")
            return True  # Default to allowing trading

    def _daily_limits_exceeded(self) -> bool:
        """Check if daily trading limits have been exceeded."""
        try:
            if not self.mt5_connected:
                return False

            # Check daily loss limit
            if self.start_balance and self.start_balance > 0:
                account_info = self.trade_executor.get_account_info()
                current_equity = account_info.get('equity', 0)
                daily_loss = (self.start_balance - current_equity) / self.start_balance
                if daily_loss > self.config.MAX_DAILY_LOSS:
                    self.logger.warning(
                        f"Daily loss limit exceeded: {daily_loss:.2%} > {self.config.MAX_DAILY_LOSS:.2%}"
                    )
                    return True

            # Check daily trade count
            if (
                hasattr(self, '_daily_trade_count')
                and self._daily_trade_count >= self.config.MAX_TRADES_PER_DAY
            ):
                self.logger.warning(f"Daily trade limit exceeded: {self._daily_trade_count}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking daily limits: {e}")
            return False

    def _equity_protection_triggered(self) -> bool:
        """Check if equity protection should stop trading."""
        try:
            if not self.mt5_connected or not self.start_balance:
                return False

            account_info = self.trade_executor.get_account_info()
            current_equity = account_info.get('equity', 0)

            # Check for significant equity drop
            equity_drop = (self.start_balance - current_equity) / self.start_balance
            if equity_drop > 0.05:  # 5% equity drop
                self.logger.warning(f"Equity protection triggered: {equity_drop:.2%} drop")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking equity protection: {e}")
            return False

    def _check_spread(self) -> tuple[bool, float, float]:
        """Check if current spread is acceptable."""
        try:
            from utils.helpers import is_spread_ok

            return is_spread_ok(self.config.SYMBOL, self.max_spread_points)
        except Exception as e:
            self.logger.error(f"Error checking spread: {e}")
            return True, 0.0, float(self.max_spread_points)

    def _is_new_bar(self, df: pd.DataFrame) -> bool:
        """Check if we have a new bar since last check."""
        try:
            if df is None or len(df) == 0:
                return False

            current_bar_time = df['time'].iloc[-1]
            if self.last_bar_time is None:
                self.last_bar_time = current_bar_time
                return True

            if current_bar_time > self.last_bar_time:
                self.last_bar_time = current_bar_time
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking new bar: {e}")
            return False

    def _can_open_new_trade(self, open_count: int) -> bool:
        """Check if we can open a new trade."""
        try:
            # Check open trade count
            if open_count >= self.config.MAX_OPEN_TRADES:
                return False

            # Check cooldown period
            if self.cooldown_until and datetime.now() < self.cooldown_until:
                return False

            # Check re-entry lock
            if self.reentry_lock_until and datetime.now() < self.reentry_lock_until:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking trade eligibility: {e}")
            return False

    def _generate_trading_signal(
        self, df: pd.DataFrame, tick: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate trading signal using AI system."""
        try:
            # Prepare market data for AI
            market_data = {
                'time': tick['time'],
                'open': tick['bid'],
                'high': tick['ask'],
                'low': tick['bid'],
                'close': (tick['bid'] + tick['ask']) / 2,
                'volume': tick.get('volume', 0),
            }

            # Generate ensemble signal
            signal = self.ai_system.generate_ensemble_signal(market_data)

            # Apply conformal gate if available
            if self.conformal and signal.get('signal') != 0:
                try:
                    p_value = self.conformal.get_p_value(market_data)
                    if p_value is not None:
                        signal['p_value'] = p_value
                        # Apply soft gate
                        adj_thr, req_consec, override_margin = _apply_soft_gate(
                            p_value, self.conf_min, self.config.CONSECUTIVE_SIGNALS_REQUIRED
                        )
                        signal['adjusted_threshold'] = adj_thr
                        signal['required_consecutive'] = req_consec
                except Exception as e:
                    self.logger.warning(f"Conformal gate error: {e}")

            return signal

        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return None

    def _should_execute_trade(self, signal_data: dict[str, Any], open_count: int) -> bool:
        """Determine if trade should be executed based on signal and conditions."""
        try:
            signal = signal_data.get('signal', 0)
            confidence = signal_data.get('confidence', 0.0)

            # Check signal strength
            if signal == 0:
                return False

            # Check confidence threshold
            threshold = signal_data.get('adjusted_threshold', self.conf_min)
            if confidence < threshold:
                return False

            # Check consecutive signals requirement
            required = signal_data.get(
                'required_consecutive', self.config.CONSECUTIVE_SIGNALS_REQUIRED
            )
            if self.consecutive_signals < required:
                self.consecutive_signals += 1
                self.logger.info(
                    f"Building consecutive signals: {self.consecutive_signals}/{required}"
                )
                return False

            # Check if signal direction changed
            if self.last_signal != 0 and self.last_signal != signal:
                self.logger.info(f"Signal direction changed: {self.last_signal} -> {signal}")
                self.consecutive_signals = 0

            return True

        except Exception as e:
            self.logger.error(f"Error checking trade execution: {e}")
            return False

    def _execute_trade(self, signal_data: dict[str, Any], df: pd.DataFrame) -> bool:
        """Execute the trading signal."""
        try:
            signal = signal_data.get('signal', 0)
            confidence = signal_data.get('confidence', 0.0)

            if signal == 0:
                return False

            # Multi-timeframe RSI/MACD consensus check
            mtf_consensus = self._check_multi_tf_consensus()
            if mtf_consensus == "neutral":
                self.logger.info("Multi-timeframe consensus: NEUTRAL - skipping trade")
                return False
            elif mtf_consensus != ("buy" if signal > 0 else "sell"):
                self.logger.info(
                    f"Multi-timeframe consensus: {mtf_consensus} - signal mismatch, skipping trade"
                )
                return False

            # Get current market data
            tick = self.data_manager.get_current_tick()
            if not tick:
                self.logger.error("No current tick for trade execution")
                return False

            # Determine trade direction
            trade_type = "BUY" if signal > 0 else "SELL"
            entry_price = tick['ask'] if signal > 0 else tick['bid']

            # Calculate position size
            account_info = self.trade_executor.get_account_info()
            balance = account_info.get('balance', 10000.0)

            # Calculate ATR-based SL/TP
            # Get latest data for ATR calculation
            df = self.data_manager.get_latest_data(self.atr_period + 1)
            if df is not None and len(df) >= self.atr_period:
                # Calculate ATR from the data
                atr_series = compute_atr(df, self.atr_period)
                atr_value = atr_series.iloc[-1] if not atr_series.empty else None
                self.sltp_logger.debug(
                    f"ATR calculated from {len(df)} candles, period: {self.atr_period}"
                )
            else:
                atr_value = None
                self.sltp_logger.warning(
                    "Insufficient data for ATR calculation, using fallback percentages"
                )

            # Use new ATR-based SL/TP calculator
            sltp_result = calc_sltp_from_atr(
                side="buy" if trade_type == "BUY" else "sell",
                entry_price=entry_price,
                atr_value=atr_value,
                rr=self.rr,
                sl_k=self.sl_k,
                tp_k=self.tp_k,
                fallback_sl_pct=self.fallback_sl_pct,
                fallback_tp_pct=self.fallback_tp_pct,
            )

            sl, tp = sltp_result.sl, sltp_result.tp

            # Enhanced professional logging for ATR SL/TP decisions
            self.sltp_logger.info(
                f"ATR SL/TP Decision | "
                f"Side: {trade_type} | "
                f"Entry: {entry_price:.5f} | "
                f"ATR: {atr_value:.5f if atr_value else 'N/A'} | "
                f"RR: {self.rr} | "
                f"SL_k: {self.sl_k} | "
                f"TP_k: {self.tp_k} | "
                f"Fallback_SL_pct: {self.fallback_sl_pct} | "
                f"Fallback_TP_pct: {self.fallback_tp_pct} | "
                f"Result_SL: {sl:.5f} | "
                f"Result_TP: {tp:.5f} | "
                f"Used_Fallback: {sltp_result.used_fallback}"
            )

            # Calculate lot size
            sl_distance = abs(entry_price - sl)
            lot_size = self.risk_manager.calculate_lot_size(
                balance, self.config.BASE_RISK, sl_distance, self.config.SYMBOL
            )

            # Validate lot size
            if lot_size < self.config.MIN_LOT:
                self.logger.warning(
                    f"Calculated lot size too small: {lot_size} < {self.config.MIN_LOT}"
                )
                return False

            # Place order
            order_params = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.SYMBOL,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if signal > 0 else mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": self.config.MAGIC,
                "comment": f"MRBEN_{self.run_id}_{signal_data.get('source', 'AI')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = self.trade_executor.place_order(order_params)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(
                    f"✅ {trade_type} order executed: {result.order} | "
                    f"Price: {entry_price:.2f} | Lot: {lot_size} | SL: {sl:.2f} | TP: {tp:.2f}"
                )

                # Add trailing stop
                self.risk_manager.add_trailing_stop(result.order, entry_price, sl, signal > 0)

                # Update daily trade count
                if not hasattr(self, '_daily_trade_count'):
                    self._daily_trade_count = 0
                self._daily_trade_count += 1

                # Set cooldown
                self.cooldown_until = datetime.now() + timedelta(
                    seconds=self.config.COOLDOWN_SECONDS
                )

                # Emit trade event
                if self.ev:
                    try:
                        self.ev.emit(
                            "trade_executed",
                            signal=signal,
                            confidence=confidence,
                            price=entry_price,
                            lot_size=lot_size,
                            sl=sl,
                            tp=tp,
                            source=signal_data.get('source', 'AI'),
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to emit trade event: {e}")

                return True
            else:
                self.logger.error(
                    f"❌ Order failed: {getattr(result, 'retcode', 'Unknown')} | "
                    f"{getattr(result, 'comment', 'No comment')}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False

    def _check_multi_tf_consensus(self) -> str:
        """Check multi-timeframe RSI/MACD consensus."""
        try:
            # Collect data for all timeframes
            timeframe_data = {}

            for tf in self.mtf_timeframes:
                try:
                    # Convert timeframe string to minutes for data fetching
                    tf_minutes = self._convert_timeframe_to_minutes(tf)
                    if tf_minutes is None:
                        self.logger.warning(f"Unknown timeframe: {tf}, skipping")
                        continue

                    # Fetch data for this timeframe
                    df = self.data_manager.get_latest_data(100, tf_minutes)  # Get 100 bars
                    if df is not None and len(df) >= 50:  # Ensure sufficient data
                        timeframe_data[tf] = df
                    else:
                        self.logger.warning(f"Insufficient data for timeframe {tf}, skipping")
                except Exception as e:
                    self.logger.warning(f"Error fetching data for timeframe {tf}: {e}")
                    continue

            if len(timeframe_data) < 2:
                self.logger.warning("Insufficient timeframe data for consensus check")
                return "neutral"

            # Analyze RSI/MACD for each timeframe
            signals = analyze_multi_tf_rsi_macd(
                timeframe_data,
                rsi_period=self.mtf_rsi_period,
                macd_fast=self.mtf_macd_fast,
                macd_slow=self.mtf_macd_slow,
                macd_signal=self.mtf_macd_signal,
                rsi_overbought=self.mtf_rsi_overbought,
                rsi_oversold=self.mtf_rsi_oversold,
            )

            # Count buy/sell signals
            buy_count = sum(1 for signal in signals.values() if signal == "buy")
            sell_count = sum(1 for signal in signals.values() if signal == "sell")

            # Log per-timeframe signals
            tf_signals = ", ".join([f"{tf}={signal}" for tf, signal in signals.items()])
            self.logger.info(f"Multi-timeframe signals: {tf_signals}")

            # Determine consensus
            if buy_count >= self.mtf_min_agreement:
                consensus = "buy"
            elif sell_count >= self.mtf_min_agreement:
                consensus = "sell"
            else:
                consensus = "neutral"

            # Log final consensus
            self.logger.info(
                f"Multi-timeframe consensus: {consensus} (buy={buy_count}, sell={sell_count}, min_agreement={self.mtf_min_agreement})"
            )

            return consensus

        except Exception as e:
            self.logger.error(f"Error in multi-timeframe consensus check: {e}")
            return "neutral"

    def _convert_timeframe_to_minutes(self, timeframe: str) -> int | None:
        """Convert timeframe string to minutes."""
        tf_map = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
        return tf_map.get(timeframe)

    def _update_trailing_stops(self) -> None:
        """Update trailing stops for all positions."""
        try:
            if not self.mt5_connected:
                return

            # Update trailing stops using risk manager
            updated_count = self.trade_executor.update_trailing_stops(self.config.SYMBOL)

            if updated_count > 0:
                self.logger.info(f"⛓️ Updated {updated_count} trailing stops")

        except Exception as e:
            self.logger.error(f"Error updating trailing stops: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current system status."""
        try:
            status = {
                "running": self.running,
                "run_id": self.run_id,
                "symbol": self.config.SYMBOL,
                "timeframe": self.config.TIMEFRAME_MIN,
                "mt5_connected": self.mt5_connected,
                "open_positions": _count_open_positions(self.config.SYMBOL, self.config.MAGIC),
                "consecutive_signals": self.consecutive_signals,
                "last_signal": self.last_signal,
                "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
                "reentry_lock_until": (
                    self.reentry_lock_until.isoformat() if self.reentry_lock_until else None
                ),
            }

            # Add performance metrics
            if hasattr(self, 'metrics'):
                status.update(self.metrics.get_stats())

            # Add account info if available
            if self.mt5_connected:
                try:
                    account_info = self.trade_executor.get_account_info()
                    status.update(
                        {
                            "balance": account_info.get('balance'),
                            "equity": account_info.get('equity'),
                            "margin": account_info.get('margin'),
                            "free_margin": account_info.get('free_margin'),
                        }
                    )
                except Exception:
                    pass

            return status

        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
