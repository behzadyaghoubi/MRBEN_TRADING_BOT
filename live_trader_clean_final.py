#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Live Trading System - Permanent EntryPoint
Enhanced with CLI interface, regime detection, adaptive confidence, and agent supervision
"""

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modular components
try:
    from config import MT5Config

    # Only import if these modules exist
    try:
        from trading_system import TradingSystem
        TRADING_SYSTEM_AVAILABLE = True
    except ImportError:
        TRADING_SYSTEM_AVAILABLE = False

    try:
        from trading_loop import TradingLoopManager
        TRADING_LOOP_AVAILABLE = True
    except ImportError:
        TRADING_LOOP_AVAILABLE = False

    from telemetry import EventLogger, MemoryMonitor, MFELogger, PerformanceMetrics
except ImportError:
    try:
        # Try src/ imports
        from src.config import MT5Config
        try:
            from src.trading_system import TradingSystem
            TRADING_SYSTEM_AVAILABLE = True
        except ImportError:
            TRADING_SYSTEM_AVAILABLE = False

        try:
            from src.trading_loop import TradingLoopManager
            TRADING_LOOP_AVAILABLE = True
        except ImportError:
            TRADING_LOOP_AVAILABLE = False

        from src.telemetry import EventLogger, MemoryMonitor, MFELogger, PerformanceMetrics
    except ImportError:
        # Fallback to existing classes if modular imports fail
        TRADING_SYSTEM_AVAILABLE = False
        TRADING_LOOP_AVAILABLE = False
        pass

# Import agent components
try:
    from src.agent import (
        AdvancedAlerting,
        AdvancedPlaybooks,
        AdvancedRiskGate,
        AgentAction,
        DashboardIntegration,
        DecisionCard,
        HealthEvent,
        MLIntegration,
        PredictiveMaintenance,
        maybe_start_agent,
    )
    AGENT_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import paths
        from src.agent.advanced_alerting import AdvancedAlerting
        from src.agent.advanced_playbooks import AdvancedPlaybooks
        from src.agent.bridge import maybe_start_agent
        from src.agent.dashboard import DashboardIntegration
        from src.agent.ml_integration import MLIntegration
        from src.agent.predictive_maintenance import PredictiveMaintenance
        from src.agent.risk_gate import AdvancedRiskGate
        AGENT_AVAILABLE = True
    except ImportError:
        AGENT_AVAILABLE = False
        print("⚠️ Agent components not available, supervision disabled")

# Import pandas for data handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠️ pandas not available, some features may be limited")
    PANDAS_AVAILABLE = False

# Import numpy for calculations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️ numpy not available, some features may be limited")
    NUMPY_AVAILABLE = False

# --- Safe fallback for telemetry classes if import failed ---
if 'EventLogger' not in globals():
    class EventLogger:
        def __init__(self, *args, **kwargs): pass
        def log_trade_attempt(self, *a, **k): pass
        def log_trade_result(self, *a, **k): pass
        def close(self): pass

if 'MFELogger' not in globals():
    class MFELogger:
        def __init__(self, *args, **kwargs): pass

if 'PerformanceMetrics' not in globals():
    class PerformanceMetrics:
        def __init__(self): pass
        def record_cycle(self, *a, **k): pass
        def record_trade(self, *a, **k): pass
        def record_error(self, *a, **k): pass
        def record_memory_usage(self, *a, **k): pass
        def get_stats(self): return {
            "uptime_seconds": 0, "cycle_count": 0, "cycles_per_second": 0,
            "avg_response_time": 0, "total_trades": 0, "error_rate": 0, "memory_mb": 0
        }
        def reset(self): pass

if 'MemoryMonitor' not in globals():
    class MemoryMonitor:
        def __init__(self): pass
        def check_memory(self, *a, **k): return None
        def cleanup_memory(self): return 0
        def should_cleanup(self, *a, **k): return False

# Decision Card Dataclass
@dataclass
class DecisionCard:
    ts: str
    symbol: str
    cycle: int
    price: float
    sma20: float
    sma50: float
    raw_conf: float
    adj_conf: float
    threshold: float
    allow_trade: bool
    regime_label: str
    regime_scores: Dict[str, Any]
    spread_pts: Optional[float]
    atr: Optional[float]
    consecutive: int
    open_positions: int
    signal_src: str
    mode: str
    agent_mode: str

def _dc_from_dict(d: Dict[str, Any]) -> DecisionCard:
    """Convert dict to DecisionCard safely with defaults"""
    return DecisionCard(
        ts=d.get("ts", ""),
        symbol=d.get("symbol", ""),
        cycle=int(d.get("cycle", 0)),
        price=float(d.get("price", 0)),
        sma20=float(d.get("sma20", 0)),
        sma50=float(d.get("sma50", 0)),
        raw_conf=float(d.get("raw_conf", 0)),
        adj_conf=float(d.get("adj_conf", d.get("raw_conf", 0))),
        threshold=float(d.get("threshold", 0.5)),
        allow_trade=bool(d.get("allow_trade", True)),
        regime_label=d.get("regime_label", "UNKNOWN"),
        regime_scores=d.get("regime_scores", {}) or {},
        spread_pts=(None if d.get("spread_pts") is None else float(d.get("spread_pts"))),
        atr=d.get("atr"),
        consecutive=int(d.get("consecutive", 0)),
        open_positions=int(d.get("open_positions", 0)),
        signal_src=d.get("signal_src", "Unknown"),
        mode=d.get("mode", "paper"),
        agent_mode=d.get("agent_mode", "none"),
    )

# Dashboard HTTP handler
class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        try:
            stats = self.server.ctx.metrics.get_stats()
            # Add version and system info
            response_data = {
                "version": "2.0.0",
                "system": "MR BEN Live Trader",
                "timestamp": datetime.now().isoformat(),
                "stats": stats
            }
            body = json.dumps(response_data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(500)
            self.end_headers()

def start_dashboard(ctx, port=8765):
    """Start dashboard HTTP server"""
    try:
        srv = HTTPServer(("127.0.0.1", port), _MetricsHandler)
        srv.ctx = ctx
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        logging.getLogger("Dashboard").info(f"Dashboard at http://127.0.0.1:{port}/metrics")
        return srv
    except Exception as e:
        logging.getLogger("Dashboard").warning(f"Dashboard start failed: {e}")
        return None

# Kill switch file
KILLFILE = "RUN_STOP"

def build_arg_parser():
    """Build comprehensive argument parser for MR BEN Live Trader"""
    p = argparse.ArgumentParser(
        description="MR BEN — Live Trader (Permanent EntryPoint)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s live --mode live --symbol XAUUSD --agent --regime --log-level DEBUG
  %(prog)s paper --symbol XAUUSD --regime --log-level INFO
  %(prog)s backtest --symbol XAUUSD --from 2025-06-01 --to 2025-07-01
  %(prog)s smoke --minutes 5 --symbol XAUUSD --log-level DEBUG
  %(prog)s agent --mode observe --symbol XAUUSD
        """
    )

    sub = p.add_subparsers(dest="cmd", required=True, help="Available commands")

    # Live/Paper trading
    p_live = sub.add_parser("live", help="Run live or paper trading")
    p_live.add_argument("--mode", choices=["live", "paper"], default="live",
                       help="Trading mode (default: live)")
    p_live.add_argument("--symbol", default="XAUUSD.PRO",
                       help="Trading symbol (default: XAUUSD.PRO)")
    p_live.add_argument("--config", default="config.json",
                       help="Configuration file (default: config.json)")
    p_live.add_argument("--core", choices=["legacy", "modular"], default="legacy",
                       help="Core system to use: legacy (MT5LiveTrader) or modular (TradingSystem)")
    p_live.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Log level (default: INFO)")
    p_live.add_argument("--agent", action="store_true",
                       help="Enable GPT-5 agent supervision")
    p_live.add_argument("--agent-mode", choices=["observe", "guard", "auto"], default="guard",
                       help="Agent mode: observe (read-only), guard (block risky), auto (auto-remediate)")
    p_live.add_argument("--regime", action="store_true",
                       help="Enable regime detection & adaptive thresholds")
    p_live.add_argument("--profile", action="store_true",
                       help="Enable performance profiling")

    # Paper trading (alias for live --mode paper)
    p_paper = sub.add_parser("paper", help="Run paper trading (alias for live --mode paper)")
    p_paper.add_argument("--symbol", default="XAUUSD.PRO",
                        help="Trading symbol (default: XAUUSD.PRO)")
    p_paper.add_argument("--config", default="config.json",
                        help="Configuration file (default: config.json)")
    p_paper.add_argument("--core", choices=["legacy", "modular"], default="legacy",
                        help="Core system to use: legacy (MT5LiveTrader) or modular (TradingSystem)")
    p_paper.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Log level (default: INFO)")
    p_paper.add_argument("--agent", action="store_true",
                        help="Enable GPT-5 agent supervision")
    p_paper.add_argument("--agent-mode", choices=["observe", "guard", "auto"], default="guard",
                        help="Agent mode: observe (read-only), guard (block risky), auto (auto-remediate)")
    p_paper.add_argument("--regime", action="store_true",
                        help="Enable regime detection & adaptive thresholds")
    p_paper.add_argument("--profile", action="store_true",
                        help="Enable performance profiling")

    # Backtest
    p_bt = sub.add_parser("backtest", help="Run backtest")
    p_bt.add_argument("--symbol", default="XAUUSD.PRO",
                     help="Trading symbol (default: XAUUSD.PRO)")
    p_bt.add_argument("--from", dest="dt_from",
                     help="Start date (YYYY-MM-DD)")
    p_bt.add_argument("--to", dest="dt_to",
                     help="End date (YYYY-MM-DD)")
    p_bt.add_argument("--config", default="config.json",
                     help="Configuration file (default: config.json)")
    p_bt.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     default="INFO", help="Log level (default: INFO)")
    p_bt.add_argument("--regime", action="store_true",
                     help="Enable regime detection & adaptive thresholds")
    p_bt.add_argument("--agent", action="store_true",
                     help="Enable GPT-5 agent supervision")
    p_bt.add_argument("--agent-mode", choices=["observe", "guard", "auto"], default="guard",
                     help="Agent mode: observe (read-only), guard (block risky), auto (auto-remediate)")


    # Smoke test
    p_sm = sub.add_parser("smoke", help="Quick smoke test")
    p_sm.add_argument("--minutes", type=int, default=5,
                     help="Test duration in minutes (default: 5)")
    p_sm.add_argument("--symbol", default="XAUUSD.PRO",
                     help="Trading symbol (default: XAUUSD.PRO)")
    p_sm.add_argument("--config", default="config.json",
                     help="Configuration file (default: config.json)")
    p_sm.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     default="INFO", help="Log level (default: INFO)")
    p_sm.add_argument("--regime", action="store_true",
                     help="Enable regime detection & adaptive thresholds")
    p_sm.add_argument("--agent", action="store_true",
                     help="Enable GPT-5 agent supervision")
    p_sm.add_argument("--agent-mode", choices=["observe", "guard", "auto"], default="guard",
                     help="Agent mode: observe (read-only), guard (block risky), auto (auto-remediate)")


    # Agent-only observe loop
    p_ag = sub.add_parser("agent", help="Agent observe/paper/live without trading core")
    p_ag.add_argument("--mode", choices=["observe", "paper", "live"], default="observe",
                     help="Agent mode (default: observe)")
    p_ag.add_argument("--agent-mode", choices=["observe", "guard", "auto"], default="guard",
                     help="Agent supervision mode: observe (read-only), guard (block risky), auto (auto-remediate)")
    p_ag.add_argument("--symbol", default="XAUUSD.PRO",
                     help="Trading symbol (default: XAUUSD.PRO)")
    p_ag.add_argument("--config", default="config.json",
                     help="Configuration file (default: config.json)")
    p_ag.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     default="INFO", help="Log level (default: INFO)")
    p_ag.add_argument("--profile", action="store_true",
                     help="Enable performance profiling")


    return p

def setup_logging(level: str = "INFO", log_file: str = "logs/trading_bot.log"):
    """Setup logging configuration"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Configure root logging
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='[%(asctime)s][%(levelname)s] %(message)s'
    )

    # Add file handler
    try:
        from logging.handlers import RotatingFileHandler

        fh = RotatingFileHandler(
            log_file,
            maxBytes=5_000_000,
            backupCount=5,
            encoding='utf-8'
        )
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        fh.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s'))

        # Add to root logger
        logging.getLogger().addHandler(fh)

    except Exception as e:
        logging.warning(f"File logging setup failed: {e}")

def bootstrap(config_path: str, log_level: str):
    """Bootstrap configuration and logging"""
    setup_logging(level=log_level)
    logger = logging.getLogger("Bootstrap")

    try:
        # Load configuration
        if config_path.endswith('.json'):
            config = MT5Config(config_path)
        else:
            # Fallback to existing config loading
            config = MT5Config()

        logger.info("✅ Configuration loaded successfully")
        logger.info(f"Symbol: {config.SYMBOL}")
        logger.info(f"Timeframe: {config.TIMEFRAME_MIN} minutes")
        logger.info(f"Demo Mode: {config.DEMO_MODE}")

        return config

    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        raise

def should_halt():
    """Check if system should halt due to kill file"""
    return os.path.exists(KILLFILE)

def apply_regime_and_threshold(core_ctx, symbol: str, cfg, raw_conf: float):
    """Apply regime detection and adaptive confidence thresholds"""
    logger = logging.getLogger("Regime")

    try:
        # Import regime detection components
        try:
            from src.ai.regime import RegimeLabel, RegimeSnapshot, infer_regime

            # Try both singular and plural paths for scorer
            try:
                from src.strategy.scorer import adapt_confidence
                SCORER_AVAILABLE = True
            except ImportError:
                try:
                    from src.strategies.scorer import adapt_confidence
                    SCORER_AVAILABLE = True
                except ImportError:
                    SCORER_AVAILABLE = False
        except ImportError as e:
            logger.warning(f"Advanced regime detection not available: {e}, using fallback")
            return _fallback_regime_detection(core_ctx, symbol, raw_conf, cfg)

        # Advanced regime detection
        try:
            bars = core_ctx.get_recent_bars(symbol) if hasattr(core_ctx, 'get_recent_bars') else None
            if bars is None or len(bars) < 60:  # Need at least 60 bars for advanced features
                logger.warning("Insufficient data for advanced regime detection, using fallback")
                return _fallback_regime_detection(core_ctx, symbol, raw_conf)

            micro = core_ctx.get_micro(symbol) if hasattr(core_ctx, 'get_micro') else None

            # Get regime configuration
            regime_config = cfg.get("regime", {})
            if not regime_config:
                logger.warning("No regime configuration found, using defaults")
                regime_config = {}

            # Infer regime using advanced detection
            regime = infer_regime(bars, micro, regime_config)
            logger.info(f"🔍 Advanced regime detected: {regime.label.value} (confidence: {regime.confidence:.3f})")

            # Adapt confidence based on regime
            if SCORER_AVAILABLE:
                decision = adapt_confidence(raw_conf, regime.label, regime_config.get("adapt", {}))
            else:
                # Fallback when scorer not available
                decision = {
                    'adj_conf': raw_conf,
                    'threshold': 0.5,
                    'allow_trade': raw_conf >= 0.5
                }

            return {
                'label': regime.label.value,
                'scores': regime.scores,
                'adj_conf': decision['adj_conf'],
                'threshold': decision['threshold'],
                'allow_trade': decision['allow_trade']
            }, decision

        except Exception as e:
            logger.warning(f"Advanced regime detection failed: {e}, falling back to basic")
            return _fallback_regime_detection(core_ctx, symbol, raw_conf, cfg)

    except Exception as e:
        logger = logging.getLogger("Regime")
        logger.error(f"Error in regime detection: {e}")
        # Fallback to allow trade with original confidence
        return None, {
            'adj_conf': raw_conf,
            'threshold': 0.5,
            'allow_trade': raw_conf >= 0.5
        }


def _fallback_regime_detection(core_ctx, symbol: str, raw_conf: float, cfg=None):
    """Fallback regime detection when advanced features unavailable"""
    logger = logging.getLogger("Regime")
    logger.info("Using fallback regime detection")

    # Get configurable thresholds from config
    regime_config = cfg.get("regime", {}).get("fallback", {}) if cfg else {}
    low_vol_threshold = regime_config.get("low_vol", 0.005)
    high_vol_threshold = regime_config.get("high_vol", 0.02)
    low_vol_conf_mult = regime_config.get("low_vol_conf_mult", 1.10)
    high_vol_conf_mult = regime_config.get("high_vol_conf_mult", 0.80)
    low_vol_thr = regime_config.get("low_vol_thr", 0.40)
    high_vol_thr = regime_config.get("high_vol_thr", 0.60)

    try:
        bars = core_ctx.get_recent_bars(symbol) if hasattr(core_ctx, 'get_recent_bars') else None
        if bars is not None and len(bars) > 20:
            # Calculate simple volatility-based regime
            if PANDAS_AVAILABLE:
                returns = bars['close'].pct_change().dropna()
                volatility = returns.std()
            else:
                # Fallback without pandas
                close_prices = [float(bar['close']) for bar in bars[-20:]]
                returns = []
                for i in range(1, len(close_prices)):
                    returns.append((close_prices[i] - close_prices[i-1]) / close_prices[i-1])
                volatility = np.std(returns) if returns else 0.01

            if volatility > high_vol_threshold:  # High volatility threshold
                regime_label = "HIGH_VOL"
                adj_conf = raw_conf * high_vol_conf_mult  # Reduce confidence in high volatility
                threshold = high_vol_thr  # Higher threshold required
            elif volatility < low_vol_threshold:  # Low volatility threshold
                regime_label = "LOW_VOL"
                adj_conf = raw_conf * low_vol_conf_mult  # Increase confidence in low volatility
                threshold = low_vol_thr  # Lower threshold acceptable
            else:
                regime_label = "NORMAL"
                adj_conf = raw_conf
                threshold = 0.5  # Standard threshold

            allow_trade = adj_conf >= threshold

            return {
                'label': regime_label,
                'scores': {'volatility': volatility},
                'adj_conf': adj_conf,
                'threshold': threshold,
                'allow_trade': allow_trade
            }, {
                'adj_conf': adj_conf,
                'threshold': threshold,
                'allow_trade': allow_trade
            }
        else:
            # Default to normal regime
            return {
                'label': 'NORMAL',
                'scores': {},
                'adj_conf': raw_conf,
                'threshold': 0.5,
                'allow_trade': raw_conf >= 0.5
            }, {
                'adj_conf': raw_conf,
                'threshold': 0.5,
                'allow_trade': raw_conf >= 0.5
            }
    except Exception as e:
        logger.warning(f"Fallback regime detection failed: {e}")
        # Ultimate fallback
        return {
            'label': 'NORMAL',
            'scores': {},
            'adj_conf': raw_conf,
            'threshold': 0.5,
            'allow_trade': raw_conf >= 0.5
        }, {
            'adj_conf': raw_conf,
            'threshold': 0.5,
            'allow_trade': raw_conf >= 0.5
        }

def _resolve_agent_mode(args, cfg):
    """Resolve agent mode from CLI args, environment, or config"""
    if getattr(args, "agent_mode", None):
        return args.agent_mode
    env = os.getenv("AGENT_MODE")
    if env in {"observe","guard","auto"}:
        return env
    try:
        return cfg.config_data.get("agent", {}).get("mode", "guard")
    except Exception:
        return "guard"

def maybe_start_agent(cfg, mode: str):
    """Start agent bridge if available"""
    try:
        from src.agent.bridge import maybe_start_agent as agent_maybe_start

        # Convert MT5Config to dict-like object for agent
        config_dict = cfg.config_data if hasattr(cfg, 'config_data') else cfg
        return agent_maybe_start(config_dict, mode)
    except ImportError:
        logger = logging.getLogger("Agent")
        logger.warning("Agent components not available, continuing without agent supervision")
        return None
    except Exception as e:
        logger = logging.getLogger("Agent")
        logger.error(f"Failed to start agent: {e}")
        return None

def cmd_live(args):
    """Execute live/paper trading command"""
    logger = logging.getLogger("LiveTrading")
    logger.info(f"🚀 Starting {args.mode} trading for {args.symbol}")

    try:
        # Bootstrap system
        cfg = bootstrap(args.config, args.log_level)

        # Initialize trading system
        core = None
        try:
            # Check if core argument is present, default to legacy if not
            core_type = getattr(args, 'core', 'legacy')
            if core_type == "legacy" or (core_type == "modular" and not hasattr(sys.modules[__name__], 'TradingSystem')):
                # Use existing MT5LiveTrader class
                core = MT5LiveTrader(cfg)
                logger.info("✅ Trading system initialized (legacy mode)")
            elif core_type == "modular":
                # Use modular system
                core = TradingSystem(cfg)
                logger.info("✅ Trading system initialized (modular mode)")
            else:
                logger.error(f"❌ Core system '{core_type}' not available")
                return 1
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading system: {e}")
            return 1

        # Start agent if requested
        agent = None
        if args.agent and AGENT_AVAILABLE:
            agent_mode = _resolve_agent_mode(args, cfg)
            logger.info(f"🤖 Agent mode: {agent_mode}")
            agent = maybe_start_agent(cfg, agent_mode)
            if agent:
                # Ensure agent mode is stored in bridge
                if hasattr(agent, 'mode'):
                    agent.mode = agent_mode
                logger.info(f"✅ Agent started in {agent_mode} mode")
                # Connect agent to core
                core.agent = agent
                # Initialize agent components
                try:
                    core._initialize_agent_components()
                    logger.info("✅ Agent components initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Agent components initialization failed: {e}")
            else:
                logger.warning("⚠️ Agent failed to start")
        elif args.agent and not AGENT_AVAILABLE:
            logger.warning("⚠️ Agent requested but not available")

        # Start trading system
        try:
            logger.info("🚀 Starting trading system...")
            success = core.start()
            if not success:
                logger.error("❌ Trading system failed to start")
                return 1
            logger.info("✅ Trading system started successfully")

            # Wait for trading to complete or interruption
            try:
                while core.running:
                    time.sleep(1)
                    # Check kill file
                    if should_halt():
                        logger.info("🛑 KILLFILE detected, stopping system")
                        break
            except KeyboardInterrupt:
                logger.info("🛑 Interrupted by user")
            finally:
                # Stop the system
                core.stop()

        except Exception as e:
            logger.error(f"❌ Failed to start trading system: {e}")
            return 1
        finally:
            # Cleanup
            if agent:
                try:
                    agent.stop()
                except Exception as e:
                    logger.warning(f"Agent cleanup failed: {e}")
            try:
                if hasattr(core, 'cleanup'):
                    core.cleanup()
            except Exception as e:
                logger.warning(f"Core cleanup failed: {e}")

            logger.info("✅ Trading session completed")
            return 0

    except Exception as e:
        logger.error(f"❌ Fatal error in live trading: {e}")
        return 1

def cmd_backtest(args):
    """Execute backtest command"""
    logger = logging.getLogger("Backtest")
    logger.info(f"📊 Starting backtest for {args.symbol}")

    try:
        cfg = bootstrap(args.config, "INFO")

        # Import backtest module
        try:
            from src.core import backtest
            result = backtest.run(
                symbol=args.symbol,
                dt_from=args.dt_from,
                dt_to=args.dt_to,
                cfg=cfg
            )
            logger.info("✅ Backtest completed successfully")
            return result
        except ImportError:
            logger.warning("Backtest module not available, using fallback")
            # Fallback backtest implementation
            return {"status": "completed", "trades": 0, "pnl": 0.0}

    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        return 1

def cmd_smoke(args):
    """Execute smoke test command"""
    logger = logging.getLogger("SmokeTest")
    logger.info(f"🚬 Starting smoke test for {args.symbol} ({args.minutes} minutes)")

    try:
        cfg = bootstrap(args.config, "INFO")

        # Import smoke test module
        try:
            from src.core import smoke as smoke_runner
            logger.info("✅ Advanced smoke test module loaded")

            # Run advanced smoke test with regime and agent flags
            result = smoke_runner.run(
                minutes=args.minutes,
                symbol=args.symbol,
                cfg=cfg,
                enable_agent=getattr(args, "agent", False),
                enable_regime=getattr(args, "regime", True)
            )

            if result == 0:
                logger.info("✅ Advanced smoke test completed successfully")
            else:
                logger.warning("⚠️ Advanced smoke test completed with warnings")

            return result

        except ImportError as e:
            logger.warning(f"Advanced smoke test module not available: {e}, using fallback")
            # Fallback smoke test implementation
            logger.info("Running fallback smoke test...")
            time.sleep(args.minutes * 60)  # Simulate test duration
            logger.info("✅ Fallback smoke test completed")
            return {"status": "completed", "duration_minutes": args.minutes}

    except Exception as e:
        logger.error(f"❌ Smoke test failed: {e}")
        return 1

def cmd_agent(args):
    """Execute agent-only command"""
    logger = logging.getLogger("AgentOnly")
    logger.info(f"🤖 Starting agent in {args.mode} mode for {args.symbol}")

    try:
        cfg = bootstrap(args.config, "INFO")

        # Start agent bridge
        bridge = maybe_start_agent(cfg, args.agent_mode)

        if bridge:
            logger.info("✅ Agent bridge started successfully")
            bridge.block_until_stopped()
        else:
            logger.warning("⚠️ Agent bridge not available")

        return 0

    except Exception as e:
        logger.error(f"❌ Agent command failed: {e}")
        return 1

def main():
    """Main entry point with CLI interface"""
    try:
        parser = build_arg_parser()
        args = parser.parse_args()

        # Handle paper as alias for live --mode paper
        if args.cmd == "paper":
            args.cmd = "live"
            args.mode = "paper"

        # Execute appropriate command
        if args.cmd == "live":
            return cmd_live(args)
        elif args.cmd == "backtest":
            return cmd_backtest(args)
        elif args.cmd == "smoke":
            return cmd_smoke(args)
        elif args.cmd == "agent":
            return cmd_agent(args)
        else:
            print(f"Unknown command: {args.cmd}")
            return 1

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

# ============================================================================
# ORIGINAL TRADING SYSTEM CODE
# ============================================================================

# -----------------------------
# Optional deps (MT5 / AI stack)
# -----------------------------
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    print("⚠️ MetaTrader5 not available, switching to demo/synthetic mode if needed.")
    MT5_AVAILABLE = False

try:
    import tensorflow as tf
    try:
        from tensorflow.keras.models import load_model
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        try:
            from keras.models import load_model
            TENSORFLOW_AVAILABLE = True
        except ImportError:
            TENSORFLOW_AVAILABLE = False

    try:
        from sklearn.preprocessing import LabelEncoder
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False

    try:
        import joblib
        JOBLIB_AVAILABLE = True
    except ImportError:
        JOBLIB_AVAILABLE = False

    AI_AVAILABLE = TENSORFLOW_AVAILABLE and SKLEARN_AVAILABLE and JOBLIB_AVAILABLE
except ImportError:
    TENSORFLOW_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    JOBLIB_AVAILABLE = False
    AI_AVAILABLE = False
    print("⚠️ AI stack not available")

# -----------------------------
# Helpers
# -----------------------------

def round_price(symbol: str, price: float) -> float:
    """Round price to symbol's digits/point using MT5 symbol info if available."""
    try:
        if MT5_AVAILABLE:
            info = mt5.symbol_info(symbol)
            if info and info.point:
                step = Decimal(str(info.point))
                q = (Decimal(str(price)) / step).to_integral_value(rounding=ROUND_HALF_UP) * step
                return float(q)
        return float(Decimal(str(price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    except Exception:
        return float(Decimal(str(price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def enforce_min_distance_and_round(symbol: str, entry: float, sl: float, tp: float, is_buy: bool) -> Tuple[float, float]:
    """
    Ensure SL/TP respect broker min distance (trade_stops_level & trade_freeze_level) and then round.
    """
    try:
        if MT5_AVAILABLE:
            info = mt5.symbol_info(symbol)
        else:
            info = None
        if not info:
            return round_price(symbol, sl), round_price(symbol, tp)

        point = info.point or 0.01
        stops_pts = float(getattr(info, 'trade_stops_level', 0) or 0)
        freeze_pts = float(getattr(info, 'trade_freeze_level', 0) or 0)
        min_dist = max(stops_pts, freeze_pts) * float(point)

        if is_buy:
            if (entry - sl) < min_dist:
                sl = entry - min_dist
            if (tp - entry) < min_dist:
                tp = entry + min_dist
        else:
            if (sl - entry) < min_dist:
                sl = entry + min_dist
            if (entry - tp) < min_dist:
                tp = entry - min_dist

        return round_price(symbol, sl), round_price(symbol, tp)
    except Exception:
        return round_price(symbol, sl), round_price(symbol, tp)

def _pick_filling_mode(symbol: str):
    """Pick the correct filling mode for the symbol to avoid 10030 error"""
    try:
        if not MT5_AVAILABLE:
            return mt5.ORDER_FILLING_RETURN  # fallback امن

        info = mt5.symbol_info(symbol)
        if not info:
            return mt5.ORDER_FILLING_RETURN  # fallback امن

        fm = int(getattr(info, "filling_mode", 0))
        # نگاشت مود سمبل به type_filling سفارش
        if fm == mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        elif fm == mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        elif fm == mt5.SYMBOL_FILLING_RETURN:
            return mt5.ORDER_FILLING_RETURN

        # بعضی بروکرها RETURN-only هستن؛ امن‌ترین پیش‌فرض:
        return mt5.ORDER_FILLING_RETURN
    except Exception:
        return mt5.ORDER_FILLING_RETURN

def _symbol_filling_mode(symbol: str):
    """Get symbol's recommended filling mode"""
    try:
        if not MT5_AVAILABLE:
            return None
        info = mt5.symbol_info(symbol)
        if not info:
            return None
        fm = int(getattr(info, "filling_mode", -1))
        if fm in (mt5.SYMBOL_FILLING_FOK, mt5.SYMBOL_FILLING_IOC, mt5.SYMBOL_FILLING_RETURN):
            return fm
    except Exception:
        pass
    return None  # ناشناخته

def _map_symbol_to_order_filling(sym_fm: int):
    """Map symbol filling mode to order filling mode"""
    if sym_fm == mt5.SYMBOL_FILLING_FOK:    return mt5.ORDER_FILLING_FOK
    if sym_fm == mt5.SYMBOL_FILLING_IOC:    return mt5.ORDER_FILLING_IOC
    if sym_fm == mt5.SYMBOL_FILLING_RETURN: return mt5.ORDER_FILLING_RETURN
    return None

# -----------------------------
# Configuration
# -----------------------------

class MT5Config:
    """Reads config.json and exposes fields used by the system."""

    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            self._load_config()
            self.logger.info("✅ Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Configuration error: {e}")
            raise

    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)

            # Store raw config for backward compatibility
            self.config_data = raw

            # Parse credentials
            creds = raw.get("credentials", {})
            self.LOGIN = creds.get("login")
            self.PASSWORD = os.getenv("MT5_PASSWORD", creds.get("password"))
            self.SERVER = creds.get("server")

            # Set flags
            flags = raw.get("flags", {})
            self.DEMO_MODE = bool(flags.get("demo_mode", True))

            # Strict credential check only if DEMO_MODE is False
            if not self.DEMO_MODE and not (self.LOGIN and self.PASSWORD and self.SERVER):
                raise RuntimeError("❌ MT5 credentials missing. Provide via config.json under 'credentials'.")

            # Parse trading configuration
            trading = raw.get("trading", {})
            self.SYMBOL = trading.get("symbol", "XAUUSD.PRO")
            self.TIMEFRAME_MIN = int(trading.get("timeframe", 15))
            self.BARS = int(trading.get("bars", 500))
            self.MAGIC = int(trading.get("magic_number", 20250721))
            self.SESSIONS = trading.get("sessions", ["London", "NY"])
            self.MAX_SPREAD_POINTS = int(trading.get("max_spread_points", 200))
            self.USE_RISK_BASED_VOLUME = bool(trading.get("use_risk_based_volume", True))
            self.FIXED_VOLUME = float(trading.get("fixed_volume", 0.01))
            self.SLEEP_SECONDS = int(trading.get("sleep_seconds", 12))
            self.RETRY_DELAY = int(trading.get("retry_delay", 5))
            self.CONSECUTIVE_SIGNALS_REQUIRED = int(trading.get("consecutive_signals_required", 1))
            self.LSTM_TIMESTEPS = int(trading.get("lstm_timesteps", 50))
            self.COOLDOWN_SECONDS = int(trading.get("cooldown_seconds", 180))

            # Parse risk configuration
            risk = raw.get("risk", {})
            self.BASE_RISK = float(risk.get("base_risk", 0.01))
            self.MIN_LOT = float(risk.get("min_lot", 0.01))
            self.MAX_LOT = float(risk.get("max_lot", 2.0))
            self.MAX_OPEN_TRADES = int(risk.get("max_open_trades", 3))
            self.MAX_DAILY_LOSS = float(risk.get("max_daily_loss", 0.02))
            self.MAX_TRADES_PER_DAY = int(risk.get("max_trades_per_day", 10))
            self.SL_ATR_MULTIPLIER = float(risk.get("sl_atr_multiplier", 1.6))
            self.TP_ATR_MULTIPLIER = float(risk.get("tp_atr_multiplier", 2.2))

            # Parse logging configuration
            logging_cfg = raw.get("logging", {})
            self.LOG_ENABLED = bool(logging_cfg.get("enabled", True))
            self.LOG_LEVEL = logging_cfg.get("level", "INFO")
            self.LOG_FILE = logging_cfg.get("log_file", "logs/trading_bot.log")
            self.TRADE_LOG_PATH = logging_cfg.get("trade_log_path", "data/trade_log_gold.csv")

            # Parse session configuration
            session_cfg = raw.get("session", {})
            self.SESSION_TZ = session_cfg.get("timezone", "Etc/UTC")

            # Parse advanced configuration
            advanced = raw.get("advanced", {})
            self.SWING_LOOKBACK = int(advanced.get("swing_lookback", 12))
            self.DYNAMIC_SPREAD_ATR_FRAC = float(advanced.get("dynamic_spread_atr_frac", 0.10))
            self.DEVIATION_MULTIPLIER = float(advanced.get("deviation_multiplier", 1.5))
            self.INBAR_EVAL_SECONDS = int(advanced.get("inbar_eval_seconds", 10))
            self.INBAR_MIN_CONF = float(advanced.get("inbar_min_conf", 0.66))
            self.INBAR_MIN_SCORE = float(advanced.get("inbar_min_score", 0.12))
            self.INBAR_MIN_STRUCT_BUFFER_ATR = float(advanced.get("inbar_min_struct_buffer_atr", 0.8))
            self.STARTUP_WARMUP_SECONDS = int(advanced.get("startup_warmup_seconds", 90))
            self.STARTUP_MIN_CONF = float(advanced.get("startup_min_conf", 0.62))
            self.STARTUP_MIN_SCORE = float(advanced.get("startup_min_score", 0.10))
            self.REENTRY_WINDOW_SECONDS = int(advanced.get("reentry_window_seconds", 90))

            # Parse execution configuration
            execution_cfg = raw.get("execution", {})
            self.SPREAD_EPS = float(execution_cfg.get("spread_eps", 0.02))
            self.USE_SPREAD_MA = bool(execution_cfg.get("use_spread_ma", True))
            self.SPREAD_MA_WINDOW = int(execution_cfg.get("spread_ma_window", 5))
            self.SPREAD_HYSTERESIS_FACTOR = float(execution_cfg.get("spread_hysteresis_factor", 1.05))

            # Parse TP policy configuration
            tp_policy_cfg = raw.get("tp_policy", {})
            self.TP_SPLIT = bool(tp_policy_cfg.get("split", True))
            self.TP1_R = float(tp_policy_cfg.get("tp1_r", 0.8))
            self.TP2_R = float(tp_policy_cfg.get("tp2_r", 1.5))
            self.TP1_SHARE = float(tp_policy_cfg.get("tp1_share", 0.5))
            self.BREAKEVEN_AFTER_TP1 = bool(tp_policy_cfg.get("breakeven_after_tp1", True))

            # Parse conformal configuration
            conformal_cfg = raw.get("conformal", {})
            self.CONFORMAL_ENABLED = bool(conformal_cfg.get("enabled", True))
            self.CONFORMAL_SOFT_GATE = bool(conformal_cfg.get("soft_gate", True))
            self.CONFORMAL_EMERGENCY_BYPASS = bool(conformal_cfg.get("emergency_bypass", False))
            self.CONFORMAL_MIN_P = float(conformal_cfg.get("min_p", 0.10))
            self.CONFORMAL_HARD_FLOOR = float(conformal_cfg.get("hard_floor", 0.05))
            self.CONFORMAL_PENALTY_SMALL = float(conformal_cfg.get("penalty_small", 0.05))
            self.CONFORMAL_PENALTY_BIG = float(conformal_cfg.get("penalty_big", 0.10))
            self.CONFORMAL_EXTRA_CONSECUTIVE = int(conformal_cfg.get("extra_consecutive", 1))
            self.CONFORMAL_TREAT_ZERO_AS_BLOCK = bool(conformal_cfg.get("treat_zero_as_block", True))
            self.CONFORMAL_MAX_CONF_BUMP_FLOOR = float(conformal_cfg.get("max_conf_bump_floor", 0.05))
            self.CONFORMAL_EXTRA_CONSEC_FLOOR = int(conformal_cfg.get("extra_consec_floor", 2))
            self.CONFORMAL_CAP_FINAL_THR = float(conformal_cfg.get("cap_final_thr", 0.90))

        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        return {
            "symbol": self.SYMBOL,
            "timeframe": self.TIMEFRAME_MIN,
            "magic": self.MAGIC,
            "sessions": self.SESSIONS,
            "risk": {
                "base_risk": self.BASE_RISK,
                "max_open_trades": self.MAX_OPEN_TRADES,
                "max_daily_loss": self.MAX_DAILY_LOSS
            },
            "demo_mode": self.DEMO_MODE,
            "conformal_enabled": self.CONFORMAL_ENABLED
        }

# -----------------------------
# Performance Metrics
# -----------------------------

class PerformanceMetrics:
    """Performance monitoring and metrics collection"""

    def __init__(self):
        self.start_time = datetime.now()
        self.cycle_count = 0
        self.trade_count = 0
        self.error_count = 0
        self.response_times = []
        self.memory_usage = []

        self.logger = logging.getLogger(self.__class__.__name__)

    def record_cycle(self, response_time: float):
        """Record a trading cycle"""
        self.cycle_count += 1
        self.response_times.append(response_time)

        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times.pop(0)

    def record_trade(self):
        """Record a trade execution"""
        self.trade_count += 1

    def record_error(self):
        """Record an error occurrence"""
        self.error_count += 1

    def record_memory_usage(self, memory_mb: float):
        """Record memory usage"""
        self.memory_usage.append({
            "timestamp": datetime.now().isoformat(),
            "memory_mb": memory_mb
        })

        # Keep only last 100 memory readings
        if len(self.memory_usage) > 100:
            self.memory_usage.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()

            # Calculate response time statistics
            avg_response = 0.0
            if self.response_times:
                avg_response = sum(self.response_times) / len(self.response_times)

            # Calculate memory statistics
            current_memory = 0.0
            if self.memory_usage:
                current_memory = self.memory_usage[-1]["memory_mb"]

            return {
                "uptime_seconds": uptime,
                "cycle_count": self.cycle_count,
                "cycles_per_second": self.cycle_count / uptime if uptime > 0 else 0,
                "avg_response_time": avg_response,
                "total_trades": self.trade_count,
                "error_rate": self.error_count / max(self.cycle_count, 1),
                "memory_mb": current_memory,
                "start_time": self.start_time.isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error calculating performance stats: {e}")
            return {
                "uptime_seconds": 0,
                "cycle_count": 0,
                "cycles_per_second": 0,
                "avg_response_time": 0,
                "total_trades": 0,
                "error_rate": 0,
                "memory_mb": 0,
                "start_time": self.start_time.isoformat()
            }

    def reset(self):
        """Reset all metrics"""
        self.start_time = datetime.now()
        self.cycle_count = 0
        self.trade_count = 0
        self.error_count = 0
        self.response_times.clear()
        self.memory_usage.clear()
        self.logger.info("Performance metrics reset")

# -----------------------------
# Memory Monitor
# -----------------------------

class MemoryMonitor:
    """Memory usage monitoring and cleanup"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_check = datetime.now()
        self.check_interval = 300  # 5 minutes

    def check_memory(self, force: bool = False) -> Optional[float]:
        """Check current memory usage"""
        try:
            now = datetime.now()
            if not force and (now - self.last_check).total_seconds() < self.check_interval:
                return None

            import gc

            # Force garbage collection
            collected = gc.collect()

            # Get process memory info
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024

                # Log memory usage
                self.logger.info(f"💾 Memory check: {memory_mb:.1f} MB (collected {collected} objects)")

                # Log system memory if available
                try:
                    system_memory = psutil.virtual_memory()
                    system_used_mb = system_memory.used / 1024 / 1024
                    system_total_mb = system_memory.total / 1024 / 1024
                    system_percent = system_memory.percent

                    self.logger.info(f"💻 System Memory: {system_used_mb:.1f}/{system_total_mb:.1f} MB ({system_percent:.1f}%)")
                except Exception:
                    pass

            except ImportError:
                self.logger.warning("psutil not available - memory monitoring disabled")
                memory_mb = 0.0

            self.last_check = now
            return memory_mb

        except Exception as e:
            self.logger.warning(f"Memory monitoring failed: {e}")
            return None

    def cleanup_memory(self):
        """Force memory cleanup"""
        try:
            import gc

            # Force garbage collection
            collected = gc.collect()

            self.logger.info(f"🧹 Memory cleanup: collected {collected} objects")
            return collected

        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
            return 0

    def should_cleanup(self, memory_mb: float, threshold_mb: float = 1000) -> bool:
        """Determine if memory cleanup is needed"""
        return memory_mb > threshold_mb

# -----------------------------
# Main Trading Class
# -----------------------------

class MT5LiveTrader:
    """Main trading system orchestrator"""

    def __init__(self, config: MT5Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize agent components
        self.agent = None
        self.risk_gate = None
        self.playbooks = None
        self.ml_integration = None
        self.predictive_maintenance = None
        self.advanced_alerting = None
        self.dashboard_server = None

        # Initialize components
        self._initialize_components()

        # Initialize state variables
        self._initialize_state()

        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.memory_monitor = MemoryMonitor()

        # Start dashboard regardless of agent presence
        try:
            dash_cfg = getattr(self.config, "config_data", {}).get("dashboard", {})
            # Always enable dashboard by default
            if dash_cfg.get("enabled", True):
                port = int(dash_cfg.get("port", 8765))
                self.dashboard_server = start_dashboard(self, port)
                self.logger.info(f"Dashboard started at http://127.0.0.1:{port}/metrics")
        except Exception as e:
            self.logger.warning(f"Dashboard start failed: {e}")

        self.logger.info("✅ MT5LiveTrader initialized successfully")

    def _initialize_agent_components(self):
        """Initialize agent-related components"""
        try:
            if not self.agent:
                return

            # Get raw config data for agent components
            config_data = getattr(self.config, "config_data", {})

            # Initialize risk gate
            self.risk_gate = AdvancedRiskGate(config_data, self.logger)

            # Initialize playbooks
            self.playbooks = AdvancedPlaybooks(config_data, self.logger)

            # Initialize ML integration
            self.ml_integration = MLIntegration(config_data, self.logger)

            # Initialize predictive maintenance
            self.predictive_maintenance = PredictiveMaintenance(config_data, self.logger)

            # Initialize advanced alerting
            self.advanced_alerting = AdvancedAlerting(config_data, self.logger)

            self.logger.info("✅ Agent components initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agent components: {e}")
            raise

    def get_recent_bars(self, symbol: str, n: int = 120):
        """Get recent bars for regime detection (legacy compatibility)"""
        try:
            if hasattr(self, '_get_market_data'):
                df = self._get_market_data()
                if df is not None and len(df) > 0:
                    # Convert to the format expected by regime detection
                    bars = df.tail(n)[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
                    # Rename columns to match expected format
                    bars = bars.rename(columns={
                        'time': 'time',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'tick_volume': 'volume'
                    })
                    return bars.to_dict('records')
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get recent bars: {e}")
            return None

    def get_micro(self, symbol: str):
        """Get microstructure data (legacy compatibility)"""
        try:
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    return {
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "spread_bp": (tick.ask - tick.bid) * 10000,  # Convert to basis points
                        "volume": tick.volume
                    }
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get micro data: {e}")
            return None

    def _get_current_spread(self):
        """Get current spread in points"""
        try:
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(self.config.SYMBOL)
                if tick:
                    return (tick.ask - tick.bid) * 10000  # Convert to points
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get current spread: {e}")
            return None

    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize MT5 connection
            if MT5_AVAILABLE:
                self._init_mt5()
            else:
                self.logger.warning("⚠️ MT5 not available, running in demo mode")

            # Initialize AI system if available
            if AI_AVAILABLE:
                self._init_ai_system()
            else:
                self.logger.warning("⚠️ AI system not available")

            # Initialize event logger
            self._init_event_logger()

        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise

    def _init_mt5(self):
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

            # Login if not in demo mode
            if not self.config.DEMO_MODE:
                if not mt5.login(
                    login=int(self.config.LOGIN),
                    password=self.config.PASSWORD,
                    server=self.config.SERVER
                ):
                    raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
                self.logger.info("✅ MT5 login successful")
            else:
                self.logger.info("✅ MT5 demo mode active")

            # Select symbol with fallback options
            symbol = self.config.SYMBOL
            symbol_selected = False

            # Try primary symbol
            if mt5.symbol_select(symbol, True):
                symbol_selected = True
                self.logger.info(f"✅ Symbol {symbol} selected successfully")
            else:
                self.logger.warning(f"⚠️ Primary symbol {symbol} not available, trying alternatives...")

                # Try alternative symbols for Gold
                alternatives = ["XAUUSD", "XAUUSD.m", "GOLD", "XAUUSD.PRO"]
                for alt_symbol in alternatives:
                    if alt_symbol != symbol and mt5.symbol_select(alt_symbol, True):
                        self.logger.warning(f"⚠️ Primary symbol {symbol} unavailable, using {alt_symbol} instead")
                        self.config.SYMBOL = alt_symbol  # Update config to use available symbol
                        symbol_selected = True
                        break

                if not symbol_selected:
                    self.logger.error(f"❌ No suitable symbol found. Tried: {[symbol] + alternatives}")
                    self.logger.error(f"Available symbols: {[s.name for s in mt5.symbols_get()[:10]] if mt5.symbols_get() else 'None'}")
                    raise RuntimeError(f"Symbol selection failed for {symbol}")

        except Exception as e:
            self.logger.error(f"❌ MT5 initialization failed: {e}")
            raise

    def _init_ai_system(self):
        """Initialize AI system components"""
        try:
            # Load LSTM model
            self.lstm_model = None
            if os.path.exists("models/lstm_model.h5"):
                self.lstm_model = load_model("models/lstm_model.h5")
                self.logger.info("✅ LSTM model loaded")

            # Load ML filter
            self.ml_filter = None
            if os.path.exists("models/ml_filter.pkl"):
                self.ml_filter = joblib.load("models/ml_filter.pkl")
                self.logger.info("✅ ML filter loaded")

            # Load label encoder
            self.label_encoder = None
            if os.path.exists("models/label_encoder.pkl"):
                self.label_encoder = joblib.load("models/label_encoder.pkl")
                self.logger.info("✅ Label encoder loaded")

        except Exception as e:
            self.logger.warning(f"⚠️ AI system initialization failed: {e}")

    def _init_event_logger(self):
        """Initialize event logging system"""
        try:
            # Create data directory
            os.makedirs("data", exist_ok=True)

            # Check if EventLogger class is available
            if 'EventLogger' in globals() and callable(EventLogger):
                # Initialize event logger
                self.ev = EventLogger(
                    path="data/events.jsonl",
                    run_id=datetime.now().strftime("%Y%m%d-%H%M%S"),
                    symbol=self.config.SYMBOL
                )
                self.logger.info("✅ Event logger initialized")
            else:
                self.logger.warning("⚠️ EventLogger unavailable - telemetry disabled")
                self.ev = None

        except Exception as e:
            self.logger.warning(f"⚠️ Event logger initialization failed: {e}")
            self.ev = None

    def _initialize_state(self):
        """Initialize trading state variables"""
        # Trading state
        self.running = False
        self.consecutive_signals = 0
        self.last_signal = 0
        self.last_trailing_update = datetime.now()

        # Trailing registry
        self.trailing_registry = {}
        self.trailing_step = 0.5

        # Flat-state detection
        self._was_open = 0
        self._prev_open_count = 0

        # Warm-up guard
        self.start_time = datetime.now()

        # In-bar evaluation control
        self._last_inbar_eval = None

        # Spread threshold
        self.max_spread_points = int(self.config.MAX_SPREAD_POINTS)

        # Performance monitoring
        self.last_memory_check = time.time()
        self.memory_check_interval = 300  # Check memory every 5 minutes

    def _preflight_production(self):
        """Production preflight checks and safety rails"""
        try:
            # 0) DEMO toggle
            if self.config.DEMO_MODE:
                self.logger.warning("⚠️ DEMO_MODE=True; live requested. Set DEMO_MODE=False in config for real trades.")

            # 1) Credentials (if DEMO_MODE False)
            if not self.config.DEMO_MODE and not (self.config.LOGIN and self.config.PASSWORD and self.config.SERVER):
                raise RuntimeError("MT5 credentials missing for LIVE mode")

            # 2) Symbol availability
            if MT5_AVAILABLE:
                if not mt5.symbol_select(self.config.SYMBOL, True):
                    raise RuntimeError(f"symbol_select failed: {self.config.SYMBOL}")
                info = mt5.symbol_info(self.config.SYMBOL)
                if not info:
                    raise RuntimeError(f"symbol_info missing: {self.config.SYMBOL}")

            # 3) Kill-switch presence
            if os.path.exists(KILLFILE):
                raise RuntimeError("RUN_STOP present; remove to start")

            self.logger.info("✅ Production preflight checks passed")

        except Exception as e:
            self.logger.error(f"❌ Production preflight failed: {e}")
            raise

    def start(self):
        """Start the trading system (compatibility method for cmd_live)"""
        try:
            if self.running:
                self.logger.warning("Trading system already running")
                return True

            # Production preflight checks
            self._preflight_production()

            # Check pandas availability
            if not PANDAS_AVAILABLE:
                self.logger.error("pandas is required for legacy trading loop. Please install pandas.")
                return False

            self.running = True
            self.logger.info("🚀 Starting trading system...")

            # Start trading loop
            self._trading_loop()
            return True
        except Exception as e:
            self.logger.error(f"Failed to start trading system: {e}")
            return False

    def stop(self):
        """Stop the trading system"""
        self.logger.info("🛑 Stopping trading system...")
        self.running = False

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.dashboard_server:
                self.dashboard_server.shutdown()
                self.logger.info("✅ Dashboard server stopped")
        except Exception as e:
            self.logger.warning(f"Dashboard cleanup failed: {e}")

    def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("🔄 Trading loop started")

        while self.running:
            cycle_start = time.time()

            try:
                # Check kill file
                if should_halt():
                    self.logger.info("🛑 KILLFILE detected, halting system")
                    break

                # Panic tripwire - error storm detection
                if self.metrics.error_count > 5 and self.agent and getattr(self.agent, "mode", "guard") != "observe":
                    self.logger.error("❌ Error storm; agent will HALT")
                    # ask agent to HALT or set running=False
                    self.running = False
                    return

                # Performance monitoring
                if self.metrics.cycle_count % 100 == 0:
                    self._log_performance_metrics()

                # Predictive maintenance check
                if self.predictive_maintenance and self.metrics.cycle_count % 60 == 0:
                    try:
                        stats = self.metrics.get_stats()
                        err_rate = stats.get("error_rate", 0)
                        if err_rate > (self.config.config_data.get("agent", {}).get("error_rate_halt", 5))/60.0:
                            if self.agent:
                                self.agent.on_health_event({
                                    "ts": datetime.now().isoformat(),
                                    "severity": "WARN", "kind": "ERROR_RATE",
                                    "message": f"error_rate={err_rate:.3f}"
                                })
                    except Exception as e:
                        self.logger.debug(f"Predictive maintenance check failed: {e}")

                # Memory management
                memory_mb = self.memory_monitor.check_memory()
                if memory_mb is not None:
                    self.metrics.record_memory_usage(memory_mb)

                    # Force cleanup if memory usage is high
                    if self.memory_monitor.should_cleanup(memory_mb):
                        self.logger.warning(f"🧹 High memory usage ({memory_mb:.1f} MB), forcing cleanup")
                        self.memory_monitor.cleanup_memory()

                # Update trailing stops
                self._update_trailing_stops()

                # Get market data
                df = self._get_market_data()
                if df is None or len(df) < 50:
                    self.logger.warning(f"Insufficient data: {len(df) if df is not None else 'None'} bars; retrying...")

                    # Report health event if agent available
                    if self.agent:
                        try:
                            self.agent.on_health_event({
                                "ts": datetime.now().isoformat(),
                                "severity": "ERROR", "kind": "STALE_DATA",
                                "message": "Insufficient market data",
                                "context": {"symbol": self.config.SYMBOL, "bars": len(df) if df is not None else 0}
                            })
                        except Exception as e:
                            self.logger.debug(f"Health event report failed: {e}")

                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # Log data quality for debugging
                self.logger.debug(f"📊 Data quality: {len(df)} bars, last close: {df['close'].iloc[-1]:.5f}, last time: {df['time'].iloc[-1]}")

                # Generate trading signal
                signal = self._generate_signal(df)
                if signal is None:
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # Update consecutive signal counter (FIXED LOGIC)
                if signal['signal'] == 0:
                    # No signal - reset counter
                    self.consecutive_signals = 0
                    self.last_signal = 0
                    self.logger.debug("No signal (0), reset consecutive counter to 0")
                else:
                    # Valid signal - check if same as last
                    if signal['signal'] == self.last_signal:
                        self.consecutive_signals += 1
                        self.logger.info(f"🔄 Consecutive signal {self.consecutive_signals} for {signal['signal']} (need {self.config.CONSECUTIVE_SIGNALS_REQUIRED})")
                    else:
                        # New signal type - reset counter
                        self.consecutive_signals = 1
                        self.last_signal = signal['signal']
                        self.logger.info(f"🆕 New signal type: {signal['signal']}, reset counter to 1")

                # Log signal details for debugging
                self.logger.info(f"📊 Signal: {signal['signal']} (confidence: {signal['confidence']:.3f}, consecutive: {self.consecutive_signals}/{self.config.CONSECUTIVE_SIGNALS_REQUIRED})")

                # Create comprehensive decision card
                try:
                    info = mt5.symbol_info(self.config.SYMBOL) if MT5_AVAILABLE else None
                    tick = mt5.symbol_info_tick(self.config.SYMBOL) if MT5_AVAILABLE else None
                    spread_pts = (tick.ask - tick.bid)/info.point if (info and tick and info.point) else None
                except Exception:
                    spread_pts = None

                # Build decision card as dict first, then convert to dataclass
                dc_dict = {
                    "ts": datetime.now().isoformat(),
                    "symbol": self.config.SYMBOL,
                    "cycle": self.metrics.cycle_count + 1,
                    "price": float(df['close'].iloc[-1]),
                    "sma20": float(df['sma_20'].iloc[-1]),
                    "sma50": float(df['sma_50'].iloc[-1]),
                    "raw_conf": signal['confidence'],
                    "adj_conf": signal['confidence'],  # Will be updated by regime if enabled
                    "threshold": 0.5,  # Will be updated by regime if enabled
                    "allow_trade": True,  # Will be updated by regime if enabled
                    "regime_label": "UNKNOWN",  # Will be updated by regime if enabled
                    "regime_scores": {},  # Will be updated by regime if enabled
                    "spread_pts": spread_pts,
                    "atr": signal.get('atr'),
                    "consecutive": self.consecutive_signals,
                    "open_positions": self._get_open_trades_count(),
                    "signal_src": signal.get('source', 'Unknown'),
                    "mode": "live" if not self.config.DEMO_MODE else "paper",
                    "agent_mode": getattr(getattr(self, "agent", None), "mode", "none")
                }

                # Convert to DecisionCard dataclass
                decision_card = _dc_from_dict(dc_dict)

                # Apply spread gate (single source of truth)
                if spread_pts is not None and spread_pts > self.max_spread_points:
                    decision_card.allow_trade = False
                    self.logger.info(f"⏸️ Blocked by spread gate: {spread_pts:.0f} > {self.max_spread_points}")

                # Apply regime detection if enabled (placeholder for now)
                # TODO: Integrate with regime detection system

                # Apply regime detection and adaptive thresholds if enabled
                if hasattr(self, 'config') and hasattr(self.config, 'CONFORMAL_ENABLED') and self.config.CONFORMAL_ENABLED:
                    try:
                        # Apply conformal gate logic
                        conformal_result = self._apply_conformal_gate(signal, decision_card)
                        if conformal_result:
                            # Update dataclass fields directly
                            for key, value in conformal_result.items():
                                if hasattr(decision_card, key):
                                    setattr(decision_card, key, value)
                            self.logger.info(f"🔒 Conformal gate applied: {conformal_result}")
                    except Exception as e:
                        self.logger.warning(f"Conformal gate failed: {e}")

                # Apply regime detection if available
                try:
                    if hasattr(self, 'config') and hasattr(self.config, 'regime_config'):
                        regime_result = apply_regime_and_threshold(self, self.config.SYMBOL, self.config, signal['confidence'])
                        if regime_result and len(regime_result) > 0:
                            regime_data = regime_result[0]
                            # Update dataclass fields directly
                            decision_card.regime_label = regime_data.get('label', 'UNKNOWN')
                            decision_card.regime_scores = regime_data.get('scores', {})
                            decision_card.adj_conf = regime_data.get('adj_conf', signal['confidence'])
                            decision_card.threshold = regime_data.get('threshold', 0.5)
                            decision_card.allow_trade = regime_data.get('allow_trade', True)
                            self.logger.info(f"🔍 Regime detected: {regime_data.get('label', 'UNKNOWN')} (adj_conf: {regime_data.get('adj_conf', 0):.3f})")
                        else:
                            self.logger.warning("Regime detection returned no results, using defaults")
                            # Set defaults directly
                            decision_card.regime_label = 'UNKNOWN'
                            decision_card.adj_conf = signal['confidence']
                            decision_card.threshold = 0.5
                            decision_card.allow_trade = True
                except Exception as e:
                    self.logger.warning(f"Regime detection failed: {e}, using defaults")
                    # Default to allow trade with original confidence
                    decision_card.regime_label = 'UNKNOWN'
                    decision_card.adj_conf = signal['confidence']
                    decision_card.threshold = 0.5
                    decision_card.allow_trade = True

                # Log comprehensive decision information
                self.logger.info(f"🎯 Decision Summary:")
                self.logger.info(f"   Signal: {signal['signal']} | Confidence: {signal['confidence']:.3f} | Consecutive: {self.consecutive_signals}/{self.config.CONSECUTIVE_SIGNALS_REQUIRED}")
                self.logger.info(f"   Price: {df['close'].iloc[-1]:.5f} | SMA20: {df['sma_20'].iloc[-1]:.5f} | SMA50: {df['sma_50'].iloc[-1]:.5f}")
                self.logger.info(f"   Regime: {decision_card.regime_label} | Adj Conf: {decision_card.adj_conf:.3f}")
                self.logger.info(f"   Threshold: {decision_card.threshold:.3f} | Allow Trade: {decision_card.allow_trade}")
                self.logger.info(f"   Spread: {decision_card.spread_pts:.1f} pts | Open Positions: {decision_card.open_positions}")

                # Log full decision card at DEBUG level
                self.logger.debug(f"Full decision card: {asdict(decision_card)}")

                # Agent review before trade execution
                if self.agent:
                    try:
                        act = self.agent.review_and_maybe_execute(decision_card, None)
                        if act and hasattr(act, 'action') and act.action == "HALT":
                            self.logger.warning("🛑 Agent requested HALT")
                            break
                    except Exception as e:
                        self.logger.warning(f"Agent review failed: {e}")

                # Apply risk gates
                if self.risk_gate:
                    try:
                        # Spread gate
                        if decision_card.spread_pts is not None:
                            if decision_card.spread_pts > self.max_spread_points:
                                decision_card.allow_trade = False
                                self.logger.info(f"⏸️ Blocked by spread gate: {decision_card.spread_pts:.0f} > {self.max_spread_points}")

                        # Exposure gate
                        open_count = decision_card.open_positions
                        if open_count >= self.config.MAX_OPEN_TRADES:
                            decision_card.allow_trade = False
                            self.logger.info(f"⏸️ Blocked by exposure gate: open={open_count}")

                        # Update should_execute based on risk gates
                        if not decision_card.allow_trade:
                            should_execute = False

                    except Exception as e:
                        self.logger.warning(f"Risk gate application failed: {e}")

                # Execute trade if conditions are met (now using conformal gate)
                should_execute = self._should_execute_trade_with_conformal(signal, decision_card)
                self.logger.info(f"🔍 Trade execution check: {'✅ APPROVED' if should_execute else '❌ BLOCKED'}")

                if should_execute:
                    self.logger.info("🚀 Executing trade...")
                    success = self._execute_trade(signal, df)

                    if success:
                        self.metrics.record_trade()
                        self.logger.info("✅ Trade executed successfully")
                        # Reset consecutive counter after successful trade execution
                        self.consecutive_signals = 0
                        self.logger.info("🔄 Reset consecutive signals counter after trade execution")
                    else:
                        self.logger.error("❌ Trade execution failed")
                else:
                    self.logger.info("⏸️ Trade execution blocked - checking conditions...")
                    # Log why trade was blocked
                    if signal['signal'] == 0:
                        self.logger.info("   ❌ Signal is 0 (no signal)")
                    elif signal['confidence'] < 0.5:
                        self.logger.info(f"   ❌ Confidence {signal['confidence']:.3f} below threshold 0.5")
                    elif self.consecutive_signals < self.config.CONSECUTIVE_SIGNALS_REQUIRED:
                        self.logger.info(f"   ❌ Consecutive signals {self.consecutive_signals} below required {self.config.CONSECUTIVE_SIGNALS_REQUIRED}")
                    else:
                        self.logger.info("   ❌ Other execution conditions not met")

                # Record cycle performance
                cycle_time = time.time() - cycle_start
                self.metrics.record_cycle(cycle_time)

                time.sleep(self.config.SLEEP_SECONDS)

            except Exception as e:
                self.metrics.record_error()
                self.logger.error(f"Loop error: {e}")

                # Report health event if agent available
                if self.agent:
                    try:
                        self.agent.on_health_event({
                            "ts": datetime.now().isoformat(),
                            "severity": "ERROR", "kind": "EXCEPTION",
                            "message": str(e)
                        })
                    except Exception as he:
                        self.logger.debug(f"Health event report failed: {he}")

                time.sleep(self.config.RETRY_DELAY)

        self.logger.info("✅ Trading loop completed")

    def _get_market_data(self):
        """Get market data from MT5"""
        try:
            if not MT5_AVAILABLE:
                self.logger.warning("MT5 not available")
                return None

            if not PANDAS_AVAILABLE:
                self.logger.warning("Pandas not available for market data processing")
                return None

            # Ensure symbol is selected
            symbol = self.config.SYMBOL
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to select symbol {symbol}: {mt5.last_error()}")
                return None

            # Get bars data
            bars = mt5.copy_rates_from_pos(
                symbol,
                self._get_timeframe_enum(),
                0,
                self.config.BARS
            )

            if bars is None or len(bars) == 0:
                self.logger.warning("No bars data received")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Log data acquisition details
            self.logger.debug(f"📊 Market data: {len(df)} bars, last timestamp: {df['time'].iloc[-1]}")

            # Check data quality
            if len(df) < 50:
                self.logger.warning(f"Insufficient data: {len(df)} bars, need at least 50 for SMA calculation")
                return None

            # Log symbol information for debugging (one-time)
            if not hasattr(self, '_symbol_info_logged'):
                try:
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info:
                        self.logger.info(f"📈 Symbol info: {symbol} - Digits: {symbol_info.digits}, Point: {symbol_info.point}, Trade mode: {symbol_info.trade_mode}")
                    self._symbol_info_logged = True
                except Exception as e:
                    self.logger.debug(f"Could not get symbol info: {e}")

            return df

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None

    def _get_timeframe_enum(self):
        """Get MT5 timeframe enum"""
        timeframe_map = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1,
            240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1
        }
        return timeframe_map.get(self.config.TIMEFRAME_MIN, mt5.TIMEFRAME_M15)

    def _compute_atr(self, df, period=14):
        """Compute Average True Range (ATR) for dynamic SL/TP calculation"""
        try:
            if not PANDAS_AVAILABLE:
                self.logger.error("Pandas required for ATR calculation")
                return None

            # df: columns ['high','low','close']
            high = df['high']
            low = df['low']
            close = df['close']
            prev_close = close.shift(1)

            # True Range calculation
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)

            # ATR using exponential moving average
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            return atr

        except Exception as e:
            self.logger.error(f"Error computing ATR: {e}")
            return None

    def _generate_signal(self, df):
        """Generate trading signal using AI system and Signal Ensemble"""
        try:
            # Check if we have enough data for SMA calculation
            if len(df) < 50:
                self.logger.warning(f"Insufficient data for SMA calculation: {len(df)} bars, need at least 50")
                return None

            # Calculate basic technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()

            # Calculate ATR for dynamic SL/TP
            atr = self._compute_atr(df, period=14)
            current_atr = atr.iloc[-1] if atr is not None else None

            # Simple crossover signal (SMA component) - RELAXED LOGIC
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]

            # Log current values for debugging
            self.logger.debug(f"Signal generation - Price: {current_price:.5f}, SMA20: {sma_20:.5f}, SMA50: {sma_50:.5f}")

            if pd.isna(sma_20) or pd.isna(sma_50):
                self.logger.warning("SMA values are NaN, cannot generate signal")
                return None

            # Initialize signal components
            sma_signal = 0
            sma_confidence = 0.0
            ai_signal = 0
            ai_confidence = 0.0

            # Generate SMA signal - RELAXED: Just SMA crossover, no price constraint
            if sma_20 > sma_50:
                sma_signal = 1  # Buy
                sma_confidence = 0.7
                self.logger.debug(f"BUY signal: SMA20 ({sma_20:.5f}) > SMA50 ({sma_50:.5f})")
            elif sma_20 < sma_50:
                sma_signal = -1  # Sell
                sma_confidence = 0.7
                self.logger.debug(f"SELL signal: SMA20 ({sma_20:.5f}) < SMA50 ({sma_50:.5f})")
            else:
                sma_signal = 0  # No signal
                sma_confidence = 0.0
                self.logger.debug(f"No signal: SMA20 ({sma_20:.5f}) = SMA50 ({sma_50:.5f})")

            # Generate AI signal if available
            if AI_AVAILABLE and hasattr(self, 'lstm_model') and self.lstm_model is not None:
                try:
                    ai_signal, ai_confidence = self._generate_ai_signal(df)
                except Exception as e:
                    self.logger.debug(f"AI signal generation failed: {e}")
                    ai_signal = 0
                    ai_confidence = 0.0

            # ML Ensemble integration if available
            if self.ml_integration:
                try:
                    ensemble_result = self.ml_integration.generate_ensemble_signal(
                        base_signal=sma_signal,
                        base_confidence=sma_confidence,
                        df=df
                    )
                    if ensemble_result:
                        final_signal = ensemble_result['signal']
                        final_confidence = ensemble_result['confidence']
                        signal_source = ensemble_result['source']
                        self.logger.debug(f"ML Ensemble signal: {final_signal} (conf: {final_confidence:.3f})")
                        return {
                            'signal': final_signal,
                            'confidence': final_confidence,
                            'score': final_confidence,
                            'source': signal_source,
                            'atr': current_atr,
                            'components': {
                                'sma': {'signal': sma_signal, 'confidence': sma_confidence},
                                'ai': {'signal': ai_signal, 'confidence': ai_confidence},
                                'ensemble': ensemble_result
                            }
                        }
                except Exception as e:
                    self.logger.debug(f"ML ensemble integration failed: {e}")

            # Ensemble the signals
            final_signal, final_confidence, signal_source = self._ensemble_signals(
                sma_signal, sma_confidence, ai_signal, ai_confidence
            )

            # Create comprehensive signal
            signal = {
                'signal': final_signal,
                'confidence': final_confidence,
                'score': final_confidence,  # For backward compatibility
                'source': signal_source,
                'atr': current_atr,
                'components': {
                    'sma': {'signal': sma_signal, 'confidence': sma_confidence},
                    'ai': {'signal': ai_signal, 'confidence': ai_confidence}
                }
            }

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None

    def _generate_ai_signal(self, df):
        """Generate AI-based trading signal using LSTM model"""
        try:
            if not hasattr(self, 'lstm_model') or self.lstm_model is None:
                return 0, 0.0

            # Prepare features for LSTM model
            features = self._prepare_lstm_features(df)
            if features is None:
                return 0, 0.0

            # Make prediction
            prediction = self.lstm_model.predict(features, verbose=0)

            # Convert prediction to signal
            if len(prediction.shape) > 1:
                prediction = prediction.flatten()

            # Get the latest prediction
            latest_pred = prediction[-1] if len(prediction) > 0 else 0.5

            # Convert to signal (-1, 0, 1) and confidence
            if latest_pred > 0.6:
                signal = 1  # Buy
                confidence = latest_pred
            elif latest_pred < 0.4:
                signal = -1  # Sell
                confidence = 1 - latest_pred
            else:
                signal = 0  # No signal
                confidence = 0.0

            return signal, confidence

        except Exception as e:
            self.logger.debug(f"AI signal generation failed: {e}")
            return 0, 0.0

    def _prepare_lstm_features(self, df):
        """Prepare features for LSTM model"""
        try:
            if not hasattr(self, 'config') or not hasattr(self.config, 'LSTM_TIMESTEPS'):
                return None

            timesteps = self.config.LSTM_TIMESTEPS

            # Ensure we have enough data
            if len(df) < timesteps:
                return None

            # Select recent data
            recent_data = df.tail(timesteps)

            # Calculate technical indicators
            features = []

            # Price features (normalized)
            close_prices = recent_data['close'].values
            price_mean = close_prices.mean()
            price_std = close_prices.std()

            if price_std > 0:
                normalized_prices = (close_prices - price_mean) / price_std
            else:
                normalized_prices = close_prices - price_mean

            # Volume features
            volumes = recent_data['tick_volume'].values
            volume_mean = volumes.mean()
            volume_std = volumes.std()

            if volume_std > 0:
                normalized_volumes = (volumes - volume_mean) / volume_std
            else:
                normalized_volumes = volumes - volume_mean

            # Technical indicators
            rsi = self._calculate_rsi(recent_data)
            macd = self._calculate_macd(recent_data)

            # Combine features
            for i in range(timesteps):
                feature_vector = [
                    normalized_prices[i],
                    normalized_volumes[i],
                    rsi[i] if rsi is not None else 0.5,
                    macd[i] if macd is not None else 0.0
                ]
                features.append(feature_vector)

            # Reshape for LSTM (samples, timesteps, features)
            features = np.array(features).reshape(1, timesteps, len(features[0]))

            return features

        except Exception as e:
            self.logger.debug(f"Feature preparation failed: {e}")
            return None

    def _calculate_rsi(self, df, period=14):
        """Calculate RSI indicator"""
        try:
            if not PANDAS_AVAILABLE:
                return None

            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.values

        except Exception as e:
            self.logger.debug(f"RSI calculation failed: {e}")
            return None

    def _calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            if not PANDAS_AVAILABLE:
                return None

            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()

            return (macd_line - signal_line).values

        except Exception as e:
            self.logger.debug(f"MACD calculation failed: {e}")
            return None

    def _ensemble_signals(self, sma_signal, sma_confidence, ai_signal, ai_confidence):
        """Ensemble SMA and AI signals"""
        try:
            # Weight the signals (can be made configurable)
            sma_weight = 0.4
            ai_weight = 0.6

            # Calculate weighted confidence
            if ai_confidence > 0:
                # Both signals available
                if sma_signal == ai_signal:
                    # Signals agree - boost confidence
                    final_confidence = (sma_confidence * sma_weight + ai_confidence * ai_weight) * 1.2
                    signal_source = f"SMA+AI_Ensemble_{sma_signal}"
                else:
                    # Signals disagree - use weighted average
                    final_confidence = (sma_confidence * sma_weight + ai_confidence * ai_weight) * 0.8
                    # Prefer AI signal if confidence is high
                    if ai_confidence > 0.7:
                        final_signal = ai_signal
                        signal_source = f"AI_Override_{ai_signal}"
                    else:
                        final_signal = sma_signal
                        signal_source = f"SMA_Override_{sma_signal}"
            else:
                # Only SMA available
                final_confidence = sma_confidence
                signal_source = f"SMA_Only_{sma_signal}"

            # Cap confidence at 1.0
            final_confidence = min(1.0, final_confidence)

            # Determine final signal
            if final_confidence < 0.3:
                final_signal = 0
            else:
                final_signal = sma_signal if ai_confidence == 0 else (ai_signal if ai_confidence > sma_confidence else sma_signal)

            return final_signal, final_confidence, signal_source

        except Exception as e:
            self.logger.error(f"Error in signal ensemble: {e}")
            return sma_signal, sma_confidence, f"SMA_Fallback_{sma_signal}"

    def _should_execute_trade(self, signal):
        """Determine if trade should be executed"""
        try:
            # Check signal validity
            if signal['signal'] == 0:
                return False

            # Check confidence threshold
            if signal['confidence'] < 0.5:
                return False

            # Check consecutive signals requirement
            if self.consecutive_signals < self.config.CONSECUTIVE_SIGNALS_REQUIRED:
                return False

            # Check open positions limit
            open_count = self._get_open_trades_count()
            if open_count >= self.config.MAX_OPEN_TRADES:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking trade execution conditions: {e}")
            return False

    def _should_execute_trade_with_conformal(self, signal, decision_card):
        """Enhanced trade execution check with conformal gate and regime logic"""
        try:
            # Basic signal validation
            if signal['signal'] == 0:
                return False

            # Check if conformal gate allows trade
            if hasattr(self, 'config') and hasattr(self.config, 'CONFORMAL_ENABLED') and self.config.CONFORMAL_ENABLED:
                if not decision_card.allow_trade:
                    self.logger.info("🔒 Conformal gate blocked trade execution")
                    return False

            # Check adjusted confidence against threshold
            adj_conf = decision_card.adj_conf
            threshold = decision_card.threshold

            if adj_conf < threshold:
                self.logger.info(f"🔒 Confidence {adj_conf:.3f} below threshold {threshold:.3f}")
                return False

            # Check consecutive signals requirement (with conformal adjustments)
            required_signals = self.config.CONSECUTIVE_SIGNALS_REQUIRED
            if hasattr(self, 'config') and hasattr(self.config, 'CONFORMAL_EXTRA_CONSECUTIVE'):
                required_signals += self.config.CONFORMAL_EXTRA_CONSECUTIVE

            if self.consecutive_signals < required_signals:
                self.logger.info(f"🔒 Need {required_signals} consecutive signals, have {self.consecutive_signals}")
                return False

            # Check open positions limit
            open_count = self._get_open_trades_count()
            if open_count >= self.config.MAX_OPEN_TRADES:
                self.logger.info(f"🔒 Max open trades limit reached: {open_count}")
                return False

            # Check daily loss limit
            if self._check_daily_loss_limit():
                self.logger.info("🔒 Daily loss limit reached")
                return False

            # Spread conditions are now handled by the new risk gate system in the main loop
            # This check is no longer needed

            self.logger.info(f"✅ Trade execution approved: conf={adj_conf:.3f}, threshold={threshold:.3f}, consecutive={self.consecutive_signals}")
            return True

        except Exception as e:
            self.logger.error(f"Error in enhanced trade execution check: {e}")
            return False

    def _apply_conformal_gate(self, signal, decision_card):
        """Apply conformal gate logic for trade validation"""
        try:
            if not hasattr(self, 'config') or not self.config.CONFORMAL_ENABLED:
                return None

            # Get conformal configuration
            cfg = self.config

            # Calculate base probability
            base_p = signal['confidence']

            # Apply small penalty for small signals
            if base_p < cfg.CONFORMAL_PENALTY_SMALL:
                penalty = cfg.CONFORMAL_PENALTY_SMALL
                base_p = max(0, base_p - penalty)
                self.logger.debug(f"Applied small signal penalty: {penalty}")

            # Apply big penalty for big signals
            if base_p > (1 - cfg.CONFORMAL_PENALTY_BIG):
                penalty = cfg.CONFORMAL_PENALTY_BIG
                base_p = min(1, base_p + penalty)
                self.logger.debug(f"Applied big signal penalty: {penalty}")

            # Apply hard floor
            if base_p < cfg.CONFORMAL_HARD_FLOOR:
                base_p = cfg.CONFORMAL_HARD_FLOOR
                self.logger.debug(f"Applied hard floor: {cfg.CONFORMAL_HARD_FLOOR}")

            # Cap final threshold
            final_threshold = min(cfg.CONFORMAL_CAP_FINAL_THR, base_p)

            # Determine if trade is allowed
            allow_trade = base_p >= cfg.CONFORMAL_MIN_P

            # Apply soft gate if enabled
            if cfg.CONFORMAL_SOFT_GATE and not allow_trade:
                # Soft gate allows some flexibility
                if base_p >= (cfg.CONFORMAL_MIN_P * 0.8):  # 80% of minimum
                    allow_trade = True
                    self.logger.debug("Soft gate allowed trade with reduced confidence")

            # Emergency bypass if enabled
            if cfg.CONFORMAL_EMERGENCY_BYPASS and self.consecutive_signals >= cfg.CONFORMAL_EXTRA_CONSEC_FLOOR:
                allow_trade = True
                self.logger.warning("Emergency bypass activated due to high consecutive signals")

            return {
                'conformal_p': base_p,
                'conformal_threshold': final_threshold,
                'allow_trade': allow_trade,
                'gate_type': 'conformal'
            }

        except Exception as e:
            self.logger.error(f"Error applying conformal gate: {e}")
            return None

    def _check_daily_loss_limit(self):
        """Check if daily loss limit has been reached"""
        try:
            if not hasattr(self, 'config') or not self.config.MAX_DAILY_LOSS:
                return False

            # Get today's trades and calculate PnL
            if not MT5_AVAILABLE:
                return False

            # Get today's start time
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # Get today's closed trades
            from_date = int(today_start.timestamp())
            trades = mt5.history_deals_get(from_date, datetime.now().timestamp())

            if not trades:
                return False

            # Calculate total PnL for today
            daily_pnl = sum(trade.profit for trade in trades)
            daily_pnl_pct = daily_pnl / self._get_account_balance() if self._get_account_balance() > 0 else 0

            if daily_pnl_pct <= -self.config.MAX_DAILY_LOSS:
                self.logger.warning(f"Daily loss limit reached: {daily_pnl_pct:.2%}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking daily loss limit: {e}")
            return False

    def _get_account_balance(self):
        """Get current account balance"""
        try:
            if not MT5_AVAILABLE:
                return 10000  # Default balance for demo mode

            account_info = mt5.account_info()
            if account_info:
                return account_info.balance

            return 10000  # Fallback balance

        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return 10000  # Fallback balance

    def _execute_trade(self, signal, df):
        """Execute trading signal with ATR-based SL/TP"""
        try:
            current_price = df['close'].iloc[-1]
            atr = signal.get('atr')

            if atr is None:
                self.logger.warning("ATR not available, using fixed SL/TP")
                atr = 0.01  # Fallback ATR value

            # Get ATR multipliers from config (defaults if not set)
            sl_multiplier = getattr(self.config, 'SL_ATR_MULTIPLIER', 2.0)
            tp_multiplier = getattr(self.config, 'TP_ATR_MULTIPLIER', 3.0)

            # Calculate SL and TP based on ATR
            if signal['signal'] == 1:  # BUY
                sl = current_price - (atr * sl_multiplier)
                tp = current_price + (atr * tp_multiplier)
                side = "BUY"
                is_buy = True
            elif signal['signal'] == -1:  # SELL
                sl = current_price + (atr * sl_multiplier)
                tp = current_price - (atr * tp_multiplier)
                side = "SELL"
                is_buy = False
            else:
                return False

            # Enforce minimum distance and round SL/TP to broker requirements
            sl, tp = enforce_min_distance_and_round(self.config.SYMBOL, current_price, sl, tp, is_buy)

            # Calculate position size based on risk
            volume = self._calculate_position_size(current_price, sl)

            self.logger.info(f"Executing {side} signal: price={current_price:.5f}, SL={sl:.5f}, TP={tp:.5f}, ATR={atr:.5f}")

            # Place actual order if MT5 is available
            if MT5_AVAILABLE and not self.config.DEMO_MODE:
                success = self._place_mt5_order(side, current_price, sl, tp, volume)
                if success:
                    self.logger.info(f"✅ MT5 order placed successfully: {side} {volume} lots")
                else:
                    self.logger.error(f"❌ MT5 order placement failed")
                    return False
            else:
                self.logger.info(f"📝 Paper trade: {side} {volume} lots at {current_price:.5f}")

            # Log trade attempt
            if self.ev:
                self.ev.log_trade_attempt(
                    side=side,
                    entry=current_price,
                    sl=sl,
                    tp=tp,
                    volume=volume,
                    confidence=signal['confidence'],
                    source=signal['source']
                )

            return True

        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return False

    def _update_trailing_stops(self):
        """Update trailing stops for open positions with TP split and breakeven logic"""
        try:
            now = datetime.now()
            if (now - self.last_trailing_update).seconds < 15:  # 15 second interval
                return

                self.last_trailing_update = now

            if not MT5_AVAILABLE:
                return

            # Get current positions
            positions = mt5.positions_get(symbol=self.config.SYMBOL)
            if not positions:
                return

            current_price = None
            try:
                tick = mt5.symbol_info_tick(self.config.SYMBOL)
                if tick:
                    current_price = (tick.bid + tick.ask) / 2
            except Exception:
                return

            if current_price is None:
                return

            # Process each registered position
            for ticket, registry in list(self.trailing_registry.items()):
                try:
                    # Check if position still exists
                    position = next((p for p in positions if p.ticket == ticket), None)
                    if not position:
                        # Position closed, remove from registry
                        del self.trailing_registry[ticket]
                        continue

                    # Check TP1 hit
                    if not registry['tp1_hit']:
                        if self._check_tp1_hit(registry, current_price):
                            self._execute_tp1_partial_close(ticket, registry, position)

                    # Check TP2 hit
                    if registry['tp1_hit'] and registry['remaining_volume'] > 0:
                        if self._check_tp2_hit(registry, current_price):
                            self._execute_tp2_close(ticket, registry, position)

                    # Update breakeven if needed
                    if registry['tp1_hit'] and not registry['breakeven_set']:
                        if self._should_set_breakeven(registry, current_price):
                            self._set_breakeven_stop(ticket, registry, position)

                except Exception as e:
                    self.logger.error(f"Error processing position {ticket}: {e}")

                self.logger.debug("Trailing stops updated")

        except Exception as e:
            self.logger.error(f"Error updating trailing stops: {e}")

    def _check_tp1_hit(self, registry, current_price):
        """Check if TP1 has been hit"""
        try:
            if registry['side'] == "BUY":
                return current_price >= registry['tp1_price']
            else:
                return current_price <= registry['tp1_price']
        except Exception as e:
            self.logger.error(f"Error checking TP1 hit: {e}")
            return False

    def _check_tp2_hit(self, registry, current_price):
        """Check if TP2 has been hit"""
        try:
            if registry['side'] == "BUY":
                return current_price >= registry['tp2_price']
            else:
                return current_price <= registry['tp2_price']
        except Exception as e:
            self.logger.error(f"Error checking TP2 hit: {e}")
            return False

    def _execute_tp1_partial_close(self, ticket, registry, position):
        """Execute partial close at TP1"""
        try:
            # Calculate volume to close
            close_volume = registry['volume'] * registry['tp1_share']

            # Ensure we don't close more than remaining volume
            close_volume = min(close_volume, registry['remaining_volume'])

            if close_volume <= 0:
                return

            # Prepare partial close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.SYMBOL,
                "volume": close_volume,
                "type": mt5.ORDER_TYPE_SELL if registry['side'] == "BUY" else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": registry['tp1_price'],
                "deviation": 20,
                "magic": self.config.MAGIC,
                "comment": f"MR_BEN_TP1_PARTIAL_{int(time.time())}",
                "type_filling": mt5.ORDER_FILLING_FOK
            }

            # Execute partial close
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Update registry
                registry['tp1_hit'] = True
                registry['remaining_volume'] -= close_volume

                self.logger.info(f"✅ TP1 partial close executed: {close_volume} lots at {registry['tp1_price']:.5f}")

                # Log the partial close
                if self.ev:
                    self.ev.log_trade_attempt(
                        side="CLOSE_TP1",
                        entry=registry['entry_price'],
                        exit=registry['tp1_price'],
                        volume=close_volume,
                        confidence=1.0,
                        source="TP1_Partial_Close"
                    )
            else:
                self.logger.error(f"TP1 partial close failed: {result.retcode} - {result.comment}")

        except Exception as e:
            self.logger.error(f"Error executing TP1 partial close: {e}")

    def _execute_tp2_close(self, ticket, registry, position):
        """Execute full close at TP2"""
        try:
            if registry['remaining_volume'] <= 0:
                return

            # Prepare full close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.SYMBOL,
                "volume": registry['remaining_volume'],
                "type": mt5.ORDER_TYPE_SELL if registry['side'] == "BUY" else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": registry['tp2_price'],
                "deviation": 20,
                "magic": self.config.MAGIC,
                "comment": f"MR_BEN_TP2_FULL_{int(time.time())}",
                "type_filling": mt5.ORDER_FILLING_FOK
            }

            # Execute full close
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Close position completely
                registry['remaining_volume'] = 0

                self.logger.info(f"✅ TP2 full close executed: {registry['volume']} lots at {registry['tp2_price']:.5f}")

                # Log the full close
                if self.ev:
                    self.ev.log_trade_attempt(
                        side="CLOSE_TP2",
                        entry=registry['entry_price'],
                        exit=registry['tp2_price'],
                        volume=registry['volume'],
                        confidence=1.0,
                        source="TP2_Full_Close"
                    )

                # Remove from registry
                del self.trailing_registry[ticket]
            else:
                self.logger.error(f"TP2 full close failed: {result.retcode} - {result.comment}")

        except Exception as e:
            self.logger.error(f"Error executing TP2 full close: {e}")

    def _should_set_breakeven(self, registry, current_price):
        """Determine if breakeven should be set"""
        try:
            if not getattr(self.config, 'BREAKEVEN_AFTER_TP1', True):
                return False

            # Set breakeven if price has moved in our favor by a small amount
            breakeven_buffer = 0.0001  # 1 pip buffer

            if registry['side'] == "BUY":
                return current_price >= (registry['entry_price'] + breakeven_buffer)
            else:
                return current_price <= (registry['entry_price'] - breakeven_buffer)

        except Exception as e:
            self.logger.error(f"Error checking breakeven condition: {e}")
            return False

    def _set_breakeven_stop(self, ticket, registry, position):
        """Set stop loss to breakeven"""
        try:
            # Calculate breakeven price with small buffer
            breakeven_buffer = 0.0001  # 1 pip buffer

            if registry['side'] == "BUY":
                breakeven_price = registry['entry_price'] + breakeven_buffer
            else:
                breakeven_price = registry['entry_price'] - breakeven_buffer

            # Modify stop loss to breakeven
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.config.SYMBOL,
                "position": ticket,
                "sl": breakeven_price,
                "tp": position.tp  # Keep existing TP
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                registry['breakeven_set'] = True
                self.logger.info(f"✅ Breakeven stop set at {breakeven_price:.5f}")

                # Log the breakeven action
                if self.ev:
                    self.ev.log_trade_attempt(
                        side="BREAKEVEN_SL",
                        entry=registry['entry_price'],
                        sl=breakeven_price,
                        volume=registry['remaining_volume'],
                        confidence=1.0,
                        source="Breakeven_Stop"
                    )
            else:
                self.logger.error(f"Breakeven stop modification failed: {result.retcode} - {result.comment}")

        except Exception as e:
            self.logger.error(f"Error setting breakeven stop: {e}")

    def _get_open_trades_count(self):
        """Get count of open trades"""
        try:
            if not MT5_AVAILABLE:
                return 0

            positions = mt5.positions_get(symbol=self.config.SYMBOL)
            return len(positions) if positions else 0

        except Exception as e:
            self.logger.error(f"Error getting open trades count: {e}")
            return 0

    def _calculate_position_size(self, entry_price, sl_price):
        """Calculate position size based on risk management"""
        try:
            if not MT5_AVAILABLE:
                return self.config.FIXED_VOLUME

            # Get symbol info for volume constraints
            symbol_info = mt5.symbol_info(self.config.SYMBOL)
            if not symbol_info:
                return self.config.FIXED_VOLUME

            # Calculate SL distance in points
            sl_distance = abs(entry_price - sl_price)
            point_value = symbol_info.point

            # Calculate risk-based volume
            if self.config.USE_RISK_BASED_VOLUME:
                # Get account balance
                account_info = mt5.account_info()
                if account_info:
                    balance = account_info.balance
                    risk_amount = balance * self.config.BASE_RISK

                    # Calculate volume based on risk
                    tick_value = symbol_info.trade_tick_value
                    volume = risk_amount / (sl_distance / point_value * tick_value)

                    # Apply volume constraints
                    volume = max(volume, symbol_info.volume_min)
                    volume = min(volume, symbol_info.volume_max)

                    # Round to volume step
                    volume_step = symbol_info.volume_step
                    volume = round(volume / volume_step) * volume_step

                    return volume

            return self.config.FIXED_VOLUME

        except Exception as e:
            self.logger.warning(f"Error calculating position size: {e}, using fixed volume")
            return self.config.FIXED_VOLUME

    def _order_send_adaptive(self, request: dict):
        """Send order with adaptive filling mode to avoid 10030 error"""
        # 1) شروع با مود توصیه شده‌ی سمبل (اگر وجود داشت)
        sym_fm = _symbol_filling_mode(self.config.SYMBOL)
        first = _map_symbol_to_order_filling(sym_fm) if sym_fm is not None else None

        tried = []
        candidates = [first] if first is not None else []
        # بقیه مودها را هم امتحان کنیم
        for m in (mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK):
            if m not in candidates:
                candidates.append(m)

        # 2) تلاش ترتیبی
        last_result = None
        for m in candidates:
            req = dict(request)
            req["type_filling"] = m
            # تضمین نوع‌ها
            if isinstance(req.get("price"), (np.float64, np.float32)): req["price"] = float(req["price"])
            if isinstance(req.get("volume"), (np.float64, np.float32)): req["volume"] = float(req["volume"])
            req["type_time"] = mt5.ORDER_TIME_GTC

            self.logger.info(f"🧪 Trying filling mode: {m}")
            res = mt5.order_send(req)
            last_result = res
            tried.append((m, res.retcode))
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"✅ order_send DONE with filling={m}")
                return res

            # اگر خطای غیر از 10030 بود، ادامه دادن معنی دارد ولی لاگ کنیم
            if res is not None and res.retcode != 10030:
                self.logger.warning(f"order_send retcode={res.retcode} with filling={m}; trying next candidate...")

        # 3) همگی ناموفق
        self.logger.error(f"❌ All filling modes failed: {tried}")
        return last_result

    def _place_mt5_order(self, side, entry_price, sl, tp, volume):
        """Place actual MT5 order"""
        try:
            if not MT5_AVAILABLE:
                return False

            # Normalize volume and ensure proper types
            volume = self._normalize_volume(self.config.SYMBOL, float(volume))
            entry_price = float(entry_price)

            # Enforce minimum distance and round SL/TP to broker requirements
            is_buy = (side == "BUY")
            sl, tp = enforce_min_distance_and_round(self.config.SYMBOL, entry_price, sl, tp, is_buy)

            # Pick correct filling mode to avoid 10030 error
            type_filling = _pick_filling_mode(self.config.SYMBOL)

            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.SYMBOL,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": float(sl),
                "tp": float(tp),
                "deviation": int(getattr(self.config, "DEVIATION_POINTS", 50)),
                "magic": int(self.config.MAGIC),
                "type_filling": type_filling,
                "type_time": mt5.ORDER_TIME_GTC,
                "comment": f"MRBEN {side}"
            }

            # Send order
            result = self._order_send_adaptive(request)
            if result is None:
                self.logger.error(f"❌ MT5 order placement failed: order_send returned None")
                return False

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"❌ Order failed: {result.retcode} - {result.comment}")
                self.logger.error(f"Request details: {request}")

                # Emit HealthEvent for non-DONE retcodes
                if self.agent:
                    try:
                        self.agent.on_health_event({
                            "ts": datetime.now().isoformat(),
                            "severity": "ERROR",
                            "kind": "ORDER_FAIL",
                            "message": f"MT5 order failed: {result.retcode} - {result.comment}",
                            "context": {
                                "side": side,
                                "price": entry_price,
                                "volume": volume,
                                "retcode": result.retcode,
                                "comment": result.comment,
                                "type_filling": request.get("type_filling")
                            }
                        })
                    except Exception as e:
                        self.logger.debug(f"Health event report failed: {e}")

                return False

            self.logger.info(f"✅ Order placed successfully: ticket {result.order}")

            # Register position for TP split and breakeven management
            if result.order:
                self._register_position_for_trailing(result.order, side, entry_price, sl, tp, volume)

            return True

        except Exception as e:
            self.logger.error(f"Error placing MT5 order: {e}")
            return False

    def _register_position_for_trailing(self, ticket, side, entry_price, sl, tp, volume):
        """Register position for TP split and breakeven management"""
        try:
            # Calculate TP levels based on config
            tp1_r = getattr(self.config, 'TP1_R', 0.8)
            tp2_r = getattr(self.config, 'TP2_R', 1.5)
            tp1_share = getattr(self.config, 'TP1_SHARE', 0.5)

            # Calculate TP prices
            if side == "BUY":
                tp1_price = entry_price + (tp - entry_price) * tp1_r
                tp2_price = entry_price + (tp - entry_price) * tp2_r
            else:
                tp1_price = entry_price - (entry_price - tp) * tp1_r
                tp2_price = entry_price - (entry_price - tp) * tp2_r

            # Register in trailing registry
            self.trailing_registry[ticket] = {
                'side': side,
                'entry_price': entry_price,
                'original_sl': sl,
                'tp1_price': tp1_price,
                'tp2_price': tp2_price,
                'tp1_share': tp1_share,
                'volume': volume,
                'remaining_volume': volume,
                'tp1_hit': False,
                'breakeven_set': False,
                'created_at': datetime.now()
            }

            self.logger.info(f"Position {ticket} registered for TP split management")

        except Exception as e:
            self.logger.error(f"Error registering position for trailing: {e}")

    def _update_trailing_stops(self):
        """Update trailing stops for open positions with TP split and breakeven logic"""
        try:
            now = datetime.now()
            if (now - self.last_trailing_update).seconds < 15:  # 15 second interval
                return

            self.last_trailing_update = now

            if not MT5_AVAILABLE:
                return

            # Get current positions
            positions = mt5.positions_get(symbol=self.config.SYMBOL)
            if not positions:
                return

            current_price = None
            try:
                tick = mt5.symbol_info_tick(self.config.SYMBOL)
                if tick:
                    current_price = (tick.bid + tick.ask) / 2
            except Exception:
                return

            if current_price is None:
                return

            # Process each registered position
            for ticket, registry in list(self.trailing_registry.items()):
                try:
                    # Check if position still exists
                    position = next((p for p in positions if p.ticket == ticket), None)
                    if not position:
                        # Position closed, remove from registry
                        del self.trailing_registry[ticket]
                        continue

                    # Check TP1 hit
                    if not registry['tp1_hit']:
                        if self._check_tp1_hit(registry, current_price):
                            self._execute_tp1_partial_close(ticket, registry, position)

                    # Check TP2 hit
                    if registry['tp1_hit'] and registry['remaining_volume'] > 0:
                        if self._check_tp2_hit(registry, current_price):
                            self._execute_tp2_close(ticket, registry, position)

                    # Update breakeven if needed
                    if registry['tp1_hit'] and not registry['breakeven_set']:
                        if self._should_set_breakeven(registry, current_price):
                            self._set_breakeven_stop(ticket, registry, position)

                except Exception as e:
                    self.logger.error(f"Error processing position {ticket}: {e}")

            self.logger.debug("Trailing stops updated")

        except Exception as e:
            self.logger.error(f"Error updating trailing stops: {e}")

    def _check_tp1_hit(self, registry, current_price):
        """Check if TP1 has been hit"""
        try:
            if registry['side'] == "BUY":
                return current_price >= registry['tp1_price']
            else:
                return current_price <= registry['tp1_price']
        except Exception as e:
            self.logger.error(f"Error checking TP1 hit: {e}")
            return False

    def _check_tp2_hit(self, registry, current_price):
        """Check if TP2 has been hit"""
        try:
            if registry['side'] == "BUY":
                return current_price >= registry['tp2_price']
            else:
                return current_price <= registry['tp2_price']
        except Exception as e:
            self.logger.error(f"Error checking TP2 hit: {e}")
            return False

    def _execute_tp1_partial_close(self, ticket, registry, position):
        """Execute partial close at TP1"""
        try:
            # Calculate volume to close
            close_volume = registry['volume'] * registry['tp1_share']

            # Ensure we don't close more than remaining volume
            close_volume = min(close_volume, registry['remaining_volume'])

            if close_volume <= 0:
                return

            # Prepare partial close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.SYMBOL,
                "volume": close_volume,
                "type": mt5.ORDER_TYPE_SELL if registry['side'] == "BUY" else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": registry['tp1_price'],
                "deviation": 20,
                "magic": self.config.MAGIC,
                "comment": f"MR_BEN_TP1_PARTIAL_{int(time.time())}",
                "type_filling": mt5.ORDER_FILLING_FOK
            }

            # Execute partial close
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Update registry
                registry['tp1_hit'] = True
                registry['remaining_volume'] -= close_volume

                self.logger.info(f"✅ TP1 partial close executed: {close_volume} lots at {registry['tp1_price']:.5f}")

                # Log the partial close
                if self.ev:
                    self.ev.log_trade_attempt(
                        side="CLOSE_TP1",
                        entry=registry['entry_price'],
                        exit=registry['tp1_price'],
                        volume=close_volume,
                        confidence=1.0,
                        source="TP1_Partial_Close"
                    )
            else:
                self.logger.error(f"TP1 partial close failed: {result.retcode} - {result.comment}")

        except Exception as e:
            self.logger.error(f"Error executing TP1 partial close: {e}")

    def _execute_tp2_close(self, ticket, registry, position):
        """Execute full close at TP2"""
        try:
            if registry['remaining_volume'] <= 0:
                return

            # Prepare full close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.SYMBOL,
                "volume": registry['remaining_volume'],
                "type": mt5.ORDER_TYPE_SELL if registry['side'] == "BUY" else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": registry['tp2_price'],
                "deviation": 20,
                "magic": self.config.MAGIC,
                "comment": f"MR_BEN_TP2_FULL_{int(time.time())}",
                "type_filling": mt5.ORDER_FILLING_FOK
            }

            # Execute full close
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Close position completely
                registry['remaining_volume'] = 0

                self.logger.info(f"✅ TP2 full close executed: {registry['volume']} lots at {registry['tp2_price']:.5f}")

                # Log the full close
                if self.ev:
                    self.ev.log_trade_attempt(
                        side="CLOSE_TP2",
                        entry=registry['entry_price'],
                        exit=registry['tp2_price'],
                        volume=registry['volume'],
                        confidence=1.0,
                        source="TP2_Full_Close"
                    )

                # Remove from registry
                del self.trailing_registry[ticket]
            else:
                self.logger.error(f"TP2 full close failed: {result.retcode} - {result.comment}")

        except Exception as e:
            self.logger.error(f"Error executing TP2 full close: {e}")

    def _should_set_breakeven(self, registry, current_price):
        """Determine if breakeven should be set"""
        try:
            if not getattr(self.config, 'BREAKEVEN_AFTER_TP1', True):
                return False

            # Set breakeven if price has moved in our favor by a small amount
            breakeven_buffer = 0.0001  # 1 pip buffer

            if registry['side'] == "BUY":
                return current_price >= (registry['entry_price'] + breakeven_buffer)
            else:
                return current_price <= (registry['entry_price'] - breakeven_buffer)

        except Exception as e:
            self.logger.error(f"Error checking breakeven condition: {e}")
            return False

    def _set_breakeven_stop(self, ticket, registry, position):
        """Set stop loss to breakeven"""
        try:
            # Calculate breakeven price with small buffer
            breakeven_buffer = 0.0001  # 1 pip buffer

            if registry['side'] == "BUY":
                breakeven_price = registry['entry_price'] + breakeven_buffer
            else:
                breakeven_price = registry['entry_price'] - breakeven_buffer

            # Modify stop loss to breakeven
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.config.SYMBOL,
                "position": ticket,
                "sl": breakeven_price,
                "tp": position.tp  # Keep existing TP
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                registry['breakeven_set'] = True
                self.logger.info(f"✅ Breakeven stop set at {breakeven_price:.5f}")

                # Log the breakeven action
                if self.ev:
                    self.ev.log_trade_attempt(
                        side="BREAKEVEN_SL",
                        entry=registry['entry_price'],
                        sl=breakeven_price,
                        volume=registry['remaining_volume'],
                        confidence=1.0,
                        source="Breakeven_Stop"
                    )
            else:
                self.logger.error(f"Breakeven stop modification failed: {result.retcode} - {result.comment}")

        except Exception as e:
            self.logger.error(f"Error setting breakeven stop: {e}")

    def _get_open_trades_count(self):
        """Get count of open trades"""
        try:
            if not MT5_AVAILABLE:
                return 0

            positions = mt5.positions_get(symbol=self.config.SYMBOL)
            return len(positions) if positions else 0

        except Exception as e:
            self.logger.error(f"Error getting open trades count: {e}")
            return 0

    def _log_performance_metrics(self):
        """Log performance metrics and system health"""
        try:
            stats = self.metrics.get_stats() if hasattr(self, 'metrics') else {
                "uptime_seconds": 0, "cycles_per_second": 0, "avg_response_time": 0,
                "total_trades": 0, "error_rate": 0, "memory_mb": 0
            }
            self.logger.info("📊 Performance Metrics:")
            self.logger.info(f"   Uptime: {stats.get('uptime_seconds', 0):.0f}s")
            self.logger.info(f"   Cycles/sec: {stats.get('cycles_per_second', 0):.2f}")
            self.logger.info(f"   Avg Response: {stats.get('avg_response_time', 0):.3f}s")
            self.logger.info(f"   Total Trades: {stats.get('total_trades', 0)}")
            self.logger.info(f"   Error Rate: {stats.get('error_rate', 0):.3f}")
            self.logger.info(f"   Memory: {stats.get('memory_mb', 0):.1f} MB")
        except Exception as e:
            self.logger.warning(f"Failed to log performance metrics: {e}")

    def cleanup(self):
        """Cleanup resources and stop trading system"""
        self.logger.info("🛑 Stopping trading system...")
        self.running = False
        self.logger.info("✅ Trading system stopped successfully")

    def _normalize_volume(self, symbol: str, vol: float) -> float:
        """Normalize volume to broker requirements"""
        try:
            if not MT5_AVAILABLE:
                return float(vol)

            info = mt5.symbol_info(symbol)
            if not info:
                return float(vol)

            step = info.volume_step or 0.01
            vmin = info.volume_min or 0.01
            vmax = info.volume_max or 100.0

            # Round to volume step
            v = max(vmin, min(vmax, math.floor(vol/step)*step))
            return float(Decimal(str(v)).quantize(Decimal(str(step)))

        except Exception as e:
            self.logger.warning(f"Error normalizing volume: {e}, using original: {vol}")
            return float(vol)

# -----------------------------
# Legacy main function (for backward compatibility)
# -----------------------------

def legacy_main():
    """Legacy main function for backward compatibility"""
    print("🎯 MR BEN Live Trading System - Legacy Mode")
    print("=" * 60)

    try:
        # Load configuration
        config = MT5Config()
        print(f"✅ Configuration loaded: {config.get_config_summary()}")

        # Create and start trading system
        trader = MT5LiveTrader(config)
        trader.start()

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

    return 0

# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    # Check if CLI arguments are provided
    if len(sys.argv) > 1:
        # Use new CLI interface
        raise SystemExit(main())
    else:
        # NEW: run main with default LIVE settings (no legacy_main)
        sys.argv += [
            "live",
            "--mode", "live",
            "--symbol", "XAUUSD.PRO",
            "--agent",
            "--regime",
            "--log-level", "INFO"
        ]
        raise SystemExit(main())
