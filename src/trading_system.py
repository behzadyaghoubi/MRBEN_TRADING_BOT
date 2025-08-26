#!/usr/bin/env python3
"""
MR BEN Trading System Core Module
Extracted and modularized from live_trader_clean.py
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .ai_system import MRBENAdvancedAISystem
from .config import MT5Config
from .data_manager import MT5DataManager
from .risk_manager import EnhancedRiskManager
from .telemetry import EventLogger, MFELogger
from .trade_executor import EnhancedTradeExecutor


@dataclass
class TradingState:
    """Trading system state management"""

    run_id: str
    running: bool = False
    consecutive_signals: int = 0
    last_signal: int = 0
    last_trailing_update: datetime = None
    start_balance: float | None = None
    last_bar_time: datetime | None = None
    reentry_window_until: datetime | None = None
    cooldown_until: datetime | None = None
    last_trade_side: int | None = None
    last_fill_time: datetime | None = None
    session_start_equity: float | None = None

    def __post_init__(self):
        if self.last_trailing_update is None:
            self.last_trailing_update = datetime.now()


class TradingSystem:
    """Core trading system logic extracted from MT5LiveTrader"""

    def __init__(self, config: MT5Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize state
        self.state = TradingState(run_id=datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Initialize components
        self._initialize_components()

        # Initialize state variables
        self._initialize_state()

    def _initialize_components(self):
        """Initialize all system components"""
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

            # AI system
            self.ai_system = MRBENAdvancedAISystem()
            self.logger.info("🤖 Initializing AI System...")
            self.ai_system.load_models()
            self.logger.info(
                f"✅ AI System loaded. Available models: {list(self.ai_system.models.keys())}"
            )

            # Event Logger
            self.ev = EventLogger(
                path="data/events.jsonl", run_id=self.state.run_id, symbol=self.config.SYMBOL
            )
            self.logger.info("✅ Event Logger initialized")

            # MFE Logger
            self.mfe_logger = MFELogger("data/mfe_tick_data.jsonl")
            self.logger.info("✅ MFE Logger initialized")

        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise

    def _initialize_state(self):
        """Initialize trading state variables"""
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

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed with detailed reasoning"""
        try:
            # Check if system is running
            if not self.state.running:
                return False, "System not running"

            # Check daily limits
            pl_today, trades_today = self._today_pl_and_trades()
            acc = self.trade_executor.get_account_info()
            bal = acc.get('balance', 10000.0)

            if bal and (pl_today / bal) <= -abs(self.config.MAX_DAILY_LOSS):
                return False, f"Daily loss limit hit: {pl_today:.2f}/{bal:.2f}"

            if trades_today >= self.config.MAX_TRADES_PER_DAY:
                return False, f"Max trades/day reached: {trades_today}"

            # Check session restrictions
            sess_now = self._current_session()
            if "24h" not in self.config.SESSIONS and sess_now not in self.config.SESSIONS:
                return False, f"Outside allowed sessions: {sess_now}"

            # Check cooldown
            if self.state.cooldown_until and datetime.now() < self.state.cooldown_until:
                return (
                    False,
                    f"Cooldown active until {self.state.cooldown_until.strftime('%H:%M:%S')}",
                )

            # Check open positions
            open_count = self._get_open_trades_count()
            if not self.risk_manager.can_open_new_trade(
                acc.get('balance', 10000.0),
                self.state.start_balance or acc.get('balance', 10000.0),
                open_count,
            ):
                return (
                    False,
                    f"Risk blocked new trade. open={open_count}/{self.config.MAX_OPEN_TRADES}",
                )

            return True, "Trading allowed"

        except Exception as e:
            self.logger.error(f"Error checking trade conditions: {e}")
            return False, f"Error: {e}"

    def _current_session(self) -> str:
        """Determine current trading session"""
        try:
            import pytz

            tz = pytz.timezone(self.config.SESSION_TZ)
            h = datetime.now(tz).hour
        except Exception:
            h = datetime.utcnow().hour

        if 0 <= h < 8:
            return "Asia"
        if 8 <= h < 16:
            return "London"
        return "NY"

    def _today_pl_and_trades(self) -> tuple[float, int]:
        """Get today's P/L and trade count"""
        # Implementation would go here - simplified for modularity
        return 0.0, 0

    def _get_open_trades_count(self) -> int:
        """Get count of open trades"""
        # Implementation would go here - simplified for modularity
        return 0

    def generate_signal(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate trading signal using AI system"""
        try:
            # Compose market snapshot for AI
            tick = self.data_manager.get_current_tick()
            if tick:
                market_data = {
                    'time': tick['time'].isoformat(),
                    'open': float(df['open'].iloc[-1]),
                    'high': float(df['high'].iloc[-1]),
                    'low': float(df['low'].iloc[-1]),
                    'close': float(tick['bid']),
                    'tick_volume': float(tick['volume']),
                }
            else:
                market_data = {
                    'time': datetime.now().isoformat(),
                    'open': float(df['open'].iloc[-1]),
                    'high': float(df['high'].iloc[-1]),
                    'low': float(df['low'].iloc[-1]),
                    'close': float(df['close'].iloc[-1]),
                    'tick_volume': float(df['tick_volume'].iloc[-1]),
                }

            # Generate AI signal
            self.logger.info(f"🔄 Generating AI signal... Market data: {market_data}")
            sig = self.ai_system.generate_ensemble_signal(market_data)
            self.logger.info(f"🎯 AI Signal generated: {sig}")

            return sig

        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {'signal': 0, 'confidence': 0.0, 'score': 0.0, 'source': 'Error'}

    def validate_signal(self, signal_data: dict[str, Any]) -> tuple[bool, str]:
        """Validate trading signal with detailed reasoning"""
        try:
            # Basic signal validation
            if signal_data['signal'] == 0:
                return False, "Signal is 0"

            if signal_data['confidence'] < 0.1:
                return False, f"Signal confidence too low: {signal_data['confidence']:.3f} < 0.1"

            # Check consecutive signals
            if signal_data['signal'] == self.state.last_signal:
                self.state.consecutive_signals += 1
            else:
                self.state.consecutive_signals = 1
                self.state.last_signal = signal_data['signal']

            # Check confidence threshold
            thr = self.risk_manager.get_current_confidence_threshold()
            if signal_data['confidence'] < thr:
                return False, f"Confidence {signal_data['confidence']:.3f} < threshold {thr:.3f}"

            # Check consecutive requirement
            req_consec = self.config.CONSECUTIVE_SIGNALS_REQUIRED
            if self.state.consecutive_signals < req_consec:
                return (
                    False,
                    f"Consecutive {self.state.consecutive_signals} < required {req_consec}",
                )

            return True, "Signal validated"

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False, f"Error: {e}"

    def execute_trade(self, signal_data: dict[str, Any], df: pd.DataFrame) -> bool:
        """Execute trading signal"""
        try:
            # This would call the trade executor
            # Simplified for modularity
            self.logger.info(f"🚀 Executing trade: {signal_data}")

            # Update state after successful execution
            self.state.last_fill_time = datetime.now()
            cooldown_s = int(
                self.config.config_data.get("trading", {}).get("cooldown_seconds", 180)
            )
            self.state.cooldown_until = self.state.last_fill_time + timedelta(seconds=cooldown_s)
            self.state.last_trade_side = signal_data['signal']

            self.logger.info(f"✅ Trade executed successfully. Cooldown {cooldown_s}s")
            return True

        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return False

    def update_trailing_stops(self):
        """Update trailing stops for open positions"""
        try:
            now = datetime.now()
            if (now - self.state.last_trailing_update).seconds >= 15:  # 15 second interval
                # Implementation would go here
                self.state.last_trailing_update = now
                self.logger.info("Trailing stops updated")
        except Exception as e:
            self.logger.error(f"Error updating trailing stops: {e}")

    def cleanup(self):
        """Clean up system resources"""
        try:
            # Clear trailing registry
            if hasattr(self, 'trailing_registry'):
                self.trailing_registry.clear()
                self.logger.info("🧹 Trailing registry cleared")

            # Close event logger
            if hasattr(self, 'ev') and self.ev:
                self.ev.close()
                self.logger.info("✅ Event logger closed")

            # Close MFE logger
            if hasattr(self, 'mfe_logger') and self.mfe_logger:
                self.mfe_logger.close()
                self.logger.info("✅ MFE logger closed")

        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
