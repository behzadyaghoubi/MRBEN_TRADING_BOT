#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Configuration Management Module
Extracted and modularized from live_trader_clean.py
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class Credentials:
    """MT5 credentials configuration"""
    login: Optional[str] = None
    password: Optional[str] = None
    server: Optional[str] = None


@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    symbol: str = "XAUUSD.PRO"
    timeframe: int = 15
    bars: int = 500
    magic_number: int = 20250721
    sessions: List[str] = field(default_factory=lambda: ["London", "NY"])
    max_spread_points: int = 200
    use_risk_based_volume: bool = True
    fixed_volume: float = 0.01
    sleep_seconds: int = 12
    retry_delay: int = 5
    consecutive_signals_required: int = 1
    lstm_timesteps: int = 50
    cooldown_seconds: int = 180


@dataclass
class RiskConfig:
    """Risk management configuration"""
    base_risk: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 2.0
    max_open_trades: int = 3
    max_daily_loss: float = 0.02
    max_trades_per_day: int = 10
    sl_atr_multiplier: float = 1.6
    tp_atr_multiplier: float = 2.2


@dataclass
class LoggingConfig:
    """Logging configuration"""
    enabled: bool = True
    level: str = "INFO"
    log_file: str = "logs/trading_bot.log"
    trade_log_path: str = "data/trade_log_gold.csv"


@dataclass
class SessionConfig:
    """Session management configuration"""
    timezone: str = "Etc/UTC"


@dataclass
class AdvancedConfig:
    """Advanced trading features configuration"""
    swing_lookback: int = 12
    dynamic_spread_atr_frac: float = 0.10
    deviation_multiplier: float = 1.5
    inbar_eval_seconds: int = 10
    inbar_min_conf: float = 0.66
    inbar_min_score: float = 0.12
    inbar_min_struct_buffer_atr: float = 0.8
    startup_warmup_seconds: int = 90
    startup_min_conf: float = 0.62
    startup_min_score: float = 0.10
    reentry_window_seconds: int = 90


@dataclass
class ExecutionConfig:
    """Trade execution configuration"""
    spread_eps: float = 0.02
    use_spread_ma: bool = True
    spread_ma_window: int = 5
    spread_hysteresis_factor: float = 1.05


@dataclass
class TPPolicyConfig:
    """Take profit policy configuration"""
    split: bool = True
    tp1_r: float = 0.8
    tp2_r: float = 1.5
    tp1_share: float = 0.5
    breakeven_after_tp1: bool = True


@dataclass
class ConformalConfig:
    """Conformal prediction configuration"""
    enabled: bool = True
    soft_gate: bool = True
    emergency_bypass: bool = False
    min_p: float = 0.10
    hard_floor: float = 0.05
    penalty_small: float = 0.05
    penalty_big: float = 0.10
    extra_consecutive: int = 1
    treat_zero_as_block: bool = True
    max_conf_bump_floor: float = 0.05
    extra_consec_floor: int = 2
    cap_final_thr: float = 0.90


class MT5Config:
    """Reads config.json and exposes fields used by the system."""
    
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
            self._load_config()
            self._validate_config()
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
            
            # Parse into structured config objects
            self._parse_credentials(raw)
            self._parse_trading(raw)
            self._parse_risk(raw)
            self._parse_logging(raw)
            self._parse_session(raw)
            self._parse_advanced(raw)
            self._parse_execution(raw)
            self._parse_tp_policy(raw)
            self._parse_conformal(raw)
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def _parse_credentials(self, raw: Dict[str, Any]):
        """Parse credentials configuration"""
        creds = raw.get("credentials", {})
        self.credentials = Credentials(
            login=creds.get("login"),
            password=os.getenv("MT5_PASSWORD", creds.get("password")),
            server=creds.get("server")
        )
        
        # Set flags
        flags = raw.get("flags", {})
        self.DEMO_MODE = bool(flags.get("demo_mode", True))
        
        # Strict credential check only if DEMO_MODE is False
        if not self.DEMO_MODE and not (self.credentials.login and self.credentials.password and self.credentials.server):
            raise RuntimeError("❌ MT5 credentials missing. Provide via config.json under 'credentials'.")

    def _parse_trading(self, raw: Dict[str, Any]):
        """Parse trading configuration"""
        trading = raw.get("trading", {})
        self.trading = TradingConfig(
            symbol=trading.get("symbol", "XAUUSD.PRO"),
            timeframe=int(trading.get("timeframe", 15)),
            bars=int(trading.get("bars", 500)),
            magic_number=int(trading.get("magic_number", 20250721)),
            sessions=trading.get("sessions", ["London", "NY"]),
            max_spread_points=int(trading.get("max_spread_points", 200)),
            use_risk_based_volume=bool(trading.get("use_risk_based_volume", True)),
            fixed_volume=float(trading.get("fixed_volume", 0.01)),
            sleep_seconds=int(trading.get("sleep_seconds", 12)),
            retry_delay=int(trading.get("retry_delay", 5)),
            consecutive_signals_required=int(trading.get("consecutive_signals_required", 1)),
            lstm_timesteps=int(trading.get("lstm_timesteps", 50)),
            cooldown_seconds=int(trading.get("cooldown_seconds", 180))
        )
        
        # Set individual attributes for backward compatibility
        self.SYMBOL = self.trading.symbol
        self.TIMEFRAME_MIN = self.trading.timeframe
        self.BARS = self.trading.bars
        self.MAGIC = self.trading.magic_number
        self.SESSIONS = self.trading.sessions
        self.MAX_SPREAD_POINTS = self.trading.max_spread_points
        self.USE_RISK_BASED_VOLUME = self.trading.use_risk_based_volume
        self.FIXED_VOLUME = self.trading.fixed_volume
        self.SLEEP_SECONDS = self.trading.sleep_seconds
        self.RETRY_DELAY = self.trading.retry_delay
        self.CONSECUTIVE_SIGNALS_REQUIRED = self.trading.consecutive_signals_required
        self.LSTM_TIMESTEPS = self.trading.lstm_timesteps
        self.COOLDOWN_SECONDS = self.trading.cooldown_seconds

    def _parse_risk(self, raw: Dict[str, Any]):
        """Parse risk management configuration"""
        risk = raw.get("risk", {})
        self.risk = RiskConfig(
            base_risk=float(risk.get("base_risk", 0.01)),
            min_lot=float(risk.get("min_lot", 0.01)),
            max_lot=float(risk.get("max_lot", 2.0)),
            max_open_trades=int(risk.get("max_open_trades", 3)),
            max_daily_loss=float(risk.get("max_daily_loss", 0.02)),
            max_trades_per_day=int(risk.get("max_trades_per_day", 10)),
            sl_atr_multiplier=float(risk.get("sl_atr_multiplier", 1.6)),
            tp_atr_multiplier=float(risk.get("tp_atr_multiplier", 2.2))
        )
        
        # Set individual attributes for backward compatibility
        self.BASE_RISK = self.risk.base_risk
        self.MIN_LOT = self.risk.min_lot
        self.MAX_LOT = self.risk.max_lot
        self.MAX_OPEN_TRADES = self.risk.max_open_trades
        self.MAX_DAILY_LOSS = self.risk.max_daily_loss
        self.MAX_TRADES_PER_DAY = self.risk.max_trades_per_day

    def _parse_logging(self, raw: Dict[str, Any]):
        """Parse logging configuration"""
        logging_cfg = raw.get("logging", {})
        self.logging = LoggingConfig(
            enabled=bool(logging_cfg.get("enabled", True)),
            level=logging_cfg.get("level", "INFO"),
            log_file=logging_cfg.get("log_file", "logs/trading_bot.log"),
            trade_log_path=logging_cfg.get("trade_log_path", "data/trade_log_gold.csv")
        )
        
        # Set individual attributes for backward compatibility
        self.LOG_ENABLED = self.logging.enabled
        self.LOG_LEVEL = self.logging.level
        self.LOG_FILE = self.logging.log_file
        self.TRADE_LOG_PATH = self.logging.trade_log_path

    def _parse_session(self, raw: Dict[str, Any]):
        """Parse session configuration"""
        session_cfg = raw.get("session", {})
        self.session = SessionConfig(
            timezone=session_cfg.get("timezone", "Etc/UTC")
        )
        
        # Set individual attributes for backward compatibility
        self.SESSION_TZ = self.session.timezone

    def _parse_advanced(self, raw: Dict[str, Any]):
        """Parse advanced configuration"""
        advanced = raw.get("advanced", {})
        self.advanced = AdvancedConfig(
            swing_lookback=int(advanced.get("swing_lookback", 12)),
            dynamic_spread_atr_frac=float(advanced.get("dynamic_spread_atr_frac", 0.10)),
            deviation_multiplier=float(advanced.get("deviation_multiplier", 1.5)),
            inbar_eval_seconds=int(advanced.get("inbar_eval_seconds", 10)),
            inbar_min_conf=float(advanced.get("inbar_min_conf", 0.66)),
            inbar_min_score=float(advanced.get("inbar_min_score", 0.12)),
            inbar_min_struct_buffer_atr=float(advanced.get("inbar_min_struct_buffer_atr", 0.8)),
            startup_warmup_seconds=int(advanced.get("startup_warmup_seconds", 90)),
            startup_min_conf=float(advanced.get("startup_min_conf", 0.62)),
            startup_min_score=float(advanced.get("startup_min_score", 0.10)),
            reentry_window_seconds=int(advanced.get("reentry_window_seconds", 90))
        )

    def _parse_execution(self, raw: Dict[str, Any]):
        """Parse execution configuration"""
        execution_cfg = raw.get("execution", {})
        self.execution = ExecutionConfig(
            spread_eps=float(execution_cfg.get("spread_eps", 0.02)),
            use_spread_ma=bool(execution_cfg.get("use_spread_ma", True)),
            spread_ma_window=int(execution_cfg.get("spread_ma_window", 5)),
            spread_hysteresis_factor=float(execution_cfg.get("spread_hysteresis_factor", 1.05))
        )

    def _parse_tp_policy(self, raw: Dict[str, Any]):
        """Parse take profit policy configuration"""
        tp_policy_cfg = raw.get("tp_policy", {})
        self.tp_policy = TPPolicyConfig(
            split=tp_policy_cfg.get("split", True),
            tp1_r=float(tp_policy_cfg.get("tp1_r", 0.8)),
            tp2_r=float(tp_policy_cfg.get("tp2_r", 1.5)),
            tp1_share=float(tp_policy_cfg.get("tp1_share", 0.5)),
            breakeven_after_tp1=bool(tp_policy_cfg.get("breakeven_after_tp1", True))
        )

    def _parse_conformal(self, raw: Dict[str, Any]):
        """Parse conformal prediction configuration"""
        conformal_cfg = raw.get("conformal", {})
        self.conformal = ConformalConfig(
            enabled=bool(conformal_cfg.get("enabled", True)),
            soft_gate=bool(conformal_cfg.get("soft_gate", True)),
            emergency_bypass=bool(conformal_cfg.get("emergency_bypass", False)),
            min_p=float(conformal_cfg.get("min_p", 0.10)),
            hard_floor=float(conformal_cfg.get("hard_floor", 0.05)),
            penalty_small=float(conformal_cfg.get("penalty_small", 0.05)),
            penalty_big=float(conformal_cfg.get("penalty_big", 0.10)),
            extra_consecutive=int(conformal_cfg.get("extra_consecutive", 1)),
            treat_zero_as_block=bool(conformal_cfg.get("treat_zero_as_block", True)),
            max_conf_bump_floor=float(conformal_cfg.get("max_conf_bump_floor", 0.05)),
            extra_consec_floor=int(conformal_cfg.get("extra_consec_floor", 2)),
            cap_final_thr=float(conformal_cfg.get("cap_final_thr", 0.90))
        )

    def _validate_config(self):
        """Validate configuration values"""
        # Validate trading parameters
        if self.trading.timeframe <= 0:
            raise ValueError("Timeframe must be positive")
        if self.trading.bars <= 0:
            raise ValueError("Bars must be positive")
        if self.trading.magic_number <= 0:
            raise ValueError("Magic number must be positive")
        
        # Validate risk parameters
        if self.risk.base_risk <= 0 or self.risk.base_risk > 1:
            raise ValueError("Base risk must be between 0 and 1")
        if self.risk.min_lot <= 0:
            raise ValueError("Min lot must be positive")
        if self.risk.max_lot <= 0:
            raise ValueError("Max lot must be positive")
        if self.risk.min_lot > self.risk.max_lot:
            raise ValueError("Min lot cannot be greater than max lot")
        
        # Validate logging
        if self.logging.level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log level")
        
        self.logger.info("✅ Configuration validation passed")

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
            "conformal_enabled": self.conformal.enabled
        }

    def reload_config(self):
        """Reload configuration from file"""
        self.logger.info("🔄 Reloading configuration...")
        self._load_config()
        self._validate_config()
        self.logger.info("✅ Configuration reloaded successfully")
