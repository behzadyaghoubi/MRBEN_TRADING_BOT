#!/usr/bin/env python3
"""
MR BEN - Configuration Management
Pydantic models with environment variable overrides
"""

from __future__ import annotations

import json
import os
from typing import Literal

import yaml
from pydantic import BaseModel

# ==== Strategy Configuration ====


class StrategyLookbackCfg(BaseModel):
    fast: int = 20
    slow: int = 50


class StrategyPriceActionCfg(BaseModel):
    enabled: bool = True
    min_score: float = 0.55
    patterns: list[str] = ["engulf", "pin", "inside", "sweep"]


class StrategyMLFilterCfg(BaseModel):
    enabled: bool = True
    model_path: str = "models/ml_filter_v1.onnx"
    min_proba: float = 0.58


class StrategyLSTMCfg(BaseModel):
    enabled: bool = True
    model_path: str = "models/lstm_dir_v1.onnx"
    agree_min: float = 0.55


class StrategyCfg(BaseModel):
    core: str = "sma_cross"
    lookback: StrategyLookbackCfg = StrategyLookbackCfg()
    price_action: StrategyPriceActionCfg = StrategyPriceActionCfg()
    ml_filter: StrategyMLFilterCfg = StrategyMLFilterCfg()
    lstm_filter: StrategyLSTMCfg = StrategyLSTMCfg()


# ==== Confidence Configuration ====


class ConfidenceDynCfg(BaseModel):
    regime: dict[str, float] = {"low_vol": 1.10, "normal": 1.00, "high_vol": 0.85}
    session: dict[str, float] = {"asia": 0.90, "london": 1.05, "ny": 1.00}
    drawdown: dict[str, float] = {"calm": 1.00, "mild_dd": 0.90, "deep_dd": 0.80}


class ConfidenceThrCfg(BaseModel):
    min: float = 0.60
    max: float = 0.75


class ConfidenceCfg(BaseModel):
    base: float = 0.70
    dynamic: ConfidenceDynCfg = ConfidenceDynCfg()
    threshold: ConfidenceThrCfg = ConfidenceThrCfg()


# ==== ATR Configuration ====


class ATRCfg(BaseModel):
    sl_mult: float = 1.6
    tp_r: dict[str, float] = {"tp1": 0.8, "tp2": 1.5}


# ==== Risk Management Configuration ====


class RiskGatesCfg(BaseModel):
    spread_max_pts: int = 180
    exposure_max_positions: int = 2
    daily_loss_pct: float = 2.0
    consecutive_min: int = 2
    cooldown_sec: int = 90


class RiskManagementCfg(BaseModel):
    base_r_pct: float = 0.15
    min_lot: float = 0.10
    max_lot: float = 1.00
    gates: RiskGatesCfg = RiskGatesCfg()


# ==== Session Configuration ====


class SessionCfg(BaseModel):
    enabled: bool = True


# ==== Position Management Configuration ====


class PositionManagementCfg(BaseModel):
    tp_split_enabled: bool = True
    breakeven_enabled: bool = True
    trailing_stop_enabled: bool = True
    trailing_update_interval: int = 15


# ==== Order Management Configuration ====


class OrderManagementCfg(BaseModel):
    default_filling_mode: str = "ioc"
    max_slippage_pts: int = 50
    retry_count: int = 3
    retry_delay: float = 1.0


# ==== Metrics Configuration ====


class MetricsCfg(BaseModel):
    port: int = 8765
    enabled: bool = True


# ==== Emergency Stop Configuration ====


class EmergencyStopCfg(BaseModel):
    enabled: bool = True
    halt_file_path: str = "halt.flag"
    check_interval: float = 1.0
    auto_recovery: bool = False
    recovery_delay: float = 300.0
    monitoring_enabled: bool = True
    log_all_checks: bool = False


# ==== Logging Configuration ====


class LoggingCfg(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# ==== Root Configuration ====


class RootCfg(BaseModel):
    strategy: StrategyCfg
    confidence: ConfidenceCfg
    atr: ATRCfg
    risk_management: RiskManagementCfg
    session: SessionCfg
    position_management: PositionManagementCfg
    order_management: OrderManagementCfg
    metrics: MetricsCfg
    emergency_stop: EmergencyStopCfg
    logging: LoggingCfg


# ==== ENV Override helper ====


def _smart_cast(val: str):
    """Smart cast string values to appropriate types"""
    v = val.strip()
    vl = v.lower()
    if vl in ("true", "false"):
        return vl == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except:
        return v


def apply_env_overrides(cfg_dict: dict, prefix: str = "MRBEN__") -> dict:
    """Apply environment variable overrides to configuration"""
    # Example: MRBEN__CONFIDENCE__THRESHOLD__MIN=0.62
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        path = k[len(prefix) :].lower().split("__")
        ref = cfg_dict
        for p in path[:-1]:
            ref = ref.setdefault(p, {})
        ref[path[-1]] = _smart_cast(v)
    return cfg_dict


# ==== Loader ====


def load_config(path: str = "config/config.yaml", env_prefix: str = "MRBEN__") -> RootCfg:
    """Load configuration from YAML file with environment variable overrides"""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    merged = apply_env_overrides(raw, env_prefix)
    cfg = RootCfg.model_validate(merged)
    return cfg


# Optional: pretty JSON for logging
def cfg_to_json(cfg: RootCfg) -> str:
    """Convert configuration to JSON string for logging"""
    return json.dumps(cfg.model_dump(), ensure_ascii=False, separators=(",", ":"))
