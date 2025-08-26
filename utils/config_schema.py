# utils/config_schema.py

from pydantic import BaseModel, validator


class Credentials(BaseModel):
    login: int | None = None
    password: str | None = None
    server: str | None = None


class Flags(BaseModel):
    demo_mode: bool = True


class Trading(BaseModel):
    symbol: str = "XAUUSD.PRO"
    timeframe: int = 15
    bars: int = 500
    magic_number: int = 20250721
    sessions: list[str] = ["London", "NY"]  # یا ["24h"]
    max_spread_points: int = 200
    use_risk_based_volume: bool = True
    fixed_volume: float = 0.01
    sleep_seconds: int = 12
    retry_delay: int = 5
    consecutive_signals_required: int = 1
    lstm_timesteps: int = 50
    cooldown_seconds: int = 180
    max_risk_volume_cap: float = 2.0


class Risk(BaseModel):
    base_risk: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 2.0
    max_open_trades: int = 3
    max_daily_loss: float = 0.02
    sl_atr_multiplier: float = 1.6
    tp_atr_multiplier: float = 2.2


class LoggingCfg(BaseModel):
    enabled: bool = True
    level: str = "INFO"
    log_file: str = "logs/trading_bot.log"
    trade_log_path: str = "data/trade_log_gold.csv"


class SessionCfg(BaseModel):
    timezone: str = "Etc/UTC"


class TPPolicy(BaseModel):
    split: bool = True
    tp1_r: float = 0.8
    tp2_r: float = 1.5
    tp1_share: float = 0.5
    breakeven_after_tp1: bool = True


class Advanced(BaseModel):
    swing_lookback: int = 12
    dynamic_spread_atr_frac: float = 0.10
    deviation_multiplier: float = 1.5


class Conformal(BaseModel):
    enabled: bool = True
    soft_gate: bool = True
    emergency_bypass: bool = False
    min_p: float = 0.10
    hard_floor: float = 0.05
    penalty_small: float = 0.05
    penalty_big: float = 0.10
    extra_consecutive: int = 1
    max_conf_bump_floor: float = 0.05
    extra_consec_floor: int = 2
    cap_final_thr: float = 0.90
    # افزوده برای فاز 6
    p_floor_session: dict = {"Asia": 0.06, "London": 0.08, "NY": 0.10}


class RootConfig(BaseModel):
    credentials: Credentials = Credentials()
    flags: Flags = Flags()
    trading: Trading = Trading()
    risk: Risk = Risk()
    logging: LoggingCfg = LoggingCfg()
    session: SessionCfg = SessionCfg()
    models: dict = {}
    tp_policy: TPPolicy = TPPolicy()
    advanced: Advanced = Advanced()
    conformal: Conformal = Conformal()

    @validator("logging")
    def ensure_paths(cls, v: LoggingCfg):
        assert v.log_file, "log_file required"
        assert v.trade_log_path, "trade_log_path required"
        return v
