#!/usr/bin/env python3
"""
MR BEN Live Trading System v4.0 - Clean & Patched
- ATR-based TP/SL
- Trailing stop
- Adaptive confidence thresholds
- Risk management + daily limits
- Real-time MT5 integration
- FIX: use trade_stops_level (not stops_level)
- FIX: spread check in points + proper rounding of SL/TP respecting min distance
"""

import csv  # برای کنترل quoting در CSV
import json
import logging
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, getcontext
from typing import Any

# --- END PATCH ---
import numpy as np
import pandas as pd

# --- BEGIN PATCH [import EventLogger] ---
from telemetry.event_logger import EventLogger

getcontext().prec = 12

# Import utility modules
# from utils.regime import detect_regime  # Currently unused

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
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.models import load_model

    # from ai_filter import AISignalFilter  # Currently unused
    AI_AVAILABLE = True
except Exception as e:
    print(f"⚠️ AI stack partial/unavailable: {e}")
    AI_AVAILABLE = False

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


def enforce_min_distance_and_round(
    symbol: str, entry: float, sl: float, tp: float, is_buy: bool
) -> tuple[float, float]:
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


def is_spread_ok(symbol: str, max_spread_points: int) -> tuple[bool, float, float]:
    """
    Check if current spread (in points) is below threshold.
    Returns: (ok, spread_points, threshold_points)
    """
    try:
        if not MT5_AVAILABLE:
            return True, 0.0, float(max_spread_points)
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if not info or not tick:
            return True, 0.0, float(max_spread_points)
        point = info.point or 0.01
        spread_price = tick.ask - tick.bid
        spread_points = spread_price / point
        return (spread_points <= max_spread_points), float(spread_points), float(max_spread_points)
    except Exception:
        return True, 0.0, float(max_spread_points)


# def _linmap(x, a, b, c, d):  # Currently unused
#     """نگاشت خطی x∈[a,b] → [c,d] با کلیپ"""
#     if a == b:
#         return (c + d) / 2
#     t = max(0.0, min(1.0, (x - a) / (b - a)))
#     return c + t * (d - c)


def _rolling_atr(df: pd.DataFrame, period: int = 14) -> float:
    """محاسبه ATR با fallback و محافظ NaN"""
    try:
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = np.maximum(hl, np.maximum(hc, lc))
        atr = tr.rolling(period, min_periods=period).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else 0.0
    except Exception:
        return 0.5


def _swing_extrema(df: pd.DataFrame, bars: int = 10) -> tuple[float, float]:
    """محاسبه swing high/low در N کندل اخیر"""
    try:
        bars = max(2, min(bars, len(df)))
        lo = float(df['low'].iloc[-bars:].min())
        hi = float(df['high'].iloc[-bars:].max())
        return lo, hi
    except Exception:
        return 0.0, 0.0


# --- Soft Conformal Gate helper (cap penalties, keep consec>=1, high-conf override) ---
def _apply_soft_gate(
    p_value: float,
    base_thr: float,
    base_consec: int,
    *,
    low_cut: float = 0.20,
    med_cut: float = 0.10,
    very_low_cut: float = 0.05,
    bump_low: float = 0.00,
    bump_med: float = 0.02,
    bump_vlow: float = 0.03,
    max_conf_bump: float = 0.03,
    add_consec_med: int = 0,
    add_consec_vlow: int = 1,
    high_conf_override_margin: float = 0.02,
):
    """
    Returns: (adj_thr, req_consec, override_margin)
    - آستانهٔ اطمینان را حداکثر 0.03 زیاد می‌کند.
    - حداقل consecutive همیشه 1 می‌ماند.
    - اگر conf >= adj_thr + 0.02 باشد، نیاز به consecutive اضافی حذف می‌شود.
    """
    # پیش‌فرض‌ها
    conf_bump = bump_low
    add_consec = 0

    if p_value >= low_cut:
        conf_bump = 0.0
        add_consec = 0
    elif med_cut <= p_value < low_cut:
        conf_bump = bump_med
        add_consec = add_consec_med
    elif very_low_cut <= p_value < med_cut:
        conf_bump = bump_vlow
        add_consec = add_consec_vlow
    else:
        # خیلی پایین: کمی سخت‌گیرتر، اما capped
        conf_bump = min(max_conf_bump, bump_vlow + 0.005)
        add_consec = add_consec_vlow

    conf_bump = min(conf_bump, max_conf_bump)
    adj_thr = base_thr + conf_bump
    req_consec = max(1, base_consec + add_consec)  # هرگز > 1 نشود اگر base_consec=1

    return adj_thr, req_consec, high_conf_override_margin


# --- Position Management Helpers ---
def _get_open_positions(symbol: str, magic: int):
    """Fetch open positions for this symbol & magic as dict[ticket]=position."""
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        return {}
    by_ticket = {}
    for p in pos:
        # توجه: بعضی بروکرها فیلد magic را صفر می‌دهند؛ اگر magic شما روی trade request ست می‌شود، فیلتر نگه دارید
        if getattr(p, 'magic', 0) == magic or magic is None:
            by_ticket[p.ticket] = p
    return by_ticket


def _modify_position_sltp(
    position_ticket: int,
    symbol: str,
    new_sl: float = None,
    new_tp: float = None,
    magic: int = None,
    deviation: int = 20,
):
    """Safely modify SL/TP of an open position by ticket."""
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": position_ticket,
        "deviation": deviation,
    }
    if new_sl is not None:
        request["sl"] = float(new_sl)
    if new_tp is not None:
        request["tp"] = float(new_tp)
    if magic is not None:
        request["magic"] = int(magic)

    result = mt5.order_send(request)
    return result


def _prune_trailing_registry(trailing_registry: dict, open_pos_dict: dict, logger):
    """Remove stale tickets not present in current open positions."""
    stale = [t for t in list(trailing_registry.keys()) if t not in open_pos_dict]
    for t in stale:
        trailing_registry.pop(t, None)
    if stale:
        logger.info("🧹 Trailing prune: removed %d stale tickets: %s", len(stale), stale)


def _count_open_positions(symbol: str, magic: int) -> int:
    return len(_get_open_positions(symbol, magic))


# -----------------------------
# Config
# -----------------------------


class MT5Config:
    """Reads config.json and exposes fields used by the system."""

    def __init__(self):
        config_path = 'config.json'
        try:
            with open(config_path, encoding='utf-8') as f:
                self.config_data = json.load(f)
        except Exception:
            self.config_data = {}

        # Credentials
        creds = self.config_data.get("credentials", {})
        self.LOGIN = creds.get("login")
        self.PASSWORD = os.getenv("MT5_PASSWORD", creds.get("password"))
        self.SERVER = creds.get("server")

        # Flags
        flags = self.config_data.get("flags", {})
        self.DEMO_MODE = bool(flags.get("demo_mode", True))

        # Trading
        trading = self.config_data.get("trading", {})
        self.SYMBOL = trading.get("symbol", "XAUUSD.PRO")
        self.TIMEFRAME_MIN = int(trading.get("timeframe", 15))
        self.BARS = int(trading.get("bars", 500))
        self.MAGIC = int(trading.get("magic_number", 20250721))
        self.SESSIONS: list[str] = trading.get("sessions", ["London", "NY"])
        self.MAX_SPREAD_POINTS = int(trading.get("max_spread_points", 200))
        self.USE_RISK_BASED_VOLUME = bool(trading.get("use_risk_based_volume", True))
        self.FIXED_VOLUME = float(trading.get("fixed_volume", 0.01))
        self.SLEEP_SECONDS = int(trading.get("sleep_seconds", 12))
        self.RETRY_DELAY = int(trading.get("retry_delay", 5))
        self.CONSECUTIVE_SIGNALS_REQUIRED = int(trading.get("consecutive_signals_required", 1))
        self.LSTM_TIMESTEPS = int(trading.get("lstm_timesteps", 50))
        self.COOLDOWN_SECONDS = int(trading.get("cooldown_seconds", 180))

        # Risk
        risk = self.config_data.get("risk", {})
        self.BASE_RISK = float(risk.get("base_risk", 0.01))
        self.MIN_LOT = float(risk.get("min_lot", 0.01))
        self.MAX_LOT = float(risk.get("max_lot", 2.0))
        self.MAX_OPEN_TRADES = int(risk.get("max_open_trades", 3))
        self.MAX_DAILY_LOSS = float(risk.get("max_daily_loss", 0.02))
        self.MAX_TRADES_PER_DAY = int(risk.get("max_trades_per_day", 10))

        # Logging
        logging_cfg = self.config_data.get("logging", {})
        self.LOG_ENABLED = bool(logging_cfg.get("enabled", True))
        self.LOG_LEVEL = logging_cfg.get("level", "INFO")
        self.LOG_FILE = logging_cfg.get("log_file", "logs/trading_bot.log")
        self.TRADE_LOG_PATH = logging_cfg.get("trade_log_path", "data/trade_log_gold.csv")

        # Models (optional flags)
        self.MODELS = self.config_data.get("models", {})

        # --- BEGIN PATCH [Session TZ config] ---
        session_cfg = self.config_data.get("session", {})
        self.SESSION_TZ = session_cfg.get("timezone", "Etc/UTC")
        # --- END PATCH ---

        # Strict credential check only if DEMO_MODE is False
        if not self.DEMO_MODE and not (self.LOGIN and self.PASSWORD and self.SERVER):
            raise RuntimeError(
                "❌ MT5 credentials missing. Provide via config.json under 'credentials'."
            )


# -----------------------------
# Data Manager
# -----------------------------


class MT5DataManager:
    def __init__(self, symbol: str, timeframe_min: int):
        self.symbol = symbol
        self.timeframe_min = timeframe_min
        self.mt5_connected = False
        self.current_data: pd.DataFrame | None = None

        if MT5_AVAILABLE:
            self._initialize_mt5()
        else:
            print("⚠️ MT5 not available, synthetic data mode.")

    def _initialize_mt5(self) -> bool:
        try:
            if not mt5.initialize():
                print(f"❌ MT5 initialize failed: {mt5.last_error()}")
                return False
            print(f"✅ MT5 initialized for data: {self.symbol}")
            self.mt5_connected = True
            return True
        except Exception as e:
            print(f"❌ MT5 init error: {e}")
            return False

    def _tf_to_mt5(self, minutes: int):
        # Map minute timeframe to MT5 enum
        m = minutes
        if m == 1:
            return mt5.TIMEFRAME_M1
        if m == 2:
            return mt5.TIMEFRAME_M2
        if m == 3:
            return mt5.TIMEFRAME_M3
        if m == 4:
            return mt5.TIMEFRAME_M4
        if m == 5:
            return mt5.TIMEFRAME_M5
        if m == 10:
            return mt5.TIMEFRAME_M10
        if m == 15:
            return mt5.TIMEFRAME_M15
        if m == 30:
            return mt5.TIMEFRAME_M30
        if m == 60:
            return mt5.TIMEFRAME_H1
        return mt5.TIMEFRAME_M15

    def get_latest_data(self, bars: int = 500) -> pd.DataFrame:
        if not MT5_AVAILABLE or not self.mt5_connected:
            return self._get_synthetic_data(bars)

        try:
            tf = self._tf_to_mt5(self.timeframe_min)
            rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)
            if rates is None or len(rates) == 0:
                print("⚠️ MT5 rates empty, using synthetic data.")
                return self._get_synthetic_data(bars)

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = self._indicators(df)
            self.current_data = df
            return df
        except Exception as e:
            print(f"❌ get_latest_data error: {e}")
            return self._get_synthetic_data(bars)

    def get_current_tick(self) -> dict[str, Any] | None:
        if not MT5_AVAILABLE:
            return None
        try:
            t = mt5.symbol_info_tick(self.symbol)
            if not t:
                return None

            # make tick time timezone-aware using trader config timezone
            try:
                import pytz

                with open('config.json', encoding='utf-8') as f:
                    cfg = json.load(f)
                tzname = cfg.get("session", {}).get("timezone", "Etc/UTC")
                tz = pytz.timezone(tzname)
            except Exception:
                tz = None

            tick_dt_utc = datetime.fromtimestamp(t.time, tz=UTC)  # epoch is UTC
            tick_time = tick_dt_utc.astimezone(tz) if tz else tick_dt_utc

            # staleness check in the same tz basis (seconds don't depend on tz)
            if (datetime.now(UTC) - tick_dt_utc).total_seconds() > 300:
                return None

            return {'bid': t.bid, 'ask': t.ask, 'time': tick_time, 'volume': t.volume}
        except Exception:
            return None

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df['close'].diff().fillna(0)
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(14).mean()

        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        return df.ffill().bfill()

    def _get_synthetic_data(self, bars: int) -> pd.DataFrame:
        base = 3300.0
        data = []
        now = datetime.now()
        for i in range(bars):
            op = base + np.random.uniform(-10, 10)
            cl = op + np.random.uniform(-5, 5)
            hi = max(op, cl) + np.random.uniform(0, 3)
            lo = min(op, cl) - np.random.uniform(0, 3)
            vol = np.random.randint(100, 1000)
            data.append(
                {
                    'time': now - timedelta(minutes=i * self.timeframe_min),
                    'open': op,
                    'high': hi,
                    'low': lo,
                    'close': cl,
                    'tick_volume': vol,
                }
            )
        df = pd.DataFrame(data)[::-1].reset_index(drop=True)
        return self._indicators(df)

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# -----------------------------
# AI System (Ensemble / Simple)
# -----------------------------


class MRBENAdvancedAISystem:
    def __init__(self):
        self.logger = logging.getLogger('MRBENAdvancedAI')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # جلوگیری از دوباره‌چاپ شدن
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
            self.logger.addHandler(ch)

        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.ensemble_weights = [0.4, 0.3, 0.3]  # LSTM, ML, TECH

    def load_models(self):
        try:
            if os.path.exists('models/advanced_lstm_model.h5'):
                self.models['lstm'] = load_model('models/advanced_lstm_model.h5')
                self.logger.info("LSTM model loaded")
            if os.path.exists('models/quick_fix_ml_filter.joblib'):
                model_data = joblib.load('models/quick_fix_ml_filter.joblib')
                self.models['ml_filter'] = model_data['model']
                self.scalers['ml_filter'] = model_data['scaler']
                # اگر روی sklearn>=1.2 هستیم، خروجی اسکیلر را DataFrame کن تا نام ستون‌ها حفظ شود
                try:
                    if hasattr(self.scalers['ml_filter'], "set_output"):
                        self.scalers['ml_filter'].set_output(transform="pandas")
                except Exception:
                    pass
                self.logger.info("ML filter model loaded")
            self.logger.info(f"Loaded models: {list(self.models.keys())}")
        except Exception as e:
            self.logger.error(f"Model load error: {e}")

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = np.maximum(hl, np.maximum(hc, lc))
        return tr.rolling(period).mean()

    def generate_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        session = pd.Series(index=df.index, dtype='object')
        h = df['hour']
        session[(h >= 0) & (h < 8)] = 'Asia'
        session[(h >= 8) & (h < 16)] = 'London'
        session[(h >= 16) & (h < 24)] = 'NY'
        df['session'] = session

        if 'session_encoder' not in self.label_encoders:
            self.label_encoders['session_encoder'] = LabelEncoder()
            df['session_encoded'] = self.label_encoders['session_encoder'].fit_transform(
                df['session']
            )
        else:
            try:
                df['session_encoded'] = self.label_encoders['session_encoder'].transform(
                    df['session']
                )
            except Exception:
                # اگر transform شکست خورد، دوباره fit نکن - فقط fallback
                df['session_encoded'] = 0  # fallback به Asia session

        if 'atr' not in df.columns:
            df['atr'] = self._calculate_atr(df)
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        return df.ffill().bfill()

    def _tech_pred(self, df: pd.DataFrame) -> dict[str, Any]:
        try:
            if len(df) == 1:
                ch = (df['close'].iloc[0] - df['open'].iloc[0]) / df['open'].iloc[0]
                if ch > 0.0005:
                    return {'signal': 1, 'confidence': 0.7, 'score': 0.3}
                if ch < -0.0005:
                    return {'signal': -1, 'confidence': 0.7, 'score': -0.3}
                return {'signal': 0, 'confidence': 0.6, 'score': 0.0}
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macds = df['macd_signal'].iloc[-1]
            score = 0.0
            conf = 0.5
            if rsi < 35:
                score += 0.4
                conf += 0.2
            elif rsi < 45:
                score += 0.2
            elif rsi > 65:
                score -= 0.4
                conf += 0.2
            elif rsi > 55:
                score -= 0.2
            ms = macd - macds
            if ms > 0.03:
                score += 0.3
                conf += 0.1
            elif ms < -0.03:
                score -= 0.3
                conf += 0.1
            signal = 1 if score > 0.05 else (-1 if score < -0.05 else 0)
            return {'signal': signal, 'confidence': min(conf, 0.9), 'score': score}
        except Exception:
            return {'signal': 0, 'confidence': 0.5, 'score': 0.0}

    def _lstm_simple(self, df: pd.DataFrame) -> dict[str, Any]:
        try:
            if len(df) == 1:
                ch = (df['close'].iloc[0] - df['open'].iloc[0]) / df['open'].iloc[0]
                if ch > 0.0002:
                    return {'signal': 1, 'confidence': 0.75, 'score': 0.3}
                if ch < -0.0002:
                    return {'signal': -1, 'confidence': 0.75, 'score': -0.3}
                return {'signal': 0, 'confidence': 0.65, 'score': 0.0}
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            if rsi < 40:
                return {'signal': 1, 'confidence': 0.8, 'score': 0.4}
            if rsi > 60:
                return {'signal': -1, 'confidence': 0.8, 'score': -0.4}
            return {'signal': 0, 'confidence': 0.6, 'score': 0.0}
        except Exception:
            return {'signal': 0, 'confidence': 0.5, 'score': 0.0}

    def _ml_pred(self, df: pd.DataFrame) -> dict[str, Any]:
        try:
            if 'ml_filter' not in self.models:
                return {'signal': 0, 'confidence': 0.5, 'score': 0.0}
            # Expect scaler feature_names_in_
            scaler = self.scalers.get('ml_filter', None)
            if not scaler or not hasattr(scaler, 'feature_names_in_'):
                return {'signal': 0, 'confidence': 0.5, 'score': 0.0}
            # حفظ دقیــق ترتیب و کامل بودن فیچرها
            feats = [c for c in list(scaler.feature_names_in_) if c in df.columns]
            if len(feats) != len(getattr(scaler, 'feature_names_in_', [])):
                return {'signal': 0, 'confidence': 0.5, 'score': 0.0}
            X = df[feats].astype(float)  # نوع عددی قطعی
            X = X[scaler.feature_names_in_]  # ترتیب دقیق
            Xs = scaler.transform(X)  # با set_output، DataFrame می‌ماند و هشدار نمی‌دهد
            proba = self.models['ml_filter'].predict_proba(Xs)[0]
            cls = np.argmax(proba)
            conf = float(np.max(proba))
            signal = 1 if cls == 1 else -1
            return {'signal': signal, 'confidence': conf, 'score': float(proba[1] - proba[0])}
        except Exception:
            return {'signal': 0, 'confidence': 0.5, 'score': 0.0}

    def generate_ensemble_signal(self, market_data: dict[str, Any]) -> dict[str, Any]:
        try:
            self.logger.info(f"🔍 AI System: Processing market data: {market_data}")
            df = pd.DataFrame([market_data])
            df = self.generate_meta_features(df)
            self.logger.info(f"🔍 AI System: Meta features generated, shape: {df.shape}")

            tech = self._tech_pred(df)
            self.logger.info(f"🔍 AI System: Technical prediction: {tech}")

            lstm = self._lstm_simple(df) if 'lstm' in self.models else tech
            self.logger.info(f"🔍 AI System: LSTM prediction: {lstm}")

            ml = self._ml_pred(df) if 'ml_filter' in self.models else tech
            self.logger.info(f"🔍 AI System: ML prediction: {ml}")

            score = (
                lstm['score'] * self.ensemble_weights[0]
                + ml['score'] * self.ensemble_weights[1]
                + tech['score'] * self.ensemble_weights[2]
            )
            if score > 0.05:
                sig = 1
            elif score < -0.05:
                sig = -1
            else:
                sig = 0

            conf = (
                lstm['confidence'] * self.ensemble_weights[0]
                + ml['confidence'] * self.ensemble_weights[1]
                + tech['confidence'] * self.ensemble_weights[2]
            )
            self.logger.info(
                f"Ensemble: LSTM({lstm['signal']},{lstm['score']:.3f}) ML({ml['signal']},{ml['score']:.3f}) TECH({tech['signal']},{tech['score']:.3f}) => Final({sig},{score:.3f})"
            )
            return {
                'signal': sig,
                'confidence': float(conf),
                'score': float(score),
                'source': 'Advanced AI Ensemble',
            }
        except Exception as e:
            self.logger.error(f"Ensemble error: {e}")
            return {'signal': 0, 'confidence': 0.5, 'score': 0.0, 'source': 'Error'}

    def ensemble_proba_win(self, market_df: pd.DataFrame) -> float:
        """
        Map ensemble score/confidence to a pseudo-probability of success.
        We'll squash score via sigmoid and blend with confidence.
        """
        try:
            # استفاده از همان generate_meta_features
            df = self.generate_meta_features(market_df.copy())
            tech = self._tech_pred(df)
            lstm = self._lstm_simple(df) if 'lstm' in self.models else tech
            ml = self._ml_pred(df) if 'ml_filter' in self.models else tech
            score = (
                lstm['score'] * self.ensemble_weights[0]
                + ml['score'] * self.ensemble_weights[1]
                + tech['score'] * self.ensemble_weights[2]
            )
            conf = (
                lstm['confidence'] * self.ensemble_weights[0]
                + ml['confidence'] * self.ensemble_weights[1]
                + tech['confidence'] * self.ensemble_weights[2]
            )
            # نگاشت به احتمال
            import math

            p = 1.0 / (1.0 + math.exp(-4.0 * score))  # سیگموید با شیب 4
            # blend با اعتماد
            p = 0.5 * p + 0.5 * max(0.0, min(1.0, conf))
            return float(max(0.0, min(1.0, p)))
        except Exception:
            return 0.5


# -----------------------------
# Risk Manager
# -----------------------------


class EnhancedRiskManager:
    def __init__(
        self,
        base_risk=0.01,
        min_lot=0.01,
        max_lot=2.0,
        max_open_trades=3,
        max_drawdown=0.10,
        atr_period=14,
        sl_atr_multiplier=2.0,
        tp_atr_multiplier=4.0,
        trailing_atr_multiplier=1.5,
        base_confidence_threshold=0.35,
        adaptive_confidence=True,
        performance_window=20,
        confidence_adjustment_factor=0.1,
        tf_minutes=15,
    ):
        self.base_risk = base_risk
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.max_open_trades = max_open_trades
        self.max_drawdown = max_drawdown
        self.atr_period = atr_period
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.base_confidence_threshold = base_confidence_threshold
        self.adaptive_confidence = adaptive_confidence
        self.performance_window = performance_window
        self.confidence_adjustment_factor = confidence_adjustment_factor
        self.tf_minutes = tf_minutes

        # ATR cache fields
        self._atr_cache = {"value": None, "ts": 0.0}

        self.recent_performances: list[float] = []
        self.current_confidence_threshold = base_confidence_threshold
        self.trailing_stops: dict[int, dict[str, Any]] = {}

        self.logger = logging.getLogger("EnhancedRiskManager")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # جلوگیری از دوباره‌چاپ شدن
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
            self.logger.addHandler(h)

    def get_atr(self, symbol: str) -> float | None:
        """Get current ATR value for symbol using configured timeframe (with small cache)."""
        try:
            now = time.time()
            if self._atr_cache["value"] is not None and (now - self._atr_cache["ts"]) < 5.0:
                return self._atr_cache["value"]

            if not MT5_AVAILABLE:
                return None
            tf_map = {
                1: mt5.TIMEFRAME_M1,
                5: mt5.TIMEFRAME_M5,
                15: mt5.TIMEFRAME_M15,
                30: mt5.TIMEFRAME_M30,
                60: mt5.TIMEFRAME_H1,
                240: mt5.TIMEFRAME_H4,
                1440: mt5.TIMEFRAME_D1,
            }
            tf = tf_map.get(self.tf_minutes, mt5.TIMEFRAME_M15)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, self.atr_period + 1)
            if rates is None or len(rates) < self.atr_period:
                return None
            df = pd.DataFrame(rates)
            hl = df['high'] - df['low']
            hc = (df['high'] - df['close'].shift()).abs()
            lc = (df['low'] - df['close'].shift()).abs()
            tr = np.maximum(hl, np.maximum(hc, lc))
            val = float(tr.rolling(self.atr_period, min_periods=self.atr_period).mean().iloc[-1])
            if pd.isna(val):
                return None
            self._atr_cache.update({"value": val, "ts": now})
            return val
        except Exception as e:
            self.logger.error(f"ATR error: {e}")
            return None

    def calculate_dynamic_sl_tp(
        self, symbol: str, entry_price: float, signal: str
    ) -> tuple[float, float]:
        atr = self.get_atr(symbol)
        if atr is None:
            # fallback distances (instrument dependent; conservative)
            dist_sl = 0.5
            dist_tp = 1.0
        else:
            dist_sl = atr * self.sl_atr_multiplier
            dist_tp = atr * self.tp_atr_multiplier

        if signal == "BUY":
            sl = entry_price - dist_sl
            tp = entry_price + dist_tp
        else:
            sl = entry_price + dist_sl
            tp = entry_price - dist_tp

        self.logger.info(f"Dynamic SL/TP: SL={sl:.2f} TP={tp:.2f} (ATR={atr})")
        return sl, tp

    def calculate_lot_size(
        self, balance: float, risk_per_trade: float, sl_distance: float, symbol: str
    ) -> float:
        try:
            if not MT5_AVAILABLE:
                return max(self.min_lot, min(0.1, self.max_lot))
            info = mt5.symbol_info(symbol)
            if info is None or sl_distance <= 0:
                return self.min_lot
            ticks = sl_distance / (info.trade_tick_size or info.point or 0.01)
            if ticks <= 0:
                return self.min_lot
            risk_amount = balance * risk_per_trade
            vpt = info.trade_tick_value or 1.0
            raw = risk_amount / (ticks * vpt)

            # --- BEGIN PATCH [Decimal volume alignment in calculate_lot_size] ---
            from decimal import Decimal

            step_dec = Decimal(str(info.volume_step or 0.01))
            lot_dec = Decimal(str(max(self.min_lot, min(raw, self.max_lot))))
            lot_adj = (lot_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec
            lot = float(max(info.volume_min, min(float(lot_adj), info.volume_max)))
            # --- END PATCH ---

            return float(lot)
        except Exception:
            return self.min_lot

    def can_open_new_trade(
        self, current_balance: float, start_balance: float, open_trades_count: int
    ) -> bool:
        if open_trades_count >= self.max_open_trades:
            return False
        if start_balance > 0:
            dd = (start_balance - current_balance) / start_balance
            if dd > self.max_drawdown:
                return False
        return True

    def add_trailing_stop(self, ticket: int, entry_price: float, initial_sl: float, is_buy: bool):
        self.trailing_stops[ticket] = {
            'entry_price': entry_price,
            'current_sl': initial_sl,
            'highest_price': entry_price if is_buy else float('-inf'),
            'lowest_price': entry_price if not is_buy else float('inf'),
            'is_buy': is_buy,
        }

    def remove_trailing_stop(self, ticket: int):
        if ticket in self.trailing_stops:
            del self.trailing_stops[ticket]

    def update_trailing_stops(self, symbol: str) -> list[dict[str, float]]:
        mods = []
        atr = self.get_atr(symbol)
        if atr is None:
            return mods
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return mods
            price = (tick.bid + tick.ask) / 2.0
            for ticket, st in self.trailing_stops.items():
                if st['is_buy']:
                    if price > st['highest_price']:
                        st['highest_price'] = price
                    new_sl = st['highest_price'] - (atr * self.trailing_atr_multiplier)
                    if new_sl > st['current_sl']:
                        st['current_sl'] = new_sl
                        mods.append({'ticket': ticket, 'new_sl': new_sl})
                else:
                    if price < st['lowest_price']:
                        st['lowest_price'] = price
                    new_sl = st['lowest_price'] + (atr * self.trailing_atr_multiplier)
                    if new_sl < st['current_sl']:
                        st['current_sl'] = new_sl
                        mods.append({'ticket': ticket, 'new_sl': new_sl})
        except Exception:
            return mods

        # --- BEGIN PATCH [Enhanced trailing logging] ---
        if mods:
            self.logger.info(f"⛓️ Trailing candidates: {len(mods)}")
        # --- END PATCH ---

        return mods

    def get_current_confidence_threshold(self) -> float:
        return float(self.current_confidence_threshold)

    def update_performance_from_history(self, symbol: str):
        if not MT5_AVAILABLE:
            return
        try:
            start = datetime.combine(datetime.now().date(), datetime.min.time())
            deals = mt5.history_deals_get(start, datetime.now())
            if not deals:
                return
            symbol_deals = [
                d for d in deals if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT
            ]
            if not symbol_deals:
                return
            last = max(symbol_deals, key=lambda d: d.time)
            self.recent_performances.append(last.profit)
            if len(self.recent_performances) > self.performance_window:
                self.recent_performances.pop(0)
            min_closed = max(10, self.performance_window // 2)  # حداقل ۱۰ تا
            if self.adaptive_confidence and len(self.recent_performances) >= min_closed:
                window = self.recent_performances[-self.performance_window :]
                wins = sum(1 for r in window if r > 0)
                wr = wins / len(window)
                prev = self.current_confidence_threshold
                if wr > 0.6:
                    self.current_confidence_threshold = max(
                        self.base_confidence_threshold - self.confidence_adjustment_factor,
                        self.current_confidence_threshold - self.confidence_adjustment_factor,
                    )
                elif wr < 0.4:
                    self.current_confidence_threshold = min(
                        self.base_confidence_threshold + self.confidence_adjustment_factor,
                        self.current_confidence_threshold + self.confidence_adjustment_factor,
                    )
                if prev != self.current_confidence_threshold:
                    self.logger.info(
                        f"Adaptive conf: {prev:.2f} -> {self.current_confidence_threshold:.2f} (winrate={wr:.2f})"
                    )
        except Exception as e:
            self.logger.error(f"Perf update error: {e}")


# -----------------------------
# Trade Executor
# -----------------------------


class EnhancedTradeExecutor:
    def __init__(self, risk_manager: EnhancedRiskManager):
        self.risk_manager = risk_manager
        self.trailing_meta = risk_manager.trailing_stops  # اشاره به همان دیکشنری
        self.logger = logging.getLogger("EnhancedTradeExecutor")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # جلوگیری از دوباره‌چاپ شدن
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
            self.logger.addHandler(h)

    def modify_stop_loss(
        self, symbol: str, position_ticket: int, new_sl: float, current_tp: float | None = None
    ) -> bool:
        try:
            if not MT5_AVAILABLE:
                return False

            # پوزیشن را بیاور
            pos = None
            positions = mt5.positions_get(symbol=symbol) or []
            for p in positions:
                if p.ticket == position_ticket:
                    pos = p
                    break
            if not pos:
                self.logger.error(f"Position {position_ticket} not found to modify SLTP")
                return False

            is_buy = pos.type == 0
            entry_like = pos.price_open  # به‌جای تیک لحظه‌ای، قیمت ورود خود پوزیشن

            # اگر current_tp ندادیم، TP فعلی پوزیشن را نگه‌دار
            tp_to_send = current_tp if current_tp is not None else float(pos.tp or 0.0)

            # رعایت حداقل فاصله و رُند کردن
            adj_sl, adj_tp = enforce_min_distance_and_round(
                symbol, entry_like, new_sl, tp_to_send if tp_to_send else new_sl, is_buy
            )

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": int(position_ticket),
                "symbol": symbol,
                "sl": float(adj_sl),
                "tp": (
                    float(tp_to_send) if tp_to_send else 0.0
                ),  # اگر واقعاً قصد حذف TP داری، صِفر بده
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(
                    f"SLTP modify failed pos {position_ticket}: {getattr(result,'retcode',None)} {getattr(result,'comment',None)}"
                )
                return False

            # متادیتای تریلینگ را به‌روز کن
            st = self.trailing_meta.get(position_ticket)
            if st:
                st['current_sl'] = float(adj_sl)

            self.logger.info(
                f"SL modified for position {position_ticket}: SL={adj_sl:.2f} (kept TP={tp_to_send:.2f})"
            )
            return True
        except Exception as e:
            self.logger.error(f"Modify SL error: {e}")
            return False

    def update_trailing_stops(self, symbol: str) -> int:
        mods = self.risk_manager.update_trailing_stops(symbol)
        cnt = 0
        for m in mods:
            if self.modify_stop_loss(symbol, m['ticket'], m['new_sl'], None):
                cnt += 1
                self.logger.info(f"↗ Trailing move | pos={m['ticket']} new_sl={m['new_sl']:.2f}")
        if cnt > 0:
            self.logger.info(f"✅ Trailing updated: {cnt} position(s)")
        return cnt

    def get_account_info(self) -> dict:
        try:
            if not MT5_AVAILABLE:
                return {
                    'balance': 10000.0,
                    'equity': 10000.0,
                    'margin': 0.0,
                    'free_margin': 10000.0,
                }
            a = mt5.account_info()
            if not a:
                return {
                    'balance': 10000.0,
                    'equity': 10000.0,
                    'margin': 0.0,
                    'free_margin': 10000.0,
                }
            return {
                'balance': a.balance,
                'equity': a.equity,
                'margin': a.margin,
                'free_margin': a.margin_free,
            }
        except Exception:
            return {'balance': 10000.0, 'equity': 10000.0, 'margin': 0.0, 'free_margin': 10000.0}


# -----------------------------
# Live Trader
# -----------------------------


# -----------------------------
# --- BEGIN PATCH [safe trade log append] ---
def _append_trade_log_csv(row: dict, path: str):
    """
    Append a single trade log row to CSV safely:
    - ثابت نگه‌داشتن ترتیب ستون‌ها
    - quoting برای جلوگیری از به‌هم‌ریختگی به‌خاطر comma
    - خط پایان سازگار با ویندوز/لینوکس
    - عدم کرش اگر خطا شد
    """
    import os

    import pandas as pd

    cols = [
        "run_id",
        "timestamp",
        "symbol",
        "action",
        "entry_price",
        "sl_price",
        "tp_price",
        "volume",
        "confidence",
        "source",
        "atr",
        "vol_ratio",
        "sl_mult",
        "tp_mult",
        "spread_price",
        "spread_frac",
        "R_multiple",
        "R_multiple_post_round",
        "fallback_used",
        "tp_policy",
        "mt5_order_id",
        "mt5_retcode",
        "mt5_executed",
        "mt5_error",
    ]
    # تضمین وجود کلیدها
    for c in cols:
        row.setdefault(c, None)

    df = pd.DataFrame([{c: row.get(c, None) for c in cols}])

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header_needed = not os.path.exists(path) or os.path.getsize(path) == 0
        # استفاده از quoting امن و lineterminator
        df.to_csv(
            path,
            mode="a",
            header=header_needed,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
    except Exception as e:
        logging.getLogger("TradeLog").error(f"trade_log append failed: {e}")


# --- END PATCH ---


class MT5LiveTrader:
    def __init__(self):
        self.config = MT5Config()

        # --- BEGIN PATCH [Logger before file handler] ---
        self.logger = logging.getLogger(self.__class__.__name__)
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
        # --- END PATCH ---

        # Risk Manager
        self.risk_manager = EnhancedRiskManager(
            base_risk=self.config.BASE_RISK,
            min_lot=self.config.MIN_LOT,
            max_lot=self.config.MAX_LOT,
            max_open_trades=self.config.MAX_OPEN_TRADES,
            base_confidence_threshold=0.35,
            tf_minutes=self.config.TIMEFRAME_MIN,
        )

        # Trade Executor
        self.trade_executor = EnhancedTradeExecutor(self.risk_manager)

        # Data Manager
        self.data_manager = MT5DataManager(self.config.SYMBOL, self.config.TIMEFRAME_MIN)

        # MT5 connect for trading
        self.mt5_connected = self._initialize_mt5_for_trading()

        # AI system
        self.ai_system = MRBENAdvancedAISystem()
        self.logger.info("🤖 Initializing AI System...")
        self.ai_system.load_models()
        self.logger.info(
            f"✅ AI System loaded. Available models: {list(self.ai_system.models.keys())}"
        )

        # Conformal Gate
        from utils.conformal import ConformalGate

        self.conformal = None
        try:
            self.conformal = ConformalGate("models/meta_filter.joblib", "models/conformal.json")
            self.logger.info("✅ Conformal gate loaded.")
        except Exception as e:
            self.logger.warning(f"⚠️ Conformal not available: {e}")

        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        # --- BEGIN PATCH [init EventLogger] ---
        os.makedirs("data", exist_ok=True)
        self.ev = EventLogger(
            path="data/events.jsonl", run_id=self.run_id, symbol=self.config.SYMBOL
        )
        # --- END PATCH ---

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

        # Spread threshold
        self.max_spread_points = int(self.config.MAX_SPREAD_POINTS)

        # پارامترهای داینامیک SL/TP (بهینه‌سازی شده برای TP های قابل لمس)
        rconf = self.config.config_data.get("risk", {})
        self.sl_mult_base = float(rconf.get("sl_atr_multiplier", 1.6))
        self.tp_mult_base = min(float(rconf.get("tp_atr_multiplier", 3.0)), 2.2)  # کاهش سقف TP
        self.conf_min = float(self.risk_manager.base_confidence_threshold)  # مثلا 0.35
        self.conf_max = 0.90
        self.k_sl = 0.35  # ضریب تعدیل SL بر اساس اعتماد
        self.k_tp = 0.20  # کاهش از 0.50 به 0.20 - کمتر دور کردن TP با اعتماد بالا
        self.swing_lookback = int(
            self.config.config_data.get("advanced", {}).get("swing_lookback", 12)
        )
        self.max_spread_atr_frac = float(
            self.config.config_data.get("advanced", {}).get("dynamic_spread_atr_frac", 0.10)
        )  # سخت‌گیری بیشتر

        # TP Policy برای تقسیم حجم و BE
        tp_policy_cfg = self.config.config_data.get("tp_policy", {})
        self.tp_policy = {
            "split": tp_policy_cfg.get("split", True),
            "tp1_r": tp_policy_cfg.get("tp1_r", 0.8),
            "tp2_r": tp_policy_cfg.get("tp2_r", 1.5),
            "tp1_share": tp_policy_cfg.get("tp1_share", 0.5),  # 50%
            "breakeven_after_tp1": tp_policy_cfg.get("breakeven_after_tp1", True),
        }
        self.min_R_after_round = 1.2  # نرم‌تر شده

    # --- BEGIN PATCH [Preflight check method] ---
    def _preflight_check(self) -> bool:
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

    # --- END PATCH ---

    def _r_to_price(self, entry: float, sl: float, signal: int, R: float) -> float:
        """محاسبه قیمت TP بر اساس R ratio"""
        risk = abs(entry - sl)
        if signal == 1:  # BUY
            return entry + (R * risk)
        else:  # SELL
            return entry - (R * risk)

    def _find_position_id_for_order(
        self, order_id: int, retries: int = 5, sleep_s: float = 0.5
    ) -> int | None:
        if not MT5_AVAILABLE:
            return None
        for _ in range(retries):
            end = datetime.now()
            start = end - timedelta(minutes=10)
            deals = mt5.history_deals_get(start, end) or []
            for d in deals:
                if d.order == order_id and getattr(d, 'position_id', 0):
                    return int(d.position_id)
            time.sleep(sleep_s)
        return None

    def _mfe_quantile_from_history(self, q: float = 0.6, lookback: int = 400) -> float | None:
        """تحلیل MFE (Maximum Favorable Excursion) از تاریخچه معاملات"""
        try:
            path = self.config.TRADE_LOG_PATH
            if not os.path.exists(path):
                return None
            import csv

            df = pd.read_csv(
                path,
                engine="python",  # رِیدر انعطاف‌پذیرتر
                on_bad_lines="skip",  # خطوط خراب را رد کن
                quoting=csv.QUOTE_MINIMAL,  # با نحوه‌ی نوشتن سازگار
            )
            df = df.tail(lookback)
            # اگر ستون MFE نداریم، از R_multiple_post_round به عنوان تخمین استفاده می‌کنیم
            col = 'R_multiple_post_round' if 'R_multiple_post_round' in df.columns else None
            if col and len(df) > 0:
                try:
                    s = df[col].dropna()
                    s = s[s > 0.0]  # فقط معاملات مثبت
                    if len(s) >= 50:
                        quantile_val = float(np.quantile(s.values, q))
                        self.logger.info(
                            f"MFE Q{q*100:.0f}% از {len(s)} معامله: {quantile_val:.2f}R"
                        )
                        return quantile_val
                except Exception as e:
                    self.logger.warning(f"MFE quantile calculation error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"MFE analysis error: {e}")
            return None

    def _send_order(
        self,
        order_type,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        volume: float,
        deviation: float,
    ) -> object:
        """Helper برای ارسال سفارش"""
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.SYMBOL,
            "volume": float(volume),
            "type": order_type,
            "price": float(entry_price),
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": int(deviation),
            "magic": int(self.config.MAGIC),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return mt5.order_send(req)

    def _initialize_mt5_for_trading(self) -> bool:
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

    def _validate_system(self) -> bool:
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

    def _bootstrap_trailing(self):
        """Register existing open positions into trailing on startup/restart."""
        try:
            if not MT5_AVAILABLE:
                return
            open_pos = _get_open_positions(self.config.SYMBOL, self.config.MAGIC)
            self.trailing_registry = {}
            for ticket, p in open_pos.items():
                self.trailing_registry[ticket] = {
                    "dir": "sell" if p.type == mt5.POSITION_TYPE_SELL else "buy",
                    "entry": float(p.price_open),
                    "sl": float(p.sl) if p.sl else None,
                    "tp": float(p.tp) if p.tp else None,
                }
            if open_pos:
                self.logger.info(
                    f"🔗 Trailing bootstrap: registered {len(open_pos)} open positions."
                )
        except Exception as e:
            self.logger.error(f"Trailing bootstrap error: {e}")

    def _update_trailing(self):
        """Update trailing stops for open positions using live position data."""
        try:
            if not MT5_AVAILABLE:
                return

            # 1) همگام‌سازی با پوزیشن‌های زنده
            open_pos = _get_open_positions(self.config.SYMBOL, self.config.MAGIC)
            self.logger.debug("Trailing live positions: %s", list(open_pos.keys()))
            _prune_trailing_registry(self.trailing_registry, open_pos, self.logger)

            if not open_pos:
                return

            # 2) کاندیدها، فقط از پوزیشن‌های باز فعلی
            candidates = []
            for ticket, p in open_pos.items():
                # نمونه: فقط پوزیشن‌های در سود را تریل کنیم
                profit_points = (
                    (p.price_open - p.price_current)
                    if p.type == mt5.POSITION_TYPE_SELL
                    else (p.price_current - p.price_open)
                )
                if profit_points > 0:
                    candidates.append((ticket, p, profit_points))

            self.logger.info("⛓️ Trailing candidates (live): %d", len(candidates))

            if not candidates:
                return

            # 3) منطق تریل (مثال بر پایه ATR یا فاصله ثابت)
            # فرض: از self.atr_points و self.trailing_step استفاده می‌کنید؛ اگر ندارید، یک حداقل ثابت بگذارید
            step = getattr(self, "trailing_step", 0.5)  # مثال: 0.5 دلار/اونس
            for ticket, p, prof in candidates:
                if p.type == mt5.POSITION_TYPE_SELL:
                    # قیمت به نفع حرکت کرده؛ SL را پایین بیاوریم اما نه پایین‌تر از قیمت فعلی/قوانین بروکر
                    desired_sl = min(p.price_open - step, p.price_current - step)  # محافظه‌کار
                    if p.sl is None or desired_sl < p.sl - 1e-6:
                        res = _modify_position_sltp(
                            ticket,
                            self.config.SYMBOL,
                            new_sl=desired_sl,
                            new_tp=p.tp,
                            magic=self.config.MAGIC,
                        )
                        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                            self.logger.warning(
                                "⚠️ Trailing SLTP modify failed for pos=%s ret=%s",
                                ticket,
                                getattr(res, "retcode", None),
                            )
                            # اگر تغییر ممکن نیست، حذف از رجیستری تا اسپم نشود
                            self.trailing_registry.pop(ticket, None)
                        else:
                            self.trailing_registry.setdefault(ticket, {})
                            self.trailing_registry[ticket]["sl"] = desired_sl
                            self.logger.info(
                                "↘ Trailing move | pos=%s new_sl=%.2f", ticket, desired_sl
                            )
                else:
                    # BUY
                    desired_sl = max(p.price_open + step, p.price_current + step)
                    if p.sl is None or desired_sl > p.sl + 1e-6:
                        res = _modify_position_sltp(
                            ticket,
                            self.config.SYMBOL,
                            new_sl=desired_sl,
                            new_tp=p.tp,
                            magic=self.config.MAGIC,
                        )
                        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                            self.logger.warning(
                                "⚠️ Trailing SLTP modify failed for pos=%s ret=%s",
                                ticket,
                                getattr(res, "retcode", None),
                            )
                            self.trailing_registry.pop(ticket, None)
                        else:
                            self.trailing_registry.setdefault(ticket, {})
                            self.trailing_registry[ticket]["sl"] = desired_sl
                            self.logger.info(
                                "↗ Trailing move | pos=%s new_sl=%.2f", ticket, desired_sl
                            )

        except Exception as e:
            self.logger.error(f"Trailing update error: {e}")

    def start(self):
        print("🎯 MR BEN Live Trading System v4.0")
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
        self.logger.info("🚀 MR BEN Live Trading System v4.0 Starting...")
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

        # --- BEGIN PATCH [emit start event] ---
        self.ev.emit(
            "bot_start",
            account_login=int(acc.login) if self.mt5_connected and acc else None,
            balance=float(acc.balance) if self.mt5_connected and acc else None,
            equity=float(acc.equity) if self.mt5_connected and acc else None,
            config={"tf": self.config.TIMEFRAME_MIN, "max_open": self.config.MAX_OPEN_TRADES},
        )
        # --- END PATCH ---

        # --- BEGIN PATCH [Call preflight before loop] ---
        if not self._preflight_check():
            self.logger.error("Preflight failed. Not starting loop.")
            return
        # --- END PATCH ---

        # Kill-switch اکوییتی
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
        self.logger.info("✅ Trading loop started")

        # --- BEGIN PATCH [Bootstrap trailing for existing positions] ---
        self._bootstrap_trailing()
        # --- END PATCH ---

    def stop(self):
        self.running = False
        try:
            if hasattr(self, 'data_manager') and self.data_manager:
                self.data_manager.shutdown()
        except Exception:
            pass

        # Clean MT5 shutdown
        try:
            if MT5_AVAILABLE and mt5.initialize():
                mt5.shutdown()
        except Exception:
            pass

        self.logger.info("✅ Stopped.")

        # --- BEGIN PATCH [close EventLogger] ---
        try:
            if hasattr(self, "ev") and self.ev:
                self.ev.close()
        except Exception:
            pass
        # --- END PATCH ---

    def _current_session(self) -> str:
        # --- BEGIN PATCH [Session TZ use] ---
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
        # --- END PATCH ---

    def _today_pl_and_trades(self) -> tuple[float, int]:
        if not MT5_AVAILABLE:
            return 0.0, 0
        try:
            start = datetime.combine(datetime.now().date(), datetime.min.time())
            deals = mt5.history_deals_get(start, datetime.now())
            if not deals:
                return 0.0, 0
            pl = sum(d.profit for d in deals if d.symbol == self.config.SYMBOL)
            cnt = sum(
                1 for d in deals if d.symbol == self.config.SYMBOL and d.entry == mt5.DEAL_ENTRY_IN
            )
            return float(pl), int(cnt)
        except Exception:
            return 0.0, 0

    def _get_open_trades_count(self) -> int:
        if not self.mt5_connected:
            return 0
        try:
            return _count_open_positions(self.config.SYMBOL, self.config.MAGIC)
        except Exception:
            return 0

    def _calculate_dynamic_sl_tp(
        self, df: pd.DataFrame, entry: float, signal: int, confidence: float, symbol: str
    ) -> tuple[float, float, dict]:
        """
        محاسبه SL/TP داینامیک با توجه به ATR، اعتماد سیگنال، رژیم نوسان، اسپرد نسبی و ساختار.
        خروجی: (sl_price, tp_price, meta_info)
        """
        try:
            # ATR و رژیم نوسان
            atr_now = _rolling_atr(df, period=self.risk_manager.atr_period)
            atr_now = max(atr_now, 1e-6)  # محافظ NaN
            atr_ref = pd.Series(df.get('atr', np.nan)).rolling(90).mean().iloc[-1]
            atr_ref = float(atr_ref) if pd.notna(atr_ref) else atr_now
            vol_ratio = (atr_now / atr_ref) if atr_ref > 0 else 1.0

            # ضرایب پایه
            sl_mult = self.sl_mult_base
            tp_mult = self.tp_mult_base

            # تعدیل با اعتماد سیگنال
            conf = max(self.conf_min, min(self.conf_max, float(confidence)))
            sl_mult *= 1.0 - self.k_sl * (conf - self.conf_min) / (
                self.conf_max - self.conf_min + 1e-9
            )
            tp_mult *= 1.0 + self.k_tp * (conf - self.conf_min) / (
                self.conf_max - self.conf_min + 1e-9
            )

            # تعدیل بر اساس رژیم نوسان
            if vol_ratio > 1.5:
                sl_mult *= 1.10
                tp_mult *= 1.10
            elif vol_ratio < 0.8:
                sl_mult *= 0.90
                tp_mult *= 0.90

            # اسپرد نسبی
            spread_price = 0.0
            if MT5_AVAILABLE:
                info = mt5.symbol_info(symbol)
                tick = mt5.symbol_info_tick(symbol)
                if info and tick:
                    spread_price = tick.ask - tick.bid
            spread_frac = (spread_price / atr_now) if atr_now > 0 else 0.0
            if spread_frac > self.max_spread_atr_frac:
                # سخت‌گیری: ریسک را کمی گشاد کن، سود را کمی نزدیک‌تر بگذار
                sl_mult *= 1.10
                tp_mult *= 0.85

            # فاصله‌ها
            sl_dist = max(atr_now * sl_mult, 0.1)
            tp_dist = max(atr_now * tp_mult, 0.2)

            # ساختار (swing)
            point = mt5.symbol_info(symbol).point if MT5_AVAILABLE else 0.01
            lo_sw, hi_sw = _swing_extrema(df, self.swing_lookback)

            # لاگ swing_unreliable اگر دیتای خیلی کم بود
            swing_unreliable = len(df) < self.swing_lookback * 2
            if signal == 1:  # BUY
                sl_struct = lo_sw - 3.0 * point
                sl_price = max(entry - sl_dist, sl_struct)  # SL نزدیک‌ترِ امن، نه دورتر از ساختار
                tp_price = entry + tp_dist
            else:  # SELL
                sl_struct = hi_sw + 3.0 * point
                sl_price = min(entry + sl_dist, sl_struct)  # SL نزدیک‌ترِ امن، نه دورتر از ساختار
                tp_price = entry - tp_dist

            # تعدیل بر اساس MFE واقعی معاملات قبلی
            mfe_q = self._mfe_quantile_from_history(q=0.6, lookback=400)
            if mfe_q and mfe_q > 0.4:
                # اگر MFE 60% معاملات ~0.9R بوده، TP را زیاد نکش!
                desired_R = min(max(1.2, mfe_q * 1.1), 1.6)
                tp_dist_adj = abs(entry - sl_price) * desired_R
                if signal == 1:
                    tp_price = entry + tp_dist_adj
                else:
                    tp_price = entry - tp_dist_adj

            # حداقل R (نرم‌تر شده برای TP های قابل لمس)
            r = abs((tp_price - entry) / (entry - sl_price)) if (entry - sl_price) != 0 else 0.0
            if r < 1.2:
                # سعی کن TP را به 1.2R برسونی
                desired_tp = entry + np.sign(tp_price - entry) * 1.2 * abs(entry - sl_price)
                tp_price = desired_tp
                r = 1.2

            meta = {
                "atr": atr_now,
                "vol_ratio": float(vol_ratio),
                "sl_mult": float(sl_mult),
                "tp_mult": float(tp_mult),
                "spread_price": float(spread_price),
                "spread_frac": float(spread_frac),
                "R": float(r),
                "swing_unreliable": swing_unreliable,
            }
            return float(sl_price), float(tp_price), meta

        except Exception as e:
            self.logger.error(f"Dynamic SL/TP calculation error: {e}")
            # Fallback به روش ساده
            if signal == 1:
                return (
                    entry - 0.5,
                    entry + 1.0,
                    {
                        "atr": 0.5,
                        "vol_ratio": 1.0,
                        "sl_mult": 1.0,
                        "tp_mult": 2.0,
                        "spread_price": 0.0,
                        "spread_frac": 0.0,
                        "R": 2.0,
                    },
                )
            else:
                return (
                    entry + 0.5,
                    entry - 1.0,
                    {
                        "atr": 0.5,
                        "vol_ratio": 1.0,
                        "sl_mult": 1.0,
                        "tp_mult": 2.0,
                        "spread_price": 0.0,
                        "spread_frac": 0.0,
                        "R": 2.0,
                    },
                )

    def _volume_for_trade(self, entry: float, sl: float) -> float:
        """Return either risk-based lot size or fixed volume from config."""
        if not self.config.USE_RISK_BASED_VOLUME:
            return float(self.config.FIXED_VOLUME)

        # Calculate risk-based volume using SL distance
        sl_dist = abs(entry - sl)
        acc = self.trade_executor.get_account_info()
        balance = float(acc.get('balance', 10000.0))
        dynamic_volume = self.risk_manager.calculate_lot_size(
            balance, self.config.BASE_RISK, sl_dist, self.config.SYMBOL
        )

        # Cap volume with dedicated cap (fallback به MAX_LOT)
        max_volume = float(
            self.config.config_data.get("trading", {}).get(
                "max_risk_volume_cap", self.config.MAX_LOT
            )
        )
        return min(dynamic_volume, max_volume)

    def _execute_trade(self, signal_data: dict[str, Any], df: pd.DataFrame) -> bool:
        try:
            if signal_data['signal'] == 1:
                side = "BUY"
                order_type = mt5.ORDER_TYPE_BUY
            elif signal_data['signal'] == -1:
                side = "SELL"
                order_type = mt5.ORDER_TYPE_SELL
            else:
                return False

            # Dynamic spread check based on ATR (reuse current ATR)
            atr_now = self.risk_manager.get_atr(self.config.SYMBOL)
            ok_spread, spread_price, atr_threshold = (True, 0.0, 0.0)
            if atr_now and MT5_AVAILABLE:
                info = mt5.symbol_info(self.config.SYMBOL)
                tick = mt5.symbol_info_tick(self.config.SYMBOL)
                if info and tick:
                    spread_price = tick.ask - tick.bid
                    spread_points = spread_price / info.point if info.point else 0
                    atr_threshold = atr_now * self.max_spread_atr_frac
                    ok_spread = spread_price <= atr_threshold

                    # Log spread in both points and price
                    self.logger.info(
                        f"Spread check: {spread_points:.1f} pts ({spread_price:.4f}) vs ATR threshold: {atr_threshold:.4f}"
                    )

            if not ok_spread:
                # --- BEGIN PATCH [emit spread_block] ---
                self.ev.emit(
                    "blocked_by_spread",
                    spread_price=float(spread_price),
                    atr_now=float(atr_now or 0),
                    threshold=float(atr_threshold),
                )
                # --- END PATCH ---
                self.logger.info(
                    f"Skip trade due to high spread: {spread_price:.4f} > {atr_threshold:.4f} (ATR-based)"
                )
                return False

            # Base entry from last close, then refine with current tick
            entry_price = float(df['close'].iloc[-1])
            tick = self.data_manager.get_current_tick()
            if tick:
                # Use proper bid/ask for entry
                if signal_data['signal'] == 1:  # BUY
                    entry_price = tick['ask']
                else:  # SELL
                    entry_price = tick['bid']
            else:
                # Fallback: use close price but log warning
                self.logger.warning("No current tick available, using close price for entry")

            # Calc dynamic SL/TP بر اساس شرایط سیگنال/بازار
            sl_price, tp_price, meta = self._calculate_dynamic_sl_tp(
                df,
                entry_price,
                signal_data['signal'],
                signal_data.get('confidence', 0.5),
                self.config.SYMBOL,
            )

            # Enforce min distance (trade_stops_level / freeze) + rounding
            is_buy = signal_data['signal'] == 1
            sl_price, tp_price = enforce_min_distance_and_round(
                self.config.SYMBOL, entry_price, sl_price, tp_price, is_buy
            )

            # Recompute R after rounding (واقعی)
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)
            R_post = (reward / risk) if risk > 0 else 0.0

            # تطبیق‌پذیر کردن R بعد از راندینگ (نرم‌تر شده)
            if R_post < 1.2:
                # اطلاعات لازم
                info = mt5.symbol_info(self.config.SYMBOL) if MT5_AVAILABLE else None
                point = info.point if info else 0.01
                stops_pts = float(getattr(info, 'trade_stops_level', 0) or 0) if info else 0.0
                freeze_pts = float(getattr(info, 'trade_freeze_level', 0) or 0) if info else 0.0
                min_dist = max(stops_pts, freeze_pts) * float(point)

                # 1) تلاش با نزدیک‌کردن SL (بدون شکستن ساختار)
                if is_buy:
                    sl_candidate = max(sl_price, entry_price - max(min_dist, 3.0 * point))
                else:
                    sl_candidate = min(sl_price, entry_price + max(min_dist, 3.0 * point))

                sl_try, _ = enforce_min_distance_and_round(
                    self.config.SYMBOL, entry_price, sl_candidate, tp_price, is_buy
                )
                R_try = abs(tp_price - entry_price) / max(1e-9, abs(entry_price - sl_try))

                if R_try >= 1.2:
                    sl_price = sl_try
                    R_post = R_try
                    self.logger.info(f"SL adjusted to reach R=1.2: {R_try:.2f}")
                else:
                    # 2) در غیر این صورت TP را برای 1.2R تنظیم کن
                    desired_tp = entry_price + np.sign(tp_price - entry_price) * 1.2 * abs(
                        entry_price - sl_price
                    )
                    _, tp_try = enforce_min_distance_and_round(
                        self.config.SYMBOL, entry_price, sl_price, desired_tp, is_buy
                    )
                    R_try2 = abs(tp_try - entry_price) / max(1e-9, abs(entry_price - sl_price))
                    if R_try2 >= 1.2:
                        tp_price = tp_try
                        R_post = R_try2
                        self.logger.info(f"TP adjusted to reach R=1.2: {R_try2:.2f}")
                    else:
                        self.logger.info("Skip trade due to low R after fallback")
                        return False

            # Volume (risk-based or fixed)
            volume_total = self._volume_for_trade(entry_price, sl_price)

            # --- BEGIN PATCH [Unified use_split flag] ---
            use_split = bool(self.tp_policy.get("split", True))
            # --- END PATCH ---

            if not MT5_AVAILABLE or not self.mt5_connected:
                if use_split:
                    self.logger.info(
                        f"[DEMO] {side} SPLIT: Vol1={volume_total*self.tp_policy['tp1_share']:.2f} TP1={self._r_to_price(entry_price, sl_price, signal_data['signal'], self.tp_policy['tp1_r']):.2f}, Vol2={volume_total*(1-self.tp_policy['tp1_share']):.2f} TP2={self._r_to_price(entry_price, sl_price, signal_data['signal'], self.tp_policy['tp2_r']):.2f}"
                    )
                else:
                    self.logger.info(
                        f"[DEMO] {side} at {entry_price:.2f} SL={sl_price:.2f} TP={tp_price:.2f} Vol={volume_total}"
                    )
                return True

            # deviation تطبیقی بر اساس اسپرد با multiplier
            deviation = 20
            if MT5_AVAILABLE:
                info = mt5.symbol_info(self.config.SYMBOL)
                tick = mt5.symbol_info_tick(self.config.SYMBOL)
                if info and tick:
                    spread_pts = (tick.ask - tick.bid) / info.point
                    # استفاده از deviation multiplier از config یا مقدار پیش‌فرض
                    deviation_mult = float(
                        self.config.config_data.get("advanced", {}).get("deviation_multiplier", 1.5)
                    )
                    deviation = max(10, int(spread_pts * deviation_mult))

            if use_split:
                # محاسبه TP1 و TP2
                tp1_price = self._r_to_price(
                    entry_price, sl_price, signal_data['signal'], self.tp_policy["tp1_r"]
                )
                tp2_price = self._r_to_price(
                    entry_price,
                    sl_price,
                    signal_data['signal'],
                    max(self.tp_policy["tp2_r"], self.min_R_after_round),
                )

                # رُندینگ TP ها
                _, tp1_price = enforce_min_distance_and_round(
                    self.config.SYMBOL, entry_price, sl_price, tp1_price, is_buy
                )
                _, tp2_price = enforce_min_distance_and_round(
                    self.config.SYMBOL, entry_price, sl_price, tp2_price, is_buy
                )

                # تقسیم حجم با رُندینگ دقیق به volume_step
                def _align_volume(v: float) -> float:
                    if not MT5_AVAILABLE:
                        return round(v, 2)
                    info = mt5.symbol_info(self.config.SYMBOL)
                    if not info or not info.volume_step:
                        return round(v, 2)
                    step = Decimal(str(info.volume_step))
                    vdec = Decimal(str(v))
                    aligned = (vdec / step).to_integral_value(rounding=ROUND_DOWN) * step
                    return float(max(info.volume_min, min(float(aligned), info.volume_max)))

                v1 = _align_volume(volume_total * self.tp_policy["tp1_share"])
                v2 = _align_volume(volume_total - v1)

                # چک حداقل lot
                if v1 < self.config.MIN_LOT or v2 < self.config.MIN_LOT:
                    self.logger.info("Volume too small for split, using single TP")
                    use_split = False
                    orders_to_send = [(volume_total, tp_price, "single")]
                else:
                    # ارسال دو سفارش
                    orders_to_send = [(v1, tp1_price, "TP1"), (v2, tp2_price, "TP2")]
            else:
                # سفارش تک TP
                orders_to_send = [(volume_total, tp_price, "single")]

            # Log order strategy
            if use_split and len(orders_to_send) > 1:
                self.logger.info(
                    f"📤 Sending {side} SPLIT: px={entry_price:.2f} sl={sl_price:.2f} "
                    f"TP1={orders_to_send[0][1]:.2f}(vol={orders_to_send[0][0]}) TP2={orders_to_send[1][1]:.2f}(vol={orders_to_send[1][0]}) "
                    f"(ATR={meta.get('atr',0):.3f} R_post≈{R_post:.2f})"
                )
            else:
                self.logger.info(
                    f"📤 Sending {side}: px={entry_price:.2f} sl={sl_price:.2f} tp={orders_to_send[0][1]:.2f} vol={orders_to_send[0][0]} "
                    f"(ATR={meta.get('atr',0):.3f} vratio={meta.get('vol_ratio',1):.2f} "
                    f"spread={meta.get('spread_price',0):.3f} R≈{meta.get('R',0):.2f} R_post≈{R_post:.2f})"
                )

            # --- BEGIN PATCH [emit order_attempt] ---
            self.ev.emit(
                "order_attempt",
                side=side,
                entry=float(entry_price),
                sl=float(sl_price),
                tp=float(tp_price),
                volume=float(volume_total),
                R_post=float(R_post),
                use_split=bool(use_split),
            )
            # --- END PATCH ---

            # ارسال سفارشات (تک یا چندگانه)
            results = []
            fallback_used = False

            for vol, tp, tag in orders_to_send:
                req_order = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.config.SYMBOL,
                    "volume": float(vol),
                    "type": order_type,
                    "price": float(entry_price),
                    "sl": float(sl_price),
                    "tp": float(tp),
                    "deviation": deviation,
                    "magic": int(self.config.MAGIC),
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                res = mt5.order_send(req_order)
                order_fallback = False

                if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                    # قبل از fallback
                    time.sleep(0.2)
                    # ری‌کووت/لغزش: قیمت را به‌روز کن، deviation تطبیقی و filling را عوض کن
                    tick = mt5.symbol_info_tick(self.config.SYMBOL)
                    if tick:
                        req_order["price"] = (
                            tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
                        )
                    # deviation تطبیقی بر اساس اسپرد
                    info = mt5.symbol_info(self.config.SYMBOL)
                    if info and tick:
                        spread_pts = (tick.ask - tick.bid) / info.point
                        deviation_mult = float(
                            self.config.config_data.get("advanced", {}).get(
                                "deviation_multiplier", 1.5
                            )
                        )
                        req_order["deviation"] = max(10, int(spread_pts * deviation_mult))
                    # فیلینگ جایگزین
                    req_order["type_filling"] = getattr(
                        mt5, "ORDER_FILLING_RETURN", req_order["type_filling"]
                    )
                    res = mt5.order_send(req_order)
                    order_fallback = True

                    # اگر باز هم fail بود، FOK را تست کن
                    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                        req_order["type_filling"] = getattr(
                            mt5, "ORDER_FILLING_FOK", req_order["type_filling"]
                        )
                        res = mt5.order_send(req_order)

                results.append((res, req_order, tag, order_fallback))
                if order_fallback:
                    fallback_used = True

            # برای سازگاری با کد قبلی، از اولین نتیجه استفاده می‌کنیم
            res = results[0][0] if results else None
            req = results[0][1] if results else {}
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Trade log csv برای هر سفارش
            os.makedirs(os.path.dirname(self.config.TRADE_LOG_PATH), exist_ok=True)
            executed_positions = []

            for res_data, req_order, tag, order_fallback in results:
                # --- BEGIN PATCH [emit order_result] ---
                self.ev.emit(
                    "order_result",
                    tag=tag,
                    retcode=int(getattr(res_data, "retcode", -1)) if res_data else -1,
                    executed=bool(res_data and res_data.retcode == mt5.TRADE_RETCODE_DONE),
                    order_id=int(getattr(res_data, "order", 0)) if res_data else 0,
                    fallback_used=bool(order_fallback),
                )
                # --- END PATCH ---

                row = {
                    'run_id': self.run_id,
                    'timestamp': ts,
                    'symbol': req_order['symbol'],
                    'action': f"{side}_{tag}",
                    'entry_price': req_order['price'],
                    'sl_price': req_order['sl'],
                    'tp_price': req_order['tp'],
                    'volume': req_order['volume'],
                    'confidence': signal_data.get('confidence', 0.0),
                    'source': signal_data.get('source', 'Unknown'),
                    'atr': meta.get('atr', None),
                    'vol_ratio': meta.get('vol_ratio', None),
                    'sl_mult': meta.get('sl_mult', None),
                    'tp_mult': meta.get('tp_mult', None),
                    'spread_price': meta.get('spread_price', None),
                    'spread_frac': meta.get('spread_frac', None),
                    'R_multiple': meta.get('R', None),
                    'R_multiple_post_round': R_post,
                    'fallback_used': order_fallback,
                    'tp_policy': tag,
                }

                if res_data is None:
                    self.logger.error(f"❌ MT5 order_send returned None for {tag}")
                    row.update(
                        {
                            'mt5_order_id': 0,
                            'mt5_retcode': -1,
                            'mt5_executed': False,
                            'mt5_error': 'order_send None',
                        }
                    )
                elif res_data.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.info(f"✅ EXECUTED {tag} #{res_data.order}")
                    row.update(
                        {
                            'mt5_order_id': res_data.order,
                            'mt5_retcode': res_data.retcode,
                            'mt5_executed': True,
                        }
                    )

                    # نگهداری position برای trailing
                    executed_positions.append((res_data.order, req_order))

                else:
                    self.logger.error(
                        f"❌ Order {tag} failed: retcode={res_data.retcode} comment={res_data.comment}"
                    )
                    err_text = str(res_data.comment).replace("\n", " ").replace("\r", " ")
                    row.update(
                        {
                            'mt5_order_id': 0,
                            'mt5_retcode': res_data.retcode,
                            'mt5_executed': False,
                            'mt5_error': err_text,
                        }
                    )

                try:
                    _append_trade_log_csv(row, self.config.TRADE_LOG_PATH)
                except Exception as e:
                    self.logger.error(f"trade log failed for tag={tag}: {e}")

            # ثبت پوزیشن‌های تازه بعد از اجرا
            if any(r and getattr(r[0], 'retcode', None) == mt5.TRADE_RETCODE_DONE for r in results):
                open_pos = _get_open_positions(self.config.SYMBOL, self.config.MAGIC)
                added = 0
                for ticket, p in open_pos.items():
                    if ticket not in self.trailing_registry:
                        self.trailing_registry[ticket] = {
                            "dir": "sell" if p.type == mt5.POSITION_TYPE_SELL else "buy",
                            "entry": float(p.price_open),
                            "sl": float(p.sl) if p.sl else None,
                            "tp": float(p.tp) if p.tp else None,
                        }
                        added += 1
                if added:
                    self.logger.info(
                        "🔗 Trailing attached to %d newly-opened positions: %s",
                        added,
                        list(open_pos.keys()),
                    )

            # --- BEGIN PATCH [Success evaluation based on at least one execution] ---
            executed_any = any(
                r and getattr(r[0], 'retcode', None) == mt5.TRADE_RETCODE_DONE for r in results
            )
            if executed_any:
                # اگر حداقل یکی اجرا شد، True بده
                return True
            else:
                self.logger.error("No orders executed successfully.")
                return False
            # --- END PATCH ---
        except Exception as e:
            self.logger.error(f"_execute_trade error: {e}")
            return False

    def _trading_loop(self):
        cooldown_sec = self.config.COOLDOWN_SECONDS
        last_trade_ts = 0.0
        cycle = 0
        SESSION_LOG_EVERY = 5

        while self.running:
            try:
                # Kill-switch اکوییتی
                if self.session_start_equity:
                    ai = mt5.account_info()
                    if ai and ai.equity <= self.session_start_equity * 0.98:
                        # --- BEGIN PATCH [emit kill_switch] ---
                        self.ev.emit(
                            "kill_switch",
                            reason="session_equity_drop",
                            session_start_equity=self.session_start_equity,
                            equity=float(ai.equity),
                        )
                        # --- END PATCH ---
                        self.logger.error("Kill-switch: equity down >2% in session. Stopping.")
                        self.stop()
                        break

                # Trailing stops update
                now = datetime.now()
                if (now - self.last_trailing_update).seconds >= self.trailing_update_interval:
                    self._update_trailing()
                    self.last_trailing_update = now

                # Daily limits
                pl_today, trades_today = self._today_pl_and_trades()
                acc = self.trade_executor.get_account_info()
                bal = acc.get('balance', 10000.0)
                if bal and (pl_today / bal) <= -abs(self.config.MAX_DAILY_LOSS):
                    self.logger.warning(f"Daily loss limit hit: {pl_today:.2f}/{bal:.2f}")
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue
                if trades_today >= self.config.MAX_TRADES_PER_DAY:
                    self.logger.warning(f"Max trades/day reached: {trades_today}")
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue

                # Sessions filter - allow 24h trading
                sess_now = self._current_session()

                if "24h" in self.config.SESSIONS:
                    if cycle % SESSION_LOG_EVERY == 0:
                        self.logger.info("✅ 24h trading enabled - proceeding with trade analysis")
                elif sess_now not in self.config.SESSIONS:
                    if cycle % SESSION_LOG_EVERY == 0:
                        self.logger.info(
                            f"❌ Outside allowed sessions; skipping. Current: {sess_now}, Allowed: {self.config.SESSIONS}"
                        )
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue
                else:
                    if cycle % SESSION_LOG_EVERY == 0:
                        self.logger.info(
                            f"✅ Session {sess_now} is allowed - proceeding with trade analysis"
                        )

                # کش سبک برای symbol_info (rate limiting)
                if cycle % 5 == 0:  # هر 5 سیکل یکبار
                    sym_info = mt5.symbol_info(self.config.SYMBOL) if MT5_AVAILABLE else None

                # آداپتیو کانفیدنس را فعال کن
                if cycle % 15 == 0:
                    try:
                        self.risk_manager.update_performance_from_history(self.config.SYMBOL)
                    except Exception:
                        pass

                # Data
                df = self.data_manager.get_latest_data(self.config.BARS)
                if df is None or len(df) < 50:
                    self.logger.warning("Insufficient data; retrying...")
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # Bar-gate logic
                bar_ts = (
                    pd.to_datetime(df['time'].iloc[-1])
                    .to_pydatetime()
                    .replace(second=0, microsecond=0)
                )
                if self.last_bar_time is None:
                    self.last_bar_time = bar_ts
                elif bar_ts == self.last_bar_time:
                    time.sleep(self.config.RETRY_DELAY)
                    continue
                else:
                    self.last_bar_time = bar_ts

                # Cooldown
                now_ts = time.time()
                if now_ts - last_trade_ts < cooldown_sec:
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # Risk: open positions count
                open_count = self._get_open_trades_count()
                if not self.risk_manager.can_open_new_trade(
                    acc.get('balance', 10000.0),
                    self.start_balance or acc.get('balance', 10000.0),
                    open_count,
                ):
                    self.logger.info(
                        f"Risk blocked new trade. open={open_count}/{self.config.MAX_OPEN_TRADES}"
                    )
                    time.sleep(self.config.RETRY_DELAY)
                    continue

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

                # Generate AI signal with detailed logging
                self.logger.info(f"🔄 Generating AI signal... Market data: {market_data}")
                sig = self.ai_system.generate_ensemble_signal(market_data)
                self.logger.info(f"🎯 AI Signal generated: {sig}")

                # consecutive signals logic
                if sig['signal'] == self.last_signal:
                    self.consecutive_signals += 1
                else:
                    self.consecutive_signals = 1
                    self.last_signal = sig['signal']

                self.logger.info(
                    f"📊 Signal tracking: current={sig['signal']}, last={self.last_signal}, consecutive={self.consecutive_signals}"
                )

                # --- Build feature vector for Meta/Conformal ---
                # ما از آخرین سطر df با اندیکاتورها استفاده می‌کنیم
                last = df.iloc[-1]
                meta_feats = {
                    "close": float(last['close']),
                    "ret": float(df['close'].pct_change().iloc[-1]) if len(df) > 1 else 0.0,
                    "sma_20": float(last.get('sma_20', 0.0)),
                    "sma_50": float(last.get('sma_50', 0.0)),
                    "atr": float(last.get('atr', 0.0)),
                    "rsi": float(last.get('rsi', 50.0)),
                    "macd": float(last.get('macd', 0.0)),
                    "macd_signal": float(last.get('macd_signal', 0.0)),
                    "hour": float(
                        last['time'].hour if 'time' in df.columns else datetime.now().hour
                    ),
                    "dow": float(
                        last['time'].dayofweek if 'time' in df.columns else datetime.now().weekday()
                    ),
                }

                # اگر سیگنال صفر است، کلاً رد
                if sig['signal'] == 0:
                    self.logger.info("Signal is 0 - skipping")
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # Basic signal validation
                if sig['confidence'] < 0.1:  # Very low confidence
                    self.logger.info(f"Signal confidence too low: {sig['confidence']:.3f} < 0.1")
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # --- Enhanced Conformal Gate with Fallbacks ---
                conformal_p = None
                conformal_status = "disabled"
                soft_cfg = self.config.config_data.get("conformal", {})
                soft_enabled = bool(soft_cfg.get("enabled", True))
                soft_gate = bool(soft_cfg.get("soft_gate", True))
                emergency_bypass = bool(soft_cfg.get("emergency_bypass", False))
                min_p = float(soft_cfg.get("min_p", 0.10))  # حد نرم
                hard_floor = float(soft_cfg.get("hard_floor", 0.05))  # زیر این مقدار، قطعاً رد
                penalty_small = float(soft_cfg.get("penalty_small", 0.05))
                penalty_big = float(soft_cfg.get("penalty_big", 0.10))
                extra_consec = int(soft_cfg.get("extra_consecutive", 1))

                # Emergency bypass for testing
                if emergency_bypass:
                    self.logger.warning(
                        "🚨 EMERGENCY BYPASS ENABLED - Conformal gate completely disabled!"
                    )
                    soft_enabled = False

                # Conformal Gate Logic with Comprehensive Fallbacks
                if self.conformal is not None and soft_enabled:
                    try:
                        ok, p_hat, nonconf = self.conformal.accept(meta_feats)
                        conformal_p = p_hat
                        conformal_status = "active"
                        self.logger.info(f"Conformal: ok={ok} p={p_hat:.3f}")

                        if not soft_gate:
                            # حالت قدیمی: بلاک کامل
                            if not ok or (p_hat is not None and p_hat < min_p):
                                self.logger.info(
                                    f"Conformal hard block: ok={ok}, p={p_hat:.3f} < {min_p}"
                                )
                                time.sleep(self.config.RETRY_DELAY)
                                continue
                        else:
                            # حالت نرم: فقط سخت‌تر کردن شرایط
                            if p_hat is not None and p_hat < hard_floor:
                                self.logger.info(
                                    f"Conformal below hard_floor: p={p_hat:.3f} < {hard_floor}"
                                )
                                time.sleep(self.config.RETRY_DELAY)
                                continue

                    except Exception as e:
                        self.logger.warning(f"Conformal error: {e} - proceeding with fallback")
                        conformal_status = "error"
                        conformal_p = None
                else:
                    if self.conformal is None:
                        conformal_status = "not_available"
                        self.logger.info("Conformal gate not available - proceeding without")
                    elif not soft_enabled:
                        conformal_status = "disabled"
                        self.logger.info("Conformal gate disabled - proceeding without")

                # --- Enhanced Threshold and Consecutive Logic ---
                thr = self.risk_manager.get_current_confidence_threshold()
                base_thr = max(thr, 0.55 if conformal_p is not None else thr)
                adj_thr = base_thr
                req_consec = self.config.CONSECUTIVE_SIGNALS_REQUIRED

                # --- Soft Gate (revised) ---
                if conformal_p is not None and conformal_status == "active":
                    adj_thr, req_consec, override_margin = _apply_soft_gate(
                        p_value=conformal_p,
                        base_thr=base_thr,  # استفاده از base_thr موجود
                        base_consec=1,  # کف ثابت 1
                        max_conf_bump=0.03,  # سقف افزایش آستانه
                        high_conf_override_margin=0.02,  # اگر conf >= adj_thr+0.02 => consecutive=1
                    )

                    self.logger.info("Soft-gate penalty applied:")
                    self.logger.info(
                        "  p=%.3f -> adj_thr: %.3f (base: %.3f) | req_consec: %d",
                        conformal_p,
                        adj_thr,
                        base_thr,
                        req_consec,
                    )

                    # High-confidence override
                    effective_req_consec = (
                        1 if sig['confidence'] >= (adj_thr + override_margin) else req_consec
                    )
                    if effective_req_consec != req_consec:
                        self.logger.info(
                            "  High-confidence override: conf=%.3f >= %.3f -> req_consec -> 1",
                            sig['confidence'],
                            adj_thr + override_margin,
                        )
                    req_consec = effective_req_consec

                # Fallback: If Conformal is not available, use base thresholds
                elif conformal_status in ["not_available", "disabled", "error"]:
                    adj_thr = base_thr
                    req_consec = self.config.CONSECUTIVE_SIGNALS_REQUIRED
                    self.logger.info(
                        f"Using fallback thresholds: adj_thr={adj_thr:.2f}, req_consec={req_consec}"
                    )

                should = (
                    sig['confidence'] >= adj_thr
                    and self.consecutive_signals >= req_consec
                    and sig['signal'] != 0
                    and open_count < self.config.MAX_OPEN_TRADES
                )

                # --- Comprehensive Trade Execution Check ---
                self.logger.info("🔍 Trade execution check:")
                self.logger.info("   Signal: %s", sig['signal'])
                self.logger.info(
                    "   Confidence: %.3f (threshold: %.3f)", sig['confidence'], adj_thr
                )
                self.logger.info(
                    "   Consecutive signals: %d (required: %d)",
                    self.consecutive_signals,
                    req_consec,
                )
                self.logger.info("   Open trades: %d/%d", open_count, self.config.MAX_OPEN_TRADES)
                self.logger.info("   Conformal status: %s", conformal_status)
                if conformal_p is not None:
                    self.logger.info("   Conformal p-value: %.3f", conformal_p)
                self.logger.info("   Should execute: %s", should)

                # Detailed breakdown of why execution might fail
                if not should:
                    self.logger.info("❌ Execution blocked:")
                    if sig['confidence'] < adj_thr:
                        self.logger.info("   - Confidence %.3f < %.3f", sig['confidence'], adj_thr)
                    if self.consecutive_signals < req_consec:
                        self.logger.info(
                            "   - Consecutive %d < %d", self.consecutive_signals, req_consec
                        )
                    if sig['signal'] == 0:
                        self.logger.info("   - Signal is 0")
                    if open_count >= self.config.MAX_OPEN_TRADES:
                        self.logger.info(
                            "   - Max trades reached: %d/%d",
                            open_count,
                            self.config.MAX_OPEN_TRADES,
                        )
                else:
                    self.logger.info("✅ All conditions met - proceeding to execution")

                if should:
                    self.logger.info("🚀 Executing trade...")
                    ok = self._execute_trade(sig, df)
                    if ok:
                        last_trade_ts = now_ts
                        self.logger.info(
                            f"✅ Trade executed successfully. Cooldown {cooldown_sec}s"
                        )
                    else:
                        self.logger.error("❌ Trade execution failed")
                else:
                    self.logger.info("⏸️ Trade execution skipped - conditions not met")

                # Cycle summary
                if cycle % 10 == 0:  # Every 10 cycles
                    self.logger.info(f"📊 Cycle {cycle} summary:")
                    self.logger.info(f"   Conformal status: {conformal_status}")
                    self.logger.info(
                        f"   Last signal: {sig['signal']} (confidence: {sig['confidence']:.3f})"
                    )
                    self.logger.info(f"   Consecutive: {self.consecutive_signals}")
                    self.logger.info(f"   Open trades: {open_count}")

                cycle += 1
                time.sleep(self.config.SLEEP_SECONDS)

            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                time.sleep(self.config.RETRY_DELAY)


# -----------------------------
# Main
# -----------------------------


def main():
    # برای دمو/سنتتیک: تست‌ها تکرارپذیر باشن
    np.random.seed(42)

    trader = MT5LiveTrader()
    try:
        trader.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        trader.stop()


if __name__ == "__main__":
    main()
