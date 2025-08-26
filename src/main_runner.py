#!/usr/bin/env python3
"""
MRBEN Professional Trading Bot - Main Runner (PRO+ Enhanced)
============================================================
یک سیستم معاملاتی الگوریتمی با فیلتر هوشمند سیگنال، مدیریت ریسک، لاگ حرفه‌ای و اتصال به متاتریدر۵

Author: MRBEN AI Trading System (2025)
"""

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import MetaTrader5 as mt5
import pandas as pd

# ---- Import your modules ----
from ai_filter import AISignalFilter
from risk_manager import RiskManager

# ---- CONSTANTS ----
SETTINGS_FILE = "settings.json"
AI_MODEL_PATH = "mrben_ai_signal_filter_xgb.joblib"
LOG_FILE = "live_trades_log.csv"
TRADING_LOG_FILE = "mrben_trading.log"
MAGIC_NUMBER = 20250716


# ---- CONFIG DATACLASS ----
@dataclass
class TradingConfig:
    login: int
    password: str
    server: str
    symbol: str = "XAUUSD"
    timeframe: str = "M15"
    lot_min: float = 0.01
    stop_loss_pips: int = 30
    take_profit_pips: int = 60
    risk_per_trade: float = 0.01
    start_balance: float = 10000
    enabled: bool = True
    max_open_trades: int = 3
    ai_confidence_threshold: float = 0.6


# ---- LOGGING ----
def setup_logger(log_file: str = TRADING_LOG_FILE) -> logging.Logger:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("MRBEN_Trading")
    return logger


# ---- MT5 MANAGER ----
class MT5Manager:
    """Handles MetaTrader5 connection and operations."""

    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.connected = False
        self.point = None

    def connect(self) -> bool:
        try:
            if not mt5.initialize(
                login=self.config.login, password=self.config.password, server=self.config.server
            ):
                self.logger.error(f"MT5 connection failed: {mt5.last_error()}")
                return False
            self.connected = True
            self.logger.info("MT5 connection established.")
            # Get symbol info and point value
            symbol_info = mt5.symbol_info(self.config.symbol)
            if not symbol_info:
                self.logger.error(f"Symbol '{self.config.symbol}' not found.")
                return False
            self.point = symbol_info.point
            return True
        except Exception as e:
            self.logger.error(f"MT5 connect error: {e}")
            return False

    def shutdown(self):
        if self.connected:
            mt5.shutdown()
            self.logger.info("MT5 disconnected.")
            self.connected = False

    def get_data(self, bars: int = 200) -> pd.DataFrame | None:
        try:
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
            tf = tf_map.get(self.config.timeframe.upper(), mt5.TIMEFRAME_M15)
            rates = mt5.copy_rates_from_pos(self.config.symbol, tf, 0, bars)
            if rates is None or len(rates) < 50:
                self.logger.error("Not enough market data received.")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            self.logger.error(f"Error getting MT5 data: {e}")
            return None

    def get_account_info(self):
        try:
            return mt5.account_info()
        except:
            return None

    def get_open_positions(self) -> list:
        try:
            positions = mt5.positions_get(symbol=self.config.symbol)
            return positions if positions else []
        except:
            return []

    def get_current_price(self, signal: str) -> float | None:
        try:
            tick = mt5.symbol_info_tick(self.config.symbol)
            if not tick:
                return None
            return tick.ask if signal == "BUY" else tick.bid
        except:
            return None


# ---- SIGNAL/FEATURE PROCESSING ----
def extract_features(row: pd.Series) -> list[float]:
    # مطمئن شو همین ترتیب در آموزش ML هم بوده
    return [
        row.get('SMA_FAST', 0),
        row.get('SMA_SLOW', 0),
        row.get('RSI', 0),
        row.get('MACD', 0),
        row.get('MACD_signal', 0),
        row.get('close', 0),
    ]


def load_config() -> TradingConfig | None:
    if not os.path.exists(SETTINGS_FILE):
        print(f"❌ {SETTINGS_FILE} not found.")
        return None
    with open(SETTINGS_FILE) as f:
        d = json.load(f)
    # مقادیر پیش‌فرض و چک کلیدهای اجباری
    for field in ["login", "password", "server"]:
        if field not in d or not d[field]:
            print(f"❌ settings.json: missing field {field}")
            return None
    # پیش‌فرض‌ها
    defaults = {
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "enabled": True,
        "lot_min": 0.01,
        "stop_loss_pips": 30,
        "take_profit_pips": 60,
        "risk_per_trade": 0.01,
        "start_balance": 10000,
        "max_open_trades": 3,
        "ai_confidence_threshold": 0.6,
    }
    for k, v in defaults.items():
        d.setdefault(k, v)
    return TradingConfig(**d)


# ---- TRADE EXECUTION ----
def send_order(
    mt5m: MT5Manager,
    config: TradingConfig,
    signal: str,
    lot: float,
    price: float,
    sl: float,
    tp: float,
    confidence: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": config.symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": MAGIC_NUMBER,
        "comment": f"MRBEN AI Trade Conf={confidence:.2f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    try:
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Trade executed: {signal} {lot} @ {price}")
            return {"success": True, "result": result}
        else:
            logger.warning(f"Trade rejected: {result.retcode} - {result.comment}")
            return {"success": False, "result": result}
    except Exception as e:
        logger.error(f"Order send error: {e}")
        return {"success": False, "result": None}


# ---- MAIN LOGIC ----
def main():
    logger = setup_logger()
    logger.info("==== MRBEN PRO Trading Bot Started ====")

    config = load_config()
    if not config:
        logger.error("Invalid or missing config.")
        return

    if not config.enabled:
        logger.info("Trading disabled in config.")
        return

    mt5m = MT5Manager(config, logger)
    if not mt5m.connect():
        logger.error("MT5 connection failed.")
        return

    # Initialize AI and Risk
    ai_filter = AISignalFilter(AI_MODEL_PATH)
    risk_manager = RiskManager(
        base_risk=config.risk_per_trade,
        dynamic=True,
        min_lot=config.lot_min,
        max_lot=2.0,
        max_open_trades=config.max_open_trades,
        dynamic_sensitivity=0.5,
    )

    try:
        # ----- Data & Signal -----
        df = mt5m.get_data()
        if df is None:
            logger.error("No data from MT5, aborting.")
            return

        from book_strategy import generate_book_signals  # خودت باید داشته باشی

        df = generate_book_signals(df)
        last = df.iloc[-1]
        features = extract_features(last)

        # ----- AI Signal Filter -----
        ai_decision = ai_filter.filter_signal([features])
        confidence = ai_filter.get_confidence([features])
        logger.info(f"Signal: {last['signal']} | AI: {ai_decision} | Conf: {confidence:.2f}")

        if (
            last['signal'] == "HOLD"
            or ai_decision == 0
            or confidence < config.ai_confidence_threshold
        ):
            logger.info("No trade: No valid or confident signal.")
            return

        # ----- Risk Management -----
        acc = mt5m.get_account_info()
        balance = acc.balance if acc else config.start_balance
        open_trades = len(mt5m.get_open_positions())
        pip_value = 1  # اگر نیاز بود تغییر بده
        lot = risk_manager.calc_lot_size(
            balance, config.stop_loss_pips, pip_value, open_trades, config.start_balance
        )
        if lot < config.lot_min:
            logger.warning(f"Lot {lot:.2f} < min lot ({config.lot_min}), no trade.")
            return

        price = mt5m.get_current_price(last['signal'])
        if price is None:
            logger.error("Failed to get price.")
            return

        if last['signal'] == "BUY":
            sl = price - config.stop_loss_pips * mt5m.point
            tp = price + config.take_profit_pips * mt5m.point
        else:
            sl = price + config.stop_loss_pips * mt5m.point
            tp = price - config.take_profit_pips * mt5m.point

        # ----- Execute Trade -----
        trade_result = send_order(
            mt5m, config, last['signal'], lot, price, sl, tp, confidence, logger
        )

        # ----- Log Trade -----
        log_dict = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": config.symbol,
            "action": last['signal'],
            "entry_price": price,
            "sl": sl,
            "tp": tp,
            "lot": lot,
            "balance": balance,
            "ai_decision": int(ai_decision),
            "ai_confidence": confidence,
            "result_code": trade_result["result"].retcode if trade_result["result"] else -1,
            "comment": trade_result["result"].comment if trade_result["result"] else "FAILED",
        }
        try:
            df_log = pd.DataFrame([log_dict])
            if os.path.exists(LOG_FILE):
                df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)
            else:
                df_log.to_csv(LOG_FILE, index=False)
            logger.info("Trade logged.")
        except Exception as e:
            logger.error(f"Log trade error: {e}")

        if trade_result["success"]:
            logger.info("Trade completed and logged.")
        else:
            logger.error("Trade execution failed.")

    except Exception as e:
        logger.error(f"Critical bot error: {e}")
        logger.error(traceback.format_exc())
    finally:
        mt5m.shutdown()
        logger.info("==== MRBEN PRO Trading Bot Shutdown ====")


if __name__ == "__main__":
    main()
