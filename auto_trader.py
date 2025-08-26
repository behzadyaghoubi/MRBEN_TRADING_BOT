import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import json
import os
import sys
import logging

SETTINGS_FILE = "settings.json"
SIGNAL_FILE = "combined_signals.csv"
LOG_FILE = "live_trades_log.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def read_settings():
    if not os.path.exists(SETTINGS_FILE):
        logging.error("فایل settings.json یافت نشد.")
        sys.exit(1)
    with open(SETTINGS_FILE) as f:
        return json.load(f)

def read_latest_signal():
    if not os.path.exists(SIGNAL_FILE):
        logging.error("فایل سیگنال‌ها موجود نیست.")
        sys.exit(1)
    df = pd.read_csv(SIGNAL_FILE)
    if df.empty or "final_signal" not in df.columns:
        logging.error("سیگنال معتبر یافت نشد.")
        sys.exit(1)
    last_signal = df.iloc[-1]["final_signal"]
    if last_signal not in ["BUY", "SELL"]:
        logging.warning(f"سیگنال آخر ({last_signal}) معتبر نیست.")
        sys.exit(1)
    return last_signal

def connect_mt5(login=None, password=None, server=None):
    if login and password and server:
        if not mt5.initialize(login=login, password=password, server=server):
            logging.error(f"اتصال به متاتریدر برقرار نشد: {mt5.last_error()}")
            sys.exit(1)
    else:
        if not mt5.initialize():
            logging.error(f"اتصال به متاتریدر برقرار نشد: {mt5.last_error()}")
            sys.exit(1)
    return True

def get_symbol_info(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        logging.error(f"نماد {symbol} یافت نشد.")
        mt5.shutdown()
        sys.exit(1)
    return info

def get_tick(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"داده قیمت لحظه‌ای {symbol} یافت نشد.")
        mt5.shutdown()
        sys.exit(1)
    return tick

def log_trade(log):
    df = pd.DataFrame([log])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def has_open_trade(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            if pos.magic == magic:
                return True
    return False

def execute_trade(signal, symbol, volume, sl_dist, tp_dist, deviation, magic):
    if has_open_trade(symbol, magic):
        logging.info(f"معامله باز با magic={magic} روی {symbol} وجود دارد. سفارش جدید ارسال نشد.")
        return

    info = get_symbol_info(symbol)
    point = info.point
    tick = get_tick(symbol)
    ask, bid = tick.ask, tick.bid

    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    entry = ask if signal == "BUY" else bid
    sl = entry - sl_dist * point if signal == "BUY" else entry + sl_dist * point
    tp = entry + tp_dist * point if signal == "BUY" else entry - tp_dist * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": magic,
        "comment": "MRBEN AutoTrade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "action": signal,
        "entry_price": entry,
        "sl": sl,
        "tp": tp,
        "result_code": result.retcode,
        "comment": result.comment,
        "order_id": getattr(result, 'order', None),
    }
    log_trade(log)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info("سفارش با موفقیت انجام شد.")
    else:
        logging.error(f"سفارش رد شد: {result.retcode} - {result.comment}")

def main():
    settings = read_settings()
    if not settings.get("enabled", False):
        logging.warning("ربات غیرفعاله. اجرا متوقف شد.")
        sys.exit(0)

    SYMBOL = settings.get("symbol", "XAUUSD")
    VOLUME = settings.get("volume", 0.01)
    SL_DISTANCE = settings.get("sl_distance", 300)
    TP_DISTANCE = settings.get("tp_distance", 500)
    DEVIATION = settings.get("deviation", 10)
    MAGIC = settings.get("magic", 20250615)
    LOGIN = settings.get("login")
    PASSWORD = settings.get("password")
    SERVER = settings.get("server")

    latest_signal = read_latest_signal()
    connect_mt5(LOGIN, PASSWORD, SERVER)
    logging.info(f"اتصال برقرار شد - سیگنال: {latest_signal}")
    execute_trade(
        latest_signal, SYMBOL, VOLUME, SL_DISTANCE, TP_DISTANCE, DEVIATION, MAGIC
    )
    mt5.shutdown()

if __name__ == "__main__":
    main()