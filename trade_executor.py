import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import json
import os

def send_order(symbol, signal, volume=0.01, sl_distance=300, tp_distance=500):
    DEVIATION = 10
    MAGIC = 20250615
    LOG_FILE = "live_trades_log.csv"

    if not mt5.initialize():
        print("⛔ اتصال به متاتریدر برقرار نشد:", mt5.last_error())
        return

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"⛔ نماد '{symbol}' یافت نشد.")
        mt5.shutdown()
        return

    point = symbol_info.point
    tick = mt5.symbol_info_tick(symbol)
    ask = tick.ask
    bid = tick.bid

    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    entry = ask if signal == "BUY" else bid
    sl = entry - sl_distance * point if signal == "BUY" else entry + sl_distance * point
    tp = entry + tp_distance * point if signal == "BUY" else entry - tp_distance * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "MRBEN ML Trade",
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
        "comment": result.comment
    }

    if os.path.exists(LOG_FILE):
        pd.DataFrame([log]).to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        pd.DataFrame([log]).to_csv(LOG_FILE, index=False)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("✅ سفارش با موفقیت اجرا شد و ذخیره شد")
    else:
        print(f"⛔ سفارش رد شد: {result.retcode} - {result.comment}")

    mt5.shutdown()