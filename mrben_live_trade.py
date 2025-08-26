import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import time

from strategies.strategy_ema_crossover import ema_crossover_signal
from strategies.strategy_breakout import breakout_signal
from strategies.strategy_bollinger import bollinger_signal
from strategies.strategy_risk_trailing import calc_position_size, trailing_stop

# اتصال به MT5
mt5.initialize()

symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M15
window = 60  # چند کندل قبل برای سیگنال‌گیری
lot_min = 0.01

clf = joblib.load("mrben_ai_signal_filter.joblib")
capital = 10000  # فرض اولیه برای حجم؛ می‌تونی با حساب لایو داینامیکش کنی

def get_live_data(symbol, timeframe, window):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, window)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

while True:
    df = get_live_data(symbol, timeframe, window)
    if df is None or len(df) < 50:
        print("دیتا کافی نیست...")
        time.sleep(10)
        continue

    d = df
    price = d['close'].iloc[-1]
    high = d['high'].iloc[-1]
    low = d['low'].iloc[-1]
    atr = d['high'].rolling(14).max().iloc[-1] - d['low'].rolling(14).min().iloc[-1]
    lot = max(calc_position_size(capital, 0.01, atr, 2), lot_min)

    sig_ema = ema_crossover_signal(d)
    sig_breakout = breakout_signal(d)
    sig_boll = bollinger_signal(d)
    signals = [sig_ema, sig_breakout, sig_boll]
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    ema_num = 1 if sig_ema == 'BUY' else -1 if sig_ema == 'SELL' else 0
    break_num = 1 if sig_breakout == 'BUY' else -1 if sig_breakout == 'SELL' else 0
    boll_num = 1 if sig_boll == 'BUY' else -1 if sig_boll == 'SELL' else 0
    last_res = 0  # در حالت زنده فعلاً 0 می‌گذاریم

    X_ai = [[ema_num, break_num, boll_num, lot, price, last_res]]
    ai_signal = clf.predict(X_ai)[0]

    # بررسی معاملات باز
    positions = mt5.positions_get(symbol=symbol)
    open_position = len(positions) > 0

    # ورود به معامله اگر سیگنال تایید شد و معامله باز نیست
    if buy_count >= 2 and ai_signal == 1 and not open_position:
        price_ask = mt5.symbol_info_tick(symbol).ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price_ask,
            "deviation": 10,
            "magic": 42,
            "comment": "MR BEN LIVE BUY",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        print("📈 سیگنال BUY صادر شد:", result)
    elif sell_count >= 2 and ai_signal == 1 and not open_position:
        price_bid = mt5.symbol_info_tick(symbol).bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price_bid,
            "deviation": 10,
            "magic": 42,
            "comment": "MR BEN LIVE SELL",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        print("📉 سیگنال SELL صادر شد:", result)
    else:
        print("⏳ منتظر سیگنال جدید...")

    time.sleep(60)  # هر ۱ دقیقه دیتا و سیگنال جدید چک شود

mt5.shutdown()