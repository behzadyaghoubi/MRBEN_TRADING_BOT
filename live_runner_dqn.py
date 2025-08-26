import time
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import tensorflow as tf
import numpy as np
import os

MODEL_PATH = 'trained_dqn_model.h5'
CSV_PATH = 'ohlc_data.csv'
LOG_PATH = 'live_trades.csv'
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
WINDOW_SIZE = 10

def connect():
    if not mt5.initialize():
        print("❌ اتصال به متاتریدر برقرار نشد:", mt5.last_error())
        return False
    print("✅ اتصال به متاتریدر برقرار شد")
    return True

def get_price_data(symbol=SYMBOL, timeframe=TIMEFRAME, bars=200):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        print("❌ دریافت دیتا شکست خورد")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

def load_dqn_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)

def predict_action(model, df, window_size=WINDOW_SIZE):
    state = df[['open', 'high', 'low', 'close']].values[-window_size:]
    state = np.expand_dims(state, axis=0)
    q_values = model.predict(state, verbose=0)
    action = np.argmax(q_values[0])
    return action

def get_last_signal():
    if not os.path.exists(LOG_PATH):
        return None
    try:
        df = pd.read_csv(LOG_PATH)
        return df.iloc[-1]["signal"]
    except:
        return None

def log_trade(symbol, signal, price):
    new_row = {"time": datetime.now(), "symbol": symbol, "signal": signal, "price": price}
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(LOG_PATH, index=False)

def send_order(symbol, signal, price, lot=0.1, deviation=10):
    action = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action,
        "price": price,
        "deviation": deviation,
        "magic": 234567,
        "comment": "MRBEN_DQN_TRADE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"⛔ سفارش رد شد: {result.retcode} - {result.comment}")
    else:
        print(f"✅ سفارش با موفقیت انجام شد: Ticket={result.order}")

def run_dqn_trader():
    model = load_dqn_model()
    last_signal = get_last_signal()
    action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}

    while True:
        df = get_price_data()
        if df is None or len(df) < WINDOW_SIZE:
            print("⏳ صبر برای داده بیشتر یا دیتای ناقص.")
            time.sleep(20)
            continue

        action = predict_action(model, df, window_size=WINDOW_SIZE)
        signal = action_map[action]

        if signal != last_signal and signal in ["BUY", "SELL"]:
            price = mt5.symbol_info_tick(SYMBOL).ask if signal == "BUY" else mt5.symbol_info_tick(SYMBOL).bid
            send_order(SYMBOL, signal, price)
            log_trade(SYMBOL, signal, price)
            last_signal = signal
            print(f"🚀 سیگنال DQN: {signal} در قیمت {price} اجرا شد")
        else:
            print("🔁 سیگنال جدیدی یافت نشد یا مشابه قبلی است.")

        time.sleep(20)

if __name__ == "__main__":
    if connect():
        run_dqn_trader()