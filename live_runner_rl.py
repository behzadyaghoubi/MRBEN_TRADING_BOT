import time
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
from rl_predictor import load_trained_agent, predict_next_signal
from trade_executor import send_order
import os

# اتصال به متاتریدر
def connect():
    if not mt5.initialize():
        print("❌ اتصال به متاتریدر برقرار نشد:", mt5.last_error())
        return False
    print("✅ اتصال به متاتریدر برقرار شد")
    return True

# دریافت دیتا
def get_price_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        print("❌ دریافت دیتا شکست خورد")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["price"] = df["close"]  # اضافه کردن ستون price از close
    return df

# مسیر فایل لاگ
log_path = "live_trades.csv"
if not os.path.exists(log_path):
    pd.DataFrame(columns=["time", "symbol", "signal", "price"]).to_csv(log_path, index=False)

def get_last_signal():
    try:
        df = pd.read_csv(log_path)
        return df.iloc[-1]["signal"]
    except:
        return None

def log_trade(symbol, signal, price):
    new_row = {"time": datetime.now(), "symbol": symbol, "signal": signal, "price": price}
    df = pd.read_csv(log_path)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(log_path, index=False)

def run_rl_trader():
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M5
    last_signal = get_last_signal()
    agent = load_trained_agent("trained_q_table.npy", epsilon=0.01)

    while True:
        df = get_price_data(symbol, timeframe)
        if df is None:
            time.sleep(10)
            continue

        signal = predict_next_signal(agent, df)

        if signal != last_signal:
            price = mt5.symbol_info_tick(symbol).ask if signal == "BUY" else mt5.symbol_info_tick(symbol).bid
            send_order(symbol, signal, price, df)
            log_trade(symbol, signal, price)
            last_signal = signal
            print(f"🚀 سیگنال RL {signal} در قیمت {price} اجرا شد")
        else:
            print("🔁 سیگنال RL جدیدی وجود ندارد یا تکراری است")

        time.sleep(15)

if __name__ == "__main__":
    if connect():
        run_rl_trader()
