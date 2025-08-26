import os
import time
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

# از ماژول جدید LSTM استفاده کن
from lstm_live_signal_generator import generate_lstm_live_signal

from ai_filter import AISignalFilter
from trade_executor import send_order


# اتصال به متاتریدر
def connect():
    if not mt5.initialize():
        print("❌ اتصال به متاتریدر برقرار نشد:", mt5.last_error())
        return False
    print("✅ اتصال به متاتریدر برقرار شد")
    return True


# دریافت قیمت و تبدیل به دیتافریم
def get_price_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        print("❌ دریافت دیتا شکست خورد")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


# مسیر فایل برای ذخیره سیگنال‌های اجرا شده
log_path = "live_trades.csv"
if not os.path.exists(log_path):
    pd.DataFrame(columns=["time", "symbol", "signal", "price"]).to_csv(log_path, index=False)


# گرفتن آخرین سیگنال ثبت شده
def get_last_signal():
    try:
        df = pd.read_csv(log_path)
        return df.iloc[-1]["signal"]
    except:
        return None


# ذخیره سیگنال جدید
def log_trade(symbol, signal, price):
    new_row = {"time": datetime.now(), "symbol": symbol, "signal": signal, "price": price}
    df = pd.read_csv(log_path)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(log_path, index=False)


# ساخت شیء فیلتر سیگنال (در صورت نیاز)
ai_filter = AISignalFilter(
    model_path="mrben_ai_signal_filter_xgb.joblib", model_type="joblib", threshold=0.55
)


# حلقه اصلی اجرای ربات
def run_live_bot():
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M5
    last_signal = get_last_signal()

    while True:
        try:
            if not mt5.symbol_select(symbol, True):
                print("❌ نماد انتخاب نشد")
                continue

            df = get_price_data(symbol, timeframe)
            if df is None:
                time.sleep(10)
                continue

            # استفاده از مدل LSTM جدید برای تولید سیگنال
            signal = generate_lstm_live_signal(df)
            # فیلتر سیگنال با مدل ML (در صورت نیاز)
            filtered = ai_filter.filter_signal(signal)
            if filtered in [2, 0] and filtered != last_signal:  # BUY=2, SELL=0
                price = (
                    mt5.symbol_info_tick(symbol).ask
                    if filtered == 2
                    else mt5.symbol_info_tick(symbol).bid
                )
                send_order(symbol, "BUY" if filtered == 2 else "SELL", price, df)
                log_trade(symbol, "BUY" if filtered == 2 else "SELL", price)
                last_signal = filtered
                print(f"🚀 سیگنال {('BUY' if filtered == 2 else 'SELL')} در قیمت {price} اجرا شد")
            else:
                print("🔁 سیگنال جدیدی وجود ندارد یا تکراری است")
        except Exception as e:
            print(f"❌ خطا در حلقه اجرا: {e}")

        time.sleep(15)  # فاصله بررسی سیگنال بعدی


if __name__ == "__main__":
    if connect():
        run_live_bot()
    else:
        print("⛔ اتصال به متاتریدر برقرار نشد")
