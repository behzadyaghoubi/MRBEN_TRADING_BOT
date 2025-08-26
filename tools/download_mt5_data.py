import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# اطلاعات ورود شما (اینجا مقداردهی شده)
login = 1104123
password = "-4YcBgRd"
server = "OxSecurities-Demo"
symbol = "XAUUSD.PRO"
timeframe = mt5.TIMEFRAME_M5

# تعداد کندل (هرچه بیشتر بگذاری، دیتای بیشتری جمع می‌کنی. MT5 معمولاً تا 100,000 کندل هم جواب می‌دهد)
bars = 10000

# اتصال به متاتریدر 5
if not mt5.initialize(login=login, password=password, server=server):
    print("⛔️ اتصال به متاتریدر برقرار نشد:", mt5.last_error())
    quit()

print(f"✅ متصل به {server}، جمع‌آوری داده برای نماد {symbol}، تایم‌فریم M5...")

# گرفتن دیتا از آخرین کندل (0 = آخرین کندل)
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

# قطع ارتباط
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("⛔️ دیتایی دریافت نشد. لطفاً symbol یا تنظیمات را چک کن.")
else:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    df.to_csv("ohlc_data.csv", index=False)
    print(f"✅ دیتا با موفقیت ذخیره شد! تعداد ردیف‌ها: {len(df)}")
    print(df.head())