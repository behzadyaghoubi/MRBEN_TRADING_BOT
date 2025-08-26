import MetaTrader5 as mt5
import pandas as pd

symbol = "XAUUSD.PRO"
timeframe = mt5.TIMEFRAME_M15
n_candles = 2000  # هر چقدر می‌خوای (مثلاً 2000 کندل اخیر)

if not mt5.initialize():
    print("⛔️ اتصال به متاتریدر برقرار نشد.")
    exit(1)

rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
if rates is None or len(rates) == 0:
    print("⛔️ هیچ دیتایی از متاتریدر گرفته نشد. نام نماد یا تایم‌فریم رو چک کن!")
    mt5.shutdown()
    exit(1)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.to_csv("XAUUSD_PRO_M15_history.csv", index=False)
print(f"✅ {len(df)} کندل ذخیره شد! خروجی: XAUUSD_PRO_M15_history.csv")

mt5.shutdown()