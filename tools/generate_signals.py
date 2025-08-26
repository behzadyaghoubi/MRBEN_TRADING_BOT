import pandas as pd
import talib

# ورودی دیتای ohlc با فرمت استاندارد
IN_FILE = "XAUUSD_M15_history.csv"
OUT_FILE = "XAUUSD_M15_history_with_signals.csv"

df = pd.read_csv(IN_FILE)
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])

# محاسبه اندیکاتورهای پایه
df['SMA_FAST'] = talib.SMA(df['close'], timeperiod=20)
df['SMA_SLOW'] = talib.SMA(df['close'], timeperiod=50)
df['RSI'] = talib.RSI(df['close'], timeperiod=14)

# استراتژی ساده: کراسینگ SMA و RSI با محدودیت بازتر
signals = []
for i in range(1, len(df)):
    buy = (
        (df['SMA_FAST'].iloc[i] > df['SMA_SLOW'].iloc[i])
        and (df['SMA_FAST'].iloc[i - 1] <= df['SMA_SLOW'].iloc[i - 1])
        and (df['RSI'].iloc[i] < 65)
    )
    sell = (
        (df['SMA_FAST'].iloc[i] < df['SMA_SLOW'].iloc[i])
        and (df['SMA_FAST'].iloc[i - 1] >= df['SMA_SLOW'].iloc[i - 1])
        and (df['RSI'].iloc[i] > 35)
    )
    if buy:
        signals.append("BUY")
    elif sell:
        signals.append("SELL")
    else:
        signals.append("HOLD")

# تنظیم ردیف‌ها و افزودن سیگنال
df = df.iloc[1:].copy()
df['signal'] = signals

# ذخیره خروجی
df.to_csv(OUT_FILE, index=False)
print(f"✅ Signals generated: {OUT_FILE}")
print(df['signal'].value_counts())
print(df.tail(10)[["time", "close", "SMA_FAST", "SMA_SLOW", "RSI", "signal"]])
