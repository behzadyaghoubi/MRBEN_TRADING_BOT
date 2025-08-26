import pandas as pd
import talib

# نام فایل ورودی و خروجی
INPUT_CSV = "XAUUSD_PRO_M15_history.csv"
OUTPUT_CSV = "mrben_ai_signal_dataset.csv"

# خواندن دیتای تاریخچه
df = pd.read_csv(INPUT_CSV)

# محاسبه اندیکاتورهای تکنیکال (نمونه، می‌تونی هرچی خواستی اضافه کن)
df["SMA_FAST"] = talib.SMA(df["close"], timeperiod=20)
df["SMA_SLOW"] = talib.SMA(df["close"], timeperiod=50)
df["RSI"] = talib.RSI(df["close"], timeperiod=14)
macd, macd_signal, macd_hist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
df["MACD"] = macd
df["MACD_signal"] = macd_signal
df["MACD_hist"] = macd_hist

# تولید سیگنال ساده (مثال: کراس SMA)
signals = []
for i in range(1, len(df)):
    if df["SMA_FAST"].iloc[i] > df["SMA_SLOW"].iloc[i] and df["SMA_FAST"].iloc[i-1] <= df["SMA_SLOW"].iloc[i-1]:
        signals.append("BUY")
    elif df["SMA_FAST"].iloc[i] < df["SMA_SLOW"].iloc[i] and df["SMA_FAST"].iloc[i-1] >= df["SMA_SLOW"].iloc[i-1]:
        signals.append("SELL")
    else:
        signals.append("HOLD")
signals = ["HOLD"] + signals  # اولین کندل چون قبلی نداره

df["signal"] = signals

# فقط ستون‌های موردنیاز برای آموزش ML را نگه‌دار
df_ml = df[["time", "open", "high", "low", "close", "SMA_FAST", "SMA_SLOW", "RSI", "MACD", "MACD_signal", "MACD_hist", "signal"]]

# حذف ردیف‌هایی که اندیکاتور ندارند (NaN)
df_ml = df_ml.dropna().reset_index(drop=True)

# ذخیره دیتاست نهایی
df_ml.to_csv(OUTPUT_CSV, index=False)
print(f"✅ دیتاست آموزش ML ساخته شد: {OUTPUT_CSV} ({len(df_ml)} ردیف)")