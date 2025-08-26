import pandas as pd
import numpy as np
import talib

# 1. بارگذاری دیتای سیگنال LSTM
df = pd.read_csv('ohlc_lstm_signals.csv')

# 2. اضافه کردن اندیکاتورهای مهم
df['SMA20'] = talib.SMA(df['close'], timeperiod=20)
df['SMA50'] = talib.SMA(df['close'], timeperiod=50)
df['RSI'] = talib.RSI(df['close'], timeperiod=14)
macd, macd_sig, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['MACD'] = macd
df['MACD_signal'] = macd_sig
df['MACD_hist'] = macd_hist

# 3. تارگت (هدف) برای یادگیری: حرکت بعدی بازار
df['future_close'] = df['close'].shift(-1)
df['target'] = np.where(df['future_close'] > df['close'], 1, 0)  # اگر کندل بعدی رشد کرد = 1 (BUY)

# 4. حذف مقادیر نامعتبر
df = df.dropna().reset_index(drop=True)

# 5. ذخیره دیتا برای آموزش ML
df.to_csv("ml_training_data.csv", index=False)
print("✅ دیتا برای آموزش ML Filter ذخیره شد:", df.shape)
print(df[['close', 'lstm_signal', 'RSI', 'MACD', 'SMA20', 'target']].tail(5))