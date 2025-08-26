import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# بارگذاری داده‌های ورودی برای سیگنال‌دهی (همون X_lstm.npy)
X = np.load('X_lstm.npy')

# بارگذاری مدل آموزش دیده
model = load_model('lstm_trading_model.h5')

# پیش‌بینی احتمال سیگنال خرید (1) یا فروش (0)
proba = model.predict(X).flatten()

# تبدیل احتمال به سیگنال
signals = (proba > 0.5).astype(int)

# بارگذاری داده بازار برای اضافه کردن سیگنال‌ها
df = pd.read_csv('ohlc_data.csv')

# هم اندازه کردن دیتا با سیگنال‌ها
df = df.iloc[-len(signals) :].copy()

# اضافه کردن ستون‌های سیگنال و احتمال
df['lstm_proba'] = proba
df['lstm_signal'] = signals

# ذخیره نتایج
df.to_csv('lstm_signals.csv', index=False)

print("✅ سیگنال‌های LSTM تولید و ذخیره شد: lstm_signals.csv")
