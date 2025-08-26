import numpy as np
import pandas as pd

# پارامتر طول سکانس (مثلاً 50 کندل قبل برای پیش‌بینی بعدی)
window_size = 50

# بارگذاری داده برچسب‌گذاری شده
df = pd.read_csv("lstm_train_data.csv")

# ستون‌های ورودی (مثلاً فقط قیمت بسته شدن و حجم)
features = df[['close', 'tick_volume']].values
labels = df['target'].values

X = []
y = []

for i in range(len(df) - window_size):
    X.append(features[i:i+window_size])
    y.append(labels[i + window_size])

X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ذخیره سکانس‌ها برای آموزش مدل
np.save('X_lstm.npy', X)
np.save('y_lstm.npy', y)

print("✅ سکانس‌های LSTM آماده و ذخیره شد: X_lstm.npy و y_lstm.npy")