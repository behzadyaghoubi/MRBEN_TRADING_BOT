import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# لود دیتا
df = pd.read_csv("XAUUSD_M15_history.csv")  # یا هر دیتای دلخواه
prices = df['close'].values.reshape(-1, 1)

# نرمال‌سازی
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

X = []
y = []
window_size = 20  # تعداد کندل ورودی برای پیش‌بینی

for i in range(len(prices_scaled) - window_size):
    X.append(prices_scaled[i : i + window_size])
    y.append(prices_scaled[i + window_size])

X, y = np.array(X), np.array(y)

# ساخت مدل LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# ذخیره مدل و scaler
model.save("mrben_lstm_model.h5")
joblib.dump(scaler, "mrben_lstm_scaler.save")
print("✅ مدل LSTM و Scaler ذخیره شدند.")
