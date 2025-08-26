import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- تنظیمات ---
WINDOW_PAST = 20     # تعداد کندل‌های قبلی برای ورودی مدل
WINDOW_EXOG = 20     # تعداد کندل‌های اندیکاتور قبلی
FUTURE_STEPS = 5     # تعداد کندل آینده برای پیش‌بینی

# --- لود دیتا و ساخت اندیکاتورهای پایه ---
df = pd.read_csv("XAUUSD_M15_history.csv")   # یا هر دیتای دلخواه

# اندیکاتورهای نمونه (برای حرفه‌ای‌تر شدن، اندیکاتورهای بیشتری اضافه کن)
df['ema_20'] = df['close'].ewm(span=20).mean()
df['ema_50'] = df['close'].ewm(span=50).mean()
df['ema_cross'] = (df['ema_20'] > df['ema_50']).astype(int)  # 1=خرید، 0=فروش یا خنثی

# --- ساخت دیتاست به سبک NARX Multi-Step ---
features = []
targets = []

for i in range(max(WINDOW_PAST, WINDOW_EXOG), len(df) - FUTURE_STEPS):
    price_past = df['close'].values[i - WINDOW_PAST:i]
    exog_past = df['ema_cross'].values[i - WINDOW_EXOG:i]
    X = np.concatenate([price_past, exog_past])
    y = df['close'].values[i:i + FUTURE_STEPS]  # 5 قیمت آینده
    features.append(X)
    targets.append(y)

features = np.array(features)
targets = np.array(targets)

# --- نرمال‌سازی دیتا ---
scaler_X = MinMaxScaler()
features_scaled = scaler_X.fit_transform(features)

scaler_y = MinMaxScaler()
targets_scaled = scaler_y.fit_transform(targets)

# --- ساخت مدل شبکه عصبی Multi-Output ---
model = Sequential()
model.add(InputLayer(input_shape=(features_scaled.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(FUTURE_STEPS))  # خروجی: 5 کندل آینده

model.compile(optimizer='adam', loss='mse')
model.fit(features_scaled, targets_scaled, epochs=20, batch_size=32, verbose=1)

# --- ذخیره مدل و اسکیلرها ---
model.save("mrben_narx_multistep_model.h5")
joblib.dump(scaler_X, "mrben_narx_multistep_scaler_X.save")
joblib.dump(scaler_y, "mrben_narx_multistep_scaler_y.save")
print(f"✅ مدل NARX Multi-Step و اسکیلرها ذخیره شدند ({FUTURE_STEPS} کندل آینده).")