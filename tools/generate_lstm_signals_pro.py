import numpy as np
import pandas as pd
from tensorflow import keras

# لود مدل و دیتا
model = keras.models.load_model("lstm_trading_model_pro.h5")
X = np.load("X_lstm_pro.npy")
df = pd.read_csv("lstm_train_data_pro.csv")  # شامل time و close و بقیه اطلاعات

# تولید پیش‌بینی
probas = model.predict(X)
signals = np.argmax(probas, axis=1) - 1  # تبدیل 0,1,2 به -1,0,1

# ساخت دیتافریم سیگنال‌ها
results = df.iloc[-len(signals):].copy()  # فرض بر اینکه دیتافریم و سکانس‌ها هم‌اندازه‌اند
results['lstm_buy_proba'] = probas[:, 2]   # احتمال کلاس Buy (+1)
results['lstm_hold_proba'] = probas[:, 1]  # احتمال کلاس Hold (0)
results['lstm_sell_proba'] = probas[:, 0]  # احتمال کلاس Sell (-1)
results['lstm_signal'] = signals

# فقط ستون‌های مهم را نگه دار
results = results[['time', 'close', 'lstm_buy_proba', 'lstm_hold_proba', 'lstm_sell_proba', 'lstm_signal']]

# ذخیره خروجی
results.to_csv("lstm_signals_pro.csv", index=False)
print("✅ سیگنال‌های حرفه‌ای LSTM تولید و ذخیره شد (lstm_signals_pro.csv)")
print(results.tail(10))