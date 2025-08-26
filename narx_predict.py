import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib

# پارامترها (دقیقاً مثل زمان آموزش)
WINDOW_PAST = 20
WINDOW_EXOG = 20
FUTURE_STEPS = 5

# بارگذاری مدل و اسکیلرها (سازگار با Keras 3)
model = load_model("mrben_narx_multistep_model.h5", compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError())
scaler_X = joblib.load("mrben_narx_multistep_scaler_X.save")
scaler_y = joblib.load("mrben_narx_multistep_scaler_y.save")

def narx_predict(df):
    """
    پیش‌بینی ۵ کندل آینده با مدل NARX Multi-Step
    ورودی: df شامل ستون‌های 'close', 'ema_cross'
    خروجی: لیست قیمت ۵ کندل آینده یا None اگر دیتا کافی نباشد
    """
    if len(df) < max(WINDOW_PAST, WINDOW_EXOG):
        return None
    price_past = df['close'].values[-WINDOW_PAST:]
    exog_past = df['ema_cross'].values[-WINDOW_EXOG:]
    X = np.concatenate([price_past, exog_past]).reshape(1, -1)
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return y_pred.flatten()  # خروجی: ۵ عدد قیمت آینده

# --- تست سریع برای اطمینان ---
if __name__ == "__main__":
    df = pd.read_csv("XAUUSD_M15_history.csv")
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_cross'] = (df['ema_20'] > df['ema_50']).astype(int)
    pred = narx_predict(df)
    print("پیش‌بینی ۵ کندل آینده:", pred)