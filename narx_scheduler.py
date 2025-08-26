from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


# دریافت دیتا از متاتریدر
def get_price_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, bars=200):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        print("❌ دریافت دیتا شکست خورد")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["price"] = df["close"]
    return df


# ساخت داده ورودی برای NARX
def prepare_narx_data(df, input_lags=5, output_steps=5):
    prices = df["price"].values
    X, y = [], []
    for i in range(input_lags, len(prices) - output_steps):
        X.append(prices[i - input_lags : i])
        y.append(prices[i : i + output_steps])
    return np.array(X), np.array(y)


# آموزش یا پیش‌بینی مدل
def train_and_predict_narx(df):
    X, y = prepare_narx_data(df)
    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)
    model.fit(X, y)
    last_input = df["price"].values[-5:].reshape(1, -1)
    prediction = model.predict(last_input)[0]
    return prediction


# ذخیره پیش‌بینی
def save_prediction(prediction, file_path="narx_predictions.csv"):
    now = datetime.now()
    df = pd.DataFrame(
        {
            "time": [now + pd.Timedelta(minutes=5 * i) for i in range(len(prediction))],
            "predicted_price": prediction,
        }
    )
    df.to_csv(file_path, index=False)
    print(f"✅ پیش‌بینی NARX ذخیره شد: {file_path}")


# اجرای کامل
def run_narx_forecast():
    if not mt5.initialize():
        print("❌ متاتریدر وصل نشد")
        return
    df = get_price_data()
    if df is not None:
        prediction = train_and_predict_narx(df)
        save_prediction(prediction)


if __name__ == "__main__":
    run_narx_forecast()
