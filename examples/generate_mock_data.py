from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# تنظیمات کلی
num_candles = 3000  # حدودا یک ماه M15 = ~21 روز معاملاتی
start_time = datetime.now() - timedelta(minutes=15 * num_candles)
timestamps = [start_time + timedelta(minutes=15 * i) for i in range(num_candles)]

# قیمت اولیه
price = 1950.0
prices = []

# تولید قیمت‌ها
for _ in range(num_candles):
    open_price = price
    high_price = open_price + np.random.uniform(0.5, 2.0)
    low_price = open_price - np.random.uniform(0.5, 2.0)
    close_price = np.random.uniform(low_price, high_price)
    volume = np.random.uniform(100, 1000)
    prices.append([open_price, high_price, low_price, close_price, volume])
    price = close_price  # ادامه قیمت بعدی

# DataFrame نهایی
df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close', 'tick_volume'])
df['timestamp'] = timestamps
df = df[['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']]

# ذخیره به فایل CSV
df.to_csv("mock_xauusd_m15.csv", index=False)
print("✅ فایل mock_xauusd_m15.csv با موفقیت ساخته شد.")
