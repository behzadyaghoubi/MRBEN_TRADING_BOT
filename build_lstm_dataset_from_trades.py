import pandas as pd
import numpy as np
from lstm_trading_system_pro import TradingConfig, DataPreprocessor

# فایل ورودی و خروجی
INPUT_CSV = "ai_trading_results.csv"
X_OUT = "X_lstm.npy"
y_OUT = "y_lstm.npy"

# بارگذاری داده
df = pd.read_csv(INPUT_CSV)

# بارگذاری کانفیگ پیش‌فرض (یا از فایل)
config = TradingConfig()
preprocessor = DataPreprocessor(config)

# آماده‌سازی داده و ساخت ویژگی‌ها
prepared_df = preprocessor.prepare_data(df)

# ساخت دنباله‌های زمانی و لیبل‌ها
X, y = preprocessor.create_sequences(prepared_df)

# ذخیره خروجی
np.save(X_OUT, X)
np.save(y_OUT, y)
print(f"✅ X shape: {X.shape}, y shape: {y.shape} ذخیره شد!") 