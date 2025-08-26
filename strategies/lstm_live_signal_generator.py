import numpy as np
import pandas as pd

from lstm_trading_system_pro import DataPreprocessor, LSTMModel, SignalGenerator, TradingConfig

# بارگذاری مدل و ابزارها فقط یک بار (برای سرعت)
config = TradingConfig()
preprocessor = DataPreprocessor(config)
lstm_model = LSTMModel(config)
lstm_model.load_model("lstm_trading_model.h5")
signal_generator = SignalGenerator(config)


# تابع تولید سیگنال لایو
# ورودی: df دیتافریم قیمت (OHLCV)
# خروجی: سیگنال عددی (2=BUY, 1=HOLD, 0=SELL)
def generate_lstm_live_signal(df: pd.DataFrame) -> int:
    # آماده‌سازی داده و ویژگی‌ها
    prepared = preprocessor.prepare_data(df)
    # ساخت دنباله آخر
    X, _ = preprocessor.create_sequences(prepared)
    if len(X) == 0:
        return 1  # HOLD اگر داده کافی نبود
    # فقط آخرین دنباله را پیش‌بینی کن
    pred = lstm_model.predict(X[-1:])
    signal = int(np.argmax(pred[0]))  # 2=BUY, 1=HOLD, 0=SELL
    return signal
