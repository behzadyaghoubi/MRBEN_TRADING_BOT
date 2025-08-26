import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# فایل دیتاست
CSV_FILE = "mrben_ai_signal_dataset.csv"
MODEL_FILE = "mrben_ai_signal_filter_xgb.joblib"

# خواندن دیتاست
df = pd.read_csv(CSV_FILE)

# آماده‌سازی ورودی و خروجی
features = ["SMA_FAST", "SMA_SLOW", "RSI", "MACD", "MACD_signal", "MACD_hist"]
X = df[features].values
y = df["signal"].values

# تبدیل سیگنال به عددی
le = LabelEncoder()
y_num = le.fit_transform(y)  # BUY=0, HOLD=1, SELL=2 (مثلا)

# جداکردن داده آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y_num, test_size=0.2, random_state=42, stratify=y_num
)

# مدل XGBoost
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

# ارزیابی مدل
acc = model.score(X_test, y_test)
print(f"🎯 دقت مدل روی داده تست: {acc:.2f}")

# ذخیره مدل
joblib.dump(model, MODEL_FILE)
print(f"✅ مدل آموزش‌دیده ذخیره شد: {MODEL_FILE}")

# ذخیره Encoder (برای تفسیر برچسب‌ها)
joblib.dump(le, "signal_label_encoder.joblib")
