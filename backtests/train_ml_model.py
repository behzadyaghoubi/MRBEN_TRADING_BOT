import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

data_file = "signals_for_ai_training.csv"

if not os.path.exists(data_file):
    raise FileNotFoundError(f"⛔ فایل {data_file} وجود ندارد!")

df = pd.read_csv(data_file)

# فقط رکوردهای بدون مقدار گمشده استفاده شوند
df = df.dropna(subset=["SMA_FAST", "RSI", "pinbar", "engulfing", "target"])

X = df[["SMA_FAST", "RSI", "pinbar", "engulfing"]]
y = df["target"]

# تقسیم داده به آموزش/تست برای ارزیابی کیفیت مدل
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# مدل حرفه‌ای با تنظیمات پایدار
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ارزیابی مدل
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\n🔎 Classification Report:\n", classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

# ذخیره مدل
model_path = "mrben_ai_signal_filter.joblib"
joblib.dump(model, model_path)
print(f"✅ مدل RandomForest با موفقیت ذخیره شد: {model_path}")

# تست بارگذاری و پیش‌بینی (تست کامل fail-safe)
try:
    loaded_model = joblib.load(model_path)
    test_out = loaded_model.predict(X_test.head(5))
    print("پیش‌بینی نمونه:", test_out)
except Exception as e:
    print("⛔ خطا در تست مدل:", e)