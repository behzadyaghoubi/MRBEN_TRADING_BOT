import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# --- فایل‌ها را بخوانید ---
signals = pd.read_csv("final_signal.csv")
trades = pd.read_csv("trades_log.csv")

# --- تبدیل ستون زمان به datetime ---
signals['timestamp'] = pd.to_datetime(signals['time'])
trades['timestamp'] = pd.to_datetime(trades['timestamp'])

# --- اتصال نزدیک‌ترین معامله به سیگنال ---
merged = pd.merge_asof(
    signals.sort_values('timestamp'),
    trades.sort_values('timestamp'),
    on='timestamp',
    direction='forward',
    tolerance=pd.Timedelta('3h'),
)

# --- حذف مواردی که سود ثبت نشده ---
merged = merged.dropna(subset=['profit'])

# --- برچسب موفقیت سیگنال ---
merged['label'] = merged['profit'].apply(lambda x: 1 if x > 0 else 0)

# --- انتخاب ویژگی‌ها ---
features_list = ['confidence']  # ویژگی پایه
for col in ['rsi', 'macd', 'z_score']:
    if col in merged.columns:
        features_list.append(col)
features = merged[features_list].copy()
labels = merged['label']

print(f"Selected features: {features_list}")

# --- آموزش مدل XGBoost ---
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)
model = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss'
)
model.fit(X_train, y_train)

# --- ارزیابی مدل ---
y_pred = model.predict(X_test)
print("\n📊 Advanced AI Model Performance:")
print(classification_report(y_test, y_pred))

# --- ذخیره مدل آموزش‌دیده ---
joblib.dump(model, "mrben_ai_signal_filter_xgb.joblib")
print("✅ AI model saved to mrben_ai_signal_filter_xgb.joblib")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
