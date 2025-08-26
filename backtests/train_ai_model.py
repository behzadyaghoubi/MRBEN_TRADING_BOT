# train_ai_model.py
import os

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TRAIN_FILE = "signals_for_ai_training.csv"  # دیتای بک‌تست
REAL_FILE = "live_trades_log.csv"  # دیتای واقعی (اگر وجود داشته باشد)
MODEL_FILE = "mrben_ai_signal_filter_xgb.joblib"

# دیتای آموزش را ادغام کن (ترجیحاً دیتای واقعی را اضافه کن)
dfs = []
if os.path.exists(TRAIN_FILE):
    dfs.append(pd.read_csv(TRAIN_FILE))
if os.path.exists(REAL_FILE):
    df_real = pd.read_csv(REAL_FILE)
    # فرض: در دیتای واقعی ستون target وجود دارد؛ اگر ندارد بساز
    if "target" not in df_real.columns and "result" in df_real.columns:
        df_real["target"] = (df_real["result"] > 0).astype(int)
    dfs.append(df_real[["SMA_FAST", "RSI", "pinbar", "engulfing", "target"]].dropna())

if not dfs:
    print("[ERROR] No train data found!")
    exit()

data = pd.concat(dfs, ignore_index=True)
data = data.dropna()
X = data[["SMA_FAST", "RSI", "pinbar", "engulfing"]]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.7)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

joblib.dump(model, MODEL_FILE)
print(f"[OK] Model saved: {MODEL_FILE}")
