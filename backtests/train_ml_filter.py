import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('lstm_signals_pro.csv')
# ستون‌های features را با توجه به داده‌های مدل خودت تنظیم کن
X = df[['lstm_buy_proba', 'lstm_hold_proba', 'lstm_sell_proba']]
y = (df['lstm_signal'] != 0).astype(int)  # سیگنال خرید یا فروش

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"دقت مدل فیلتر سیگنال: {accuracy_score(y_test, y_pred):.4f}")

import joblib
joblib.dump(model, 'ml_signal_filter_xgb.joblib')