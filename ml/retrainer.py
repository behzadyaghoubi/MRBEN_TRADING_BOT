# ml/retrainer.py

import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

class Retrainer:
    def __init__(self, log_dir="logs/trades", model_path="models/mrben_ai_signal_filter_xgb.joblib"):
        self.log_dir = log_dir
        self.model_path = model_path

    def _load_data(self):
        all_files = glob.glob(os.path.join(self.log_dir, "*.csv"))
        all_data = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

        feature_df = all_data['features'].str.split(";", expand=True)
        feature_df = feature_df.apply(lambda col: col.str.split("=", expand=True).iloc[:,1])
        feature_df.columns = ['RSI', 'MACD', 'ATR', 'tick_volume']
        feature_df = feature_df.astype(float)

        y = all_data['actual_result'].apply(lambda x: 1 if x == "WIN" else 0).astype(int)

        return feature_df, y

    def retrain_model(self):
        X, y = self._load_data()
        n_samples = len(X)

        if n_samples < 2:
            print("⚠️ فقط یک نمونه برای آموزش داریم. مدل با همین یک مورد آموزش داده می‌شود ولی دقت قابل اعتماد نیست.")
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X, y)
            joblib.dump(model, self.model_path)
            print(f"✅ مدل با یک نمونه ذخیره شد: {self.model_path}")
            return None

        # حالت نرمال
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"✅ مدل با دقت: {acc:.2%} آموزش داده شد.")
        joblib.dump(model, self.model_path)
        print(f"✅ مدل جدید ذخیره شد در: {self.model_path}")

        return acc