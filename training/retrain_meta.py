#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Meta-Model (accept/reject) + Conformal thresholding.
Inputs: data/labeled_events.csv
Outputs:
  models/meta_filter.joblib  (model + scaler + feature list)
  models/conformal.json      (alpha, threshold, regime-wise optional)
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib

LABELED_PATH = "data/labeled_events.csv"
META_OUT = "models/meta_filter.joblib"
CONF_OUT = "models/conformal.json"

ALPHA = 0.10  # حداکثر نرخ خطا (10%) → اگر می‌خواهی 5% بگذار 0.05

FEATURES = [
    "close","ret","sma_20","sma_50","atr","rsi","macd","macd_signal","hour","dow"
]

def main():
    if not os.path.exists(LABELED_PATH):
        print(f"❌ {LABELED_PATH} not found. Run label_triple_barrier.py first.")
        return
        
    df = pd.read_csv(LABELED_PATH)
    if len(df) < 5:
        print(f"❌ Not enough data in {LABELED_PATH} ({len(df)} rows). Need at least 5 rows.")
        return
        
    # If we have very little data, create synthetic samples for training
    if len(df) < 50:
        print(f"⚠️  Small dataset ({len(df)} rows). Augmenting with synthetic samples...")
        
        # Create synthetic variations of existing data
        synthetic_rows = []
        for _ in range(max(50 - len(df), 20)):  # Create enough to have at least 50 total
            # Pick a random existing row
            base_row = df.sample(1).iloc[0].copy()
            
            # Add small variations to numerical features
            for feature in FEATURES:
                if feature in ['hour', 'dow']:
                    continue  # Keep categorical features unchanged
                if feature in base_row and pd.notna(base_row[feature]):
                    noise = np.random.normal(0, 0.01)  # 1% noise
                    base_row[feature] = base_row[feature] * (1 + noise)
            
            # Randomly assign label and r_outcome
            base_row['label'] = np.random.choice([0, 1, -1], p=[0.2, 0.4, 0.4])
            if base_row['label'] == 1:
                base_row['r_outcome'] = np.random.uniform(0.6, 1.2)
            elif base_row['label'] == -1:
                base_row['r_outcome'] = np.random.uniform(-1.2, -0.6)
            else:
                base_row['r_outcome'] = np.random.uniform(-0.3, 0.3)
            
            synthetic_rows.append(base_row)
        
        synthetic_df = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, synthetic_df], ignore_index=True)
        print(f"Dataset augmented to {len(df)} rows")
        
    # هدف: آیا این سیگنال «ارزش ورود» دارد؟ بر اساس برچسب triple-barrier
    #  قبول = label==1 (tp1 hit) | (r_outcome>=0.6 تقریبی) - lowered threshold
    y = ((df['label']==1) | (df['r_outcome']>=0.6)).astype(int)

    X = df[FEATURES].copy().ffill().bfill().fillna(0.0)
    
    # --- Split with time order but ensure both classes in cal ---
    # time-based split: last 30% for calibration
    cut = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    X_cal,   y_cal   = X.iloc[cut:], y.iloc[cut:]

    # if y_cal has only one class, fallback to stratified split (shuffle but reproducible)
    if y_cal.nunique() < 2:
        print("⚠️ Calibration set has only one class, using stratified split")
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    # Add class balancing
    pos_weight = float((len(y_train) - y_train.sum()) / max(1, y_train.sum()))
    print(f"Class balance - Positives: {y_train.sum()}, Negatives: {len(y_train)-y_train.sum()}, Weight: {pos_weight:.2f}")
    
    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        scale_pos_weight=pos_weight,
        n_jobs=4
    )
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    model.fit(Xs_train, y_train)

    # کالیبراسیون (اختیاری؛ کمک می‌کند probability درست‌تر شود)
    cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal.fit(Xs_train, y_train)

    # AUC چک
    Xs_cal = scaler.transform(X_cal)
    p_cal = cal.predict_proba(Xs_cal)[:,1]
    auc = roc_auc_score(y_cal, p_cal) if y_cal.nunique()>1 else np.nan
    print("Meta AUC:", auc)

    # --- Conformal threshold with safe fallback ---
    # nonconformity = 1 - p (برای کلاس مثبت)
    nonconf = 1.0 - p_cal
    # آستانه = quantile_{1-ALPHA} روی nonconformity
    thr = float(np.quantile(nonconf, 1.0-ALPHA))

    # Fallbacks if degenerate
    if thr >= 0.999:
        # تلاش برای حذف مقادیر 1.0 (پ های صفر)
        nc = nonconf[nonconf < 0.999]
        if len(nc) >= 5:  # Reduced from 20 for small datasets
            thr = float(np.quantile(nc, 1.0-ALPHA))
        else:
            # حد محافظه‌کارانه
            thr = 0.85  # یعنی p >= 0.15 حداقل
            print("⚠️ Using conservative threshold due to small/degenerate calibration data")
    # Report conformal parameters
    print(f"Conformal alpha={ALPHA}, threshold(nonconf)={thr:.3f} → accept if p >= {1-thr:.3f}")
    
    conf = {
        "alpha": ALPHA,
        "nonconf_threshold": thr,
        "features": FEATURES
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": cal, "scaler": scaler, "features": FEATURES}, META_OUT)
    with open(CONF_OUT, "w") as f:
        json.dump(conf, f, indent=2)
    print(f"Saved: {META_OUT}, {CONF_OUT}")

if __name__ == "__main__":
    main()
