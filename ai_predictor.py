import pandas as pd
import joblib
import json

# بارگذاری مدل
model = joblib.load("ai_signal_model.pkl")

# بارگذاری آخرین سیگنال
with open("latest_signal.json") as f:
    signal = json.load(f)

# ساخت دیتافریم برای پیش‌بینی
features = pd.DataFrame([{
    'confidence': signal.get('confidence', 50),
    'rsi': signal.get('rsi', 50),
    'macd': signal.get('macd', 0),
    'z_score': signal.get('z_score', 0)
}])

# پیش‌بینی احتمال موفقیت سیگنال
prob = model.predict_proba(features)[0][1] * 100
label = "موفق" if prob >= 60 else "ضعیف"

print(f"📌 سیگنال: {signal['signal']} @ {signal['price']}")
print(f"🤖 احتمال موفقیت (AI): {prob:.2f}% → {label}")
