import joblib
import pandas as pd

df = pd.read_csv('lstm_signals_pro.csv')
model = joblib.load('ml_signal_filter_xgb.joblib')

# فیلتر کردن سیگنال‌ها
df['filtered_signal'] = 0
for i, row in df.iterrows():
    prob_features = row[['lstm_buy_proba', 'lstm_hold_proba', 'lstm_sell_proba']].values.reshape(
        1, -1
    )
    pred = model.predict(prob_features)
    if pred[0] == 1:
        df.at[i, 'filtered_signal'] = row['lstm_signal']
    else:
        df.at[i, 'filtered_signal'] = 0  # فیلتر شده و HOLD

df.to_csv('lstm_filtered_signals.csv', index=False)
print("✅ سیگنال‌های فیلتر شده ذخیره شدند: lstm_filtered_signals.csv")
