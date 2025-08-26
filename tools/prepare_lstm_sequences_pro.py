import numpy as np
import pandas as pd

df = pd.read_csv("lstm_train_data_pro.csv")

window_size = 50

features = [
    'close',
    'open',
    'high',
    'low',
    'tick_volume',
    'real_volume',
    'spread',
]  # اگر volume یا spread نداری حذفش کن
X = []
y = []

for i in range(len(df) - window_size):
    feat_seq = df[features].iloc[i : i + window_size].values
    label = df['target'].iloc[i + window_size - 1]
    # فقط حرکات 1 یا -1 (ترید) یا اگر خواستی هر سه رو نگه دار
    X.append(feat_seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save("X_lstm_pro.npy", X)
np.save("y_lstm_pro.npy", y)
print("✅ سکانس‌های حرفه‌ای LSTM آماده و ذخیره شد: X_lstm_pro.npy و y_lstm_pro.npy")
