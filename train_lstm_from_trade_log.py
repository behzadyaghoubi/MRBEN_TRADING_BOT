import pandas as pd
import numpy as np
import os
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# --- Parameters ---
LOOKBACK = 10
MODEL_PATH = 'models/mrben_lstm_model.h5'
SCALER_PATH = 'models/mrben_lstm_scaler.save'
TRADE_LOG = 'data/trade_log_clean.csv'

# --- Load Data ---
df = pd.read_csv(TRADE_LOG, header=None)

# --- Infer columns ---
def infer_columns(df):
    # Try to infer columns by length and value
    n_cols = df.shape[1]
    # Typical: time, signal, price, sl, tp, result, buy_proba, sell_proba, ...
    base = ['time', 'signal', 'price', 'sl', 'tp', 'result', 'buy_proba', 'sell_proba']
    extra = [f'feature_{i}' for i in range(n_cols - len(base))]
    columns = base + extra
    return columns[:n_cols]

df.columns = infer_columns(df)

# --- Feature selection ---
feature_candidates = ['price', 'sl', 'tp', 'buy_proba', 'sell_proba']
feature_candidates += [col for col in df.columns if col.startswith('feature_') or col in ['ATR']]
features = [col for col in feature_candidates if col in df.columns]
label_col = 'signal'

# --- Clean and preprocess ---
df = df[features + [label_col]].copy()
df = df.dropna()

# --- Normalize features ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features].values)
joblib.dump(scaler, SCALER_PATH)

# --- Sequence generation ---
X_seq = []
y_seq = []
for i in range(LOOKBACK, len(X_scaled)):
    X_seq.append(X_scaled[i-LOOKBACK:i])
    y_seq.append(df[label_col].iloc[i])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# --- Convert labels to 0,1,2 (for -1,0,1) ---
label_map = {-1: 0, 0: 1, 1: 2}
y_seq = np.array([label_map.get(int(x), 1) for x in y_seq])

# --- Train/test split ---
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

# --- Model ---
model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(LOOKBACK, X_seq.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Train ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)],
    verbose=2
)

# --- Save model ---
os.makedirs('models', exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ LSTM model saved to {MODEL_PATH}")

# --- Plot history ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend(); plt.title('Accuracy')
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(); plt.title('Loss')
plt.tight_layout()
plt.savefig('models/mrben_lstm_training_history.png')
plt.close()
print("✅ Training history plot saved.") 