import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import talib

# --- بارگذاری داده ---
df = pd.read_csv("ohlc_data.csv")

def prepare_features(df):
    data = df.copy()
    data['SMA_20'] = talib.SMA(data['close'], timeperiod=20)
    data['SMA_50'] = talib.SMA(data['close'], timeperiod=50)
    data['EMA_12'] = talib.EMA(data['close'], timeperiod=12)
    data['EMA_26'] = talib.EMA(data['close'], timeperiod=26)
    data['RSI'] = talib.RSI(data['close'], timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(data['close'])
    data['MACD'] = macd
    data['MACD_signal'] = macd_signal
    data['MACD_hist'] = macd_hist
    bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'])
    data['BB_upper'] = bb_upper
    data['BB_middle'] = bb_middle
    data['BB_lower'] = bb_lower
    data['BB_width'] = (bb_upper - bb_lower) / bb_middle
    stoch_k, stoch_d = talib.STOCH(data['high'], data['low'], data['close'])
    data['Stoch_K'] = stoch_k
    data['Stoch_D'] = stoch_d
    data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    data['price_change'] = data['close'].pct_change()
    data['price_vs_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
    data['price_vs_sma50'] = (data['close'] - data['SMA_50']) / data['SMA_50']
    data['bb_position'] = (data['close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
    data['volatility_10'] = data['price_change'].rolling(10).std()
    data['atr_ratio'] = data['ATR'] / data['close']
    data = data.dropna().reset_index(drop=True)
    return data

df = prepare_features(df)

lookback = 60
horizon = 5
X, y = [], []
feature_columns = [
    'open', 'high', 'low', 'close', 'tick_volume',
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_width', 'Stoch_K', 'Stoch_D', 'ATR',
    'price_change', 'price_vs_sma20', 'price_vs_sma50',
    'bb_position', 'volatility_10', 'atr_ratio'
]
for i in range(lookback, len(df) - horizon):
    X.append(df[feature_columns].iloc[i-lookback:i].values)
    future_return = (df['close'].iloc[i + horizon] - df['close'].iloc[i]) / df['close'].iloc[i]
    if future_return > 0.005:
        y.append(2)  # BUY
    elif future_return < -0.005:
        y.append(0)  # SELL
    else:
        y.append(1)  # HOLD
X = np.array(X)
y = np.array(y)

scaler = MinMaxScaler()
X_shape = X.shape
X_reshaped = X.reshape(-1, X_shape[2])
X_scaled = scaler.fit_transform(X_reshaped).reshape(X_shape)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = keras.Sequential()
model.add(keras.layers.LSTM(100, return_sequences=True, input_shape=(lookback, X.shape[2])))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(25, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=2
)
model.save("lstm_trading_model.h5")
joblib.dump(scaler, "lstm_scaler.save")
print("✅ مدل سه‌کلاسه LSTM و اسکیلر ذخیره شدند.")
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"🎯 دقت مدل روی داده اعتبارسنجی: {val_acc:.3f}")
import collections
print("توزیع کلاس‌ها در y:", collections.Counter(y))