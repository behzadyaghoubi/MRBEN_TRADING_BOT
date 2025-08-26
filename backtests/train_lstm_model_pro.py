import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# لود داده‌ها
X = np.load("X_lstm_pro.npy")
y = np.load("y_lstm_pro.npy")

# برچسب‌ها را به صورت one-hot برای classification سه‌کلاسه
num_classes = 3
y_categorical = keras.utils.to_categorical(y + 1, num_classes=num_classes)  # چون -1,0,1 داریم

# تقسیم داده‌ها
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y_categorical[:split], y_categorical[split:]

# ساخت مدل LSTM حرفه‌ای
model = keras.Sequential(
    [
        layers.Input(shape=(X.shape[1], X.shape[2])),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# آموزش
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# ذخیره مدل
model.save("lstm_trading_model_pro.h5")
print("✅ مدل LSTM حرفه‌ای ذخیره شد: lstm_trading_model_pro.h5")

# خلاصه پیش‌بینی تست
probas = model.predict(X_val)
print("Min:", np.min(probas), "| Max:", np.max(probas), "| Mean:", np.mean(probas))
