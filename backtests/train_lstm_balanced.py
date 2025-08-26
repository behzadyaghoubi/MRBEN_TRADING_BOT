#!/usr/bin/env python3
"""
Train LSTM Model on Balanced Dataset
===================================

این اسکریپت مدل LSTM را با دیتاست متعادل آموزش می‌دهد و مدل را ذخیره می‌کند.

Author: MRBEN Trading System
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = 'lstm_train_data_balanced.csv'
MODEL_FILE = 'outputs/lstm_trading_model_balanced.h5'
SCALER_FILE = 'outputs/lstm_scaler_balanced.save'
LOOKBACK = 60
EPOCHS = 50
BATCH_SIZE = 32

FEATURES = [
    'open',
    'high',
    'low',
    'close',
    'tick_volume',
    'lstm_buy_proba',
    'lstm_hold_proba',
    'lstm_sell_proba',
    'bb_upper',
    'bb_middle',
    'bb_lower',
    'bb_width',
    'bb_pos',
    'atr',
    'stoch_k',
    'stoch_d',
    'ema_fast',
    'ema_slow',
    'ema_cross',
    'pinbar',
    'engulfing',
    'RSI',
    'MACD',
    'MACD_signal',
    'MACD_hist',
]


def create_sequences(df, lookback=LOOKBACK):
    X, y = [], []
    scaler = MinMaxScaler()
    data = df[FEATURES].values
    data_scaled = scaler.fit_transform(data)
    labels = df['label'].values
    for i in range(lookback, len(df)):
        X.append(data_scaled[i - lookback : i])
        y.append(labels[i])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler


def build_lstm_model(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(32))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    logger.info('Loading balanced dataset...')
    df = pd.read_csv(DATA_FILE)
    logger.info(f'Dataset shape: {df.shape}')
    X, y, scaler = create_sequences(df)
    logger.info(f'X shape: {X.shape}, y shape: {y.shape}')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info('Building LSTM model...')
    model = build_lstm_model((X.shape[1], X.shape[2]))
    logger.info('Training model...')
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    logger.info('Saving model and scaler...')
    model.save(MODEL_FILE)
    import joblib

    joblib.dump(scaler, SCALER_FILE)
    logger.info(f'Model saved to {MODEL_FILE}')
    logger.info(f'Scaler saved to {SCALER_FILE}')
    # ارزیابی مدل
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy on test set: {acc*100:.2f}%')
    # توزیع پیش‌بینی مدل روی تست
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print('Predicted label distribution:', pd.Series(y_pred).value_counts())
    print('True label distribution:', pd.Series(y_test).value_counts())


if __name__ == '__main__':
    main()
