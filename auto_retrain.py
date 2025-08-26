import pandas as pd
import numpy as np
import logging
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("AutoRetrain")

TRADE_LOG = 'trade_log.csv'
LSTM_MODEL_PATH = 'lstm_trading_model.h5'
ML_FILTER_MODEL_PATH = 'mrben_ai_signal_filter_xgb.joblib'

# 1. Load and preprocess trade log
def load_trade_log():
    if not os.path.exists(TRADE_LOG):
        logger.error(f"{TRADE_LOG} not found.")
        return None
    df = pd.read_csv(TRADE_LOG, header=None)
    # Columns: time, signal, price, sl, tp, result, buy_proba, sell_proba, ...features
    df.columns = ['time', 'signal', 'price', 'sl', 'tp', 'result', 'buy_proba', 'sell_proba', 'atr', 'sl_val', 'tp_val']
    logger.info(f"Loaded {len(df)} trades from log.")
    return df

# 2. Prepare features/labels for LSTM and ML filter
def prepare_lstm_data(df):
    # Example: Use [buy_proba, sell_proba, atr, sl_val, tp_val] as features, result as label
    X = df[['buy_proba', 'sell_proba', 'atr', 'sl_val', 'tp_val']].values
    y = (df['result'] > 0).astype(int).values if 'result' in df.columns and df['result'].notnull().any() else df['signal'].apply(lambda x: 1 if x == 1 else 0).values
    return X, y

def prepare_ml_data(df):
    # Example: Use [buy_proba, sell_proba, atr, sl_val, tp_val, signal] as features, result as label
    X = df[['buy_proba', 'sell_proba', 'atr', 'sl_val', 'tp_val', 'signal']].values
    y = (df['result'] > 0).astype(int).values if 'result' in df.columns and df['result'].notnull().any() else df['signal'].apply(lambda x: 1 if x == 1 else 0).values
    return X, y

# 3. Retrain LSTM model
def retrain_lstm(X, y):
    logger.info("Retraining LSTM model...")
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val), callbacks=[es], verbose=2)
    model.save(LSTM_MODEL_PATH)
    logger.info(f"LSTM model saved to {LSTM_MODEL_PATH}")

# 4. Retrain ML filter
def retrain_ml_filter(X, y):
    logger.info("Retraining ML filter model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_scaled, y)
    joblib.dump(clf, ML_FILTER_MODEL_PATH)
    logger.info(f"ML filter model saved to {ML_FILTER_MODEL_PATH}")
    joblib.dump(scaler, 'ml_filter_scaler.joblib')
    logger.info("Scaler saved to ml_filter_scaler.joblib")

# 5. Main retrain process
def main():
    logger.info("Starting auto-retrain process...")
    df = load_trade_log()
    if df is None or len(df) < 20:
        logger.error("Not enough data to retrain.")
        return
    X_lstm, y_lstm = prepare_lstm_data(df)
    X_ml, y_ml = prepare_ml_data(df)
    retrain_lstm(X_lstm, y_lstm)
    retrain_ml_filter(X_ml, y_ml)
    logger.info("Auto-retrain process completed successfully.")

if __name__ == "__main__":
    main() 