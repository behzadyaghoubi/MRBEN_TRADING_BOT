#!/usr/bin/env python3
"""
Feature Engineering Pro - Add Advanced Technical Indicators
========================================================

این اسکریپت اندیکاتورهای حرفه‌ای و ترکیبی را به دیتافریم اضافه می‌کند:
- Bollinger Bands
- ATR
- Stochastic
- EMA Cross
- Price Action (Pinbar, Engulfing)

Author: MRBEN Trading System
"""

import logging

import numpy as np
import pandas as pd
import talib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = 'lstm_signals_fixed.csv'
OUTPUT_FILE = 'lstm_signals_features.csv'


def add_advanced_features(df):
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle
    df['bb_pos'] = (df['close'] - lower) / (upper - lower)
    # ATR
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    # Stochastic
    k, d = talib.STOCH(df['high'], df['low'], df['close'])
    df['stoch_k'] = k
    df['stoch_d'] = d
    # EMA Cross
    df['ema_fast'] = talib.EMA(df['close'], timeperiod=12)
    df['ema_slow'] = talib.EMA(df['close'], timeperiod=26)
    df['ema_cross'] = (df['ema_fast'] > df['ema_slow']).astype(int)
    # Price Action: Pinbar
    candle_range = df['high'] - df['low']
    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']
    pinbar = (
        (candle_range > 0) & (upper_wick > candle_range * 0.4) & (lower_wick > candle_range * 0.4)
    )
    df['pinbar'] = pinbar.astype(int)
    # Price Action: Engulfing
    engulfing = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    df['engulfing'] = np.where(engulfing > 0, 1, np.where(engulfing < 0, -1, 0))
    # RSI (در صورت نبود)
    if 'RSI' not in df.columns:
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    # MACD (در صورت نبود)
    if 'MACD' not in df.columns:
        macd, macd_sig, macd_hist = talib.MACD(df['close'])
        df['MACD'] = macd
        df['MACD_signal'] = macd_sig
        df['MACD_hist'] = macd_hist
    return df


def main():
    logger.info('Loading data...')
    df = pd.read_csv(INPUT_FILE)
    logger.info(f'Loaded {len(df)} rows.')
    logger.info('Adding advanced features...')
    df = add_advanced_features(df)
    logger.info('Saving to file...')
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f'Features saved to {OUTPUT_FILE}')
    print(df.head())


if __name__ == '__main__':
    main()
