#!/usr/bin/env python3
"""
Build Balanced LSTM Training Dataset
===================================

این اسکریپت دیتاست آموزش LSTM را به صورت متعادل (BUY/SELL/HOLD برابر) می‌سازد.

Author: MRBEN Trading System
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# پارامترهای برچسب‌گذاری
BUY_THRESHOLD = 0.002  # 0.2% رشد (کاهش آستانه)
SELL_THRESHOLD = -0.002  # 0.2% افت (کاهش آستانه)
LOOKAHEAD = 10  # 10 کندل جلوتر (افزایش lookahead)

INPUT_FILE = 'lstm_signals_features.csv'
OUTPUT_FILE = 'lstm_train_data_balanced.csv'


def label_row(df, idx, lookahead=LOOKAHEAD):
    if idx + lookahead >= len(df):
        return None
    price_now = df.iloc[idx]['close']
    price_future = df.iloc[idx + lookahead]['close']
    change = (price_future - price_now) / price_now
    if change > BUY_THRESHOLD:
        return 2  # BUY
    elif change < SELL_THRESHOLD:
        return 0  # SELL
    else:
        return 1  # HOLD


def build_balanced_dataset():
    logger.info('Loading data...')
    df = pd.read_csv(INPUT_FILE)
    logger.info(f'Loaded {len(df)} rows.')

    # برچسب‌گذاری
    logger.info('Labeling data...')
    labels = [label_row(df, i) for i in range(len(df))]
    df['label'] = labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # شمارش اولیه
    counts = df['label'].value_counts()
    logger.info(f'Initial label counts: {counts.to_dict()}')
    min_count = counts.min()
    max_count = counts.max()

    # اگر اختلاف زیاد بود، oversample برای کلاس‌های کم
    logger.info('Balancing dataset (with oversampling if needed)...')
    dfs = []
    for label in [0, 1, 2]:
        class_df = df[df['label'] == label]
        if len(class_df) < max_count:
            class_df = class_df.sample(max_count, replace=True, random_state=42)
        dfs.append(class_df)
    balanced_df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    # شمارش نهایی
    logger.info(f'Balanced label counts: {balanced_df["label"].value_counts().to_dict()}')

    # ذخیره دیتاست
    balanced_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f'Balanced dataset saved to {OUTPUT_FILE}')
    print(balanced_df['label'].value_counts())
    print(balanced_df.head())


if __name__ == '__main__':
    build_balanced_dataset()
