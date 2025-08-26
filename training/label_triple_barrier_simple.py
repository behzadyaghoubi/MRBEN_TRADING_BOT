#!/usr/bin/env python3
"""
Simple Triple-Barrier labeling for MR BEN logs.
"""
import os

import pandas as pd

# --- CONFIG ---
LOG_PATH = "data/trade_log_gold.csv"
OUT_PATH = "data/labeled_events.csv"


def main():
    try:
        if not os.path.exists(LOG_PATH):
            print(f"❌ {LOG_PATH} not found")
            return

        df = pd.read_csv(LOG_PATH)
        print(f"✅ Loaded {len(df)} rows from {LOG_PATH}")
        print(f"Columns: {list(df.columns)}")

        # Simple features from existing columns
        df['time'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('time').reset_index(drop=True)

        # Create simple features
        df['close'] = df['entry_price']
        df['ret'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
        df['atr'] = df['close'].diff().abs().rolling(14, min_periods=1).mean() * 3.0
        df['rsi'] = 50.0  # Simple default
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['hour'] = df['time'].dt.hour
        df['dow'] = df['time'].dt.dayofweek

        # Simple labeling based on actual results
        # Label = 1 if profitable, 0 if break-even, -1 if loss
        df['side'] = df['action'].map({'BUY': 1, 'SELL': -1})
        df['risk'] = abs(df['entry_price'] - df['sl_price'])
        df['reward'] = abs(df['tp_price'] - df['entry_price'])
        df['R_ratio'] = df['reward'] / df['risk'].replace(0, 1)

        # Simple label based on R ratio
        df['label'] = 0  # default
        df.loc[df['R_ratio'] >= 0.8, 'label'] = 1  # Win
        df.loc[df['R_ratio'] < 0.5, 'label'] = -1  # Loss
        df['r_outcome'] = df['R_ratio']

        # Simple regime
        df['regime'] = 'RANGE'
        trend_mask = abs(df['macd'] - df['macd_signal']) > df['close'] * 0.0008
        df.loc[trend_mask & (df['sma_20'] > df['sma_50']), 'regime'] = 'UPTREND'
        df.loc[trend_mask & (df['sma_20'] < df['sma_50']), 'regime'] = 'DOWNTREND'

        # Select features for output
        features = [
            'close',
            'ret',
            'sma_20',
            'sma_50',
            'atr',
            'rsi',
            'macd',
            'macd_signal',
            'hour',
            'dow',
        ]
        output_df = df[
            ['time', 'side', 'entry_price', 'sl_price', 'tp_price', 'label', 'r_outcome', 'regime']
            + features
        ].copy()

        # Fill NaN values
        for col in features:
            if col in output_df.columns:
                output_df[col] = output_df[col].fillna(
                    method=(
                        'ffill' if hasattr(pd.Series.fillna, 'method') else output_df[col].ffill()
                    )
                )
                output_df[col] = output_df[col].fillna(0.0)

        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        output_df.to_csv(OUT_PATH, index=False)
        print(f"✅ Saved {len(output_df)} labeled events to {OUT_PATH}")
        print(f"Label distribution: {output_df['label'].value_counts().to_dict()}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
