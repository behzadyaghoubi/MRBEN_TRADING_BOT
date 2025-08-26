# build_ml_training_dataset_from_backtest.py - FINAL PROFESSIONAL VERSION

import pandas as pd
import talib

# --- Backtest Parameters ---
TP = 20  # Take Profit (price units/pips)
SL = 20  # Stop Loss (price units/pips)
LOOKAHEAD = 5  # Number of candles to look ahead


def generate_signals(df):
    """
    Generate simple technical signals: SMA + RSI strategy.
    """
    df['SMA_FAST'] = talib.SMA(df['close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['signal'] = "HOLD"
    for i in range(20, len(df) - LOOKAHEAD):
        if df['close'].iloc[i] > df['SMA_FAST'].iloc[i] and df['RSI'].iloc[i] < 70:
            df.at[i, 'signal'] = "BUY"
        elif df['close'].iloc[i] < df['SMA_FAST'].iloc[i] and df['RSI'].iloc[i] > 30:
            df.at[i, 'signal'] = "SELL"
    return df


def backtest_labels(df):
    """
    Simulate trade outcome for each signal using TP/SL & lookahead window.
    Returns new DataFrame with column 'target': 1=win, 0=loss.
    """
    results = []
    for i in range(len(df) - LOOKAHEAD):
        row = df.iloc[i]
        signal = row['signal']
        entry = row['close']
        result = None
        if signal == "BUY":
            # Max high, min low in lookahead window
            take = df['high'].iloc[i + 1 : i + LOOKAHEAD + 1].max()
            stop = df['low'].iloc[i + 1 : i + LOOKAHEAD + 1].min()
            if take >= entry + TP:
                result = 1  # Win
            elif stop <= entry - SL:
                result = 0  # Loss
            else:
                result = int(df['close'].iloc[i + LOOKAHEAD] > entry)
        elif signal == "SELL":
            take = df['low'].iloc[i + 1 : i + LOOKAHEAD + 1].min()
            stop = df['high'].iloc[i + 1 : i + LOOKAHEAD + 1].max()
            if take <= entry - TP:
                result = 1
            elif stop >= entry + SL:
                result = 0
            else:
                result = int(df['close'].iloc[i + LOOKAHEAD] < entry)
        else:
            result = None
        results.append(result)
    df = df.iloc[:-LOOKAHEAD].copy()
    df['target'] = results
    return df


def main():
    df = pd.read_csv("ohlc_data.csv")
    df = generate_signals(df)
    df = backtest_labels(df)
    # Only keep signals, drop rows with 'HOLD' or missing label
    df = df[df['signal'] != "HOLD"]
    df = df.dropna(subset=['target'])
    # Prepare feature set
    features = ['SMA_FAST', 'RSI', 'signal', 'close', 'target']
    df[features].to_csv("signals_for_ai_training.csv", index=False)
    print("✅ ML training dataset built with backtest: signals_for_ai_training.csv")


if __name__ == "__main__":
    main()
