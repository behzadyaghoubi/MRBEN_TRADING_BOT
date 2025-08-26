import pandas as pd

OHLC_FILE = "XAUUSD_M15_history_with_signals.csv"
BT_FILE = "backtest_results.csv"
OUT_FILE = "mrben_ai_signal_dataset.csv"

# بارگذاری داده
ohlc = pd.read_csv(OHLC_FILE)
bt = pd.read_csv(BT_FILE)
ohlc['time'] = pd.to_datetime(ohlc['time'])
bt['timestamp'] = pd.to_datetime(bt['timestamp'])

dataset = []
for idx, row in bt.iterrows():
    ohlc_idx = ohlc[ohlc['time'] <= row['timestamp']].tail(1)
    if ohlc_idx.empty:
        continue
    c = ohlc_idx.iloc[0]
    sample = {
        "SMA_FAST": c.get('SMA_FAST', 0),
        "RSI": c.get('RSI', 0),
        "signal": 1 if row['signal'] == 'BUY' else -1,
        "close": c['close'],
        "target": 1 if row['profit'] > 0 else 0,
    }
    dataset.append(sample)

df_out = pd.DataFrame(dataset)
df_out.to_csv(OUT_FILE, index=False)
print(f"✅ Dataset for AI model saved: {OUT_FILE} | Samples: {len(df_out)}")
