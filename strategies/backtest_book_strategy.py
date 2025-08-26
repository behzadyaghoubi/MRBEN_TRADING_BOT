import pandas as pd

IN_FILE = "XAUUSD_M15_history_with_signals.csv"
OUT_FILE = "backtest_results.csv"

# پارامترهای بک‌تست (قابل تنظیم)
SL_PIPS = 40  # حد ضرر (پیپ)
TP_PIPS = 80  # حد سود (پیپ)
point_value = 1  # ارزش هر پیپ (برای XAUUSD)

df = pd.read_csv(IN_FILE)
df['time'] = pd.to_datetime(df['time'])
results = []

for i, row in df.iterrows():
    if row['signal'] not in ("BUY", "SELL"):
        continue
    entry = row['close']
    entry_time = row['time']
    direction = 1 if row['signal'] == "BUY" else -1

    # دنبال کندلی بگرد که حد سود یا ضرر فعال شود
    outcome = None
    for j in range(i + 1, min(i + 30, len(df))):  # ماکزیمم ۳۰ کندل جلوتر
        high = df.iloc[j]['high']
        low = df.iloc[j]['low']
        if direction == 1:
            # BUY
            if high >= entry + TP_PIPS * point_value:
                outcome = TP_PIPS
                exit_price = entry + TP_PIPS * point_value
                break
            elif low <= entry - SL_PIPS * point_value:
                outcome = -SL_PIPS
                exit_price = entry - SL_PIPS * point_value
                break
        else:
            # SELL
            if low <= entry - TP_PIPS * point_value:
                outcome = TP_PIPS
                exit_price = entry - TP_PIPS * point_value
                break
            elif high >= entry + SL_PIPS * point_value:
                outcome = -SL_PIPS
                exit_price = entry + SL_PIPS * point_value
                break
    if outcome is None:
        outcome = df.iloc[min(i + 30, len(df) - 1)]['close'] - entry
        exit_price = df.iloc[min(i + 30, len(df) - 1)]['close']

    results.append(
        {
            "timestamp": entry_time,
            "signal": row['signal'],
            "entry_price": entry,
            "exit_price": exit_price,
            "profit": outcome,
        }
    )

pd.DataFrame(results).to_csv(OUT_FILE, index=False)
print(f"✅ بک‌تست کامل شد! {len(results)} معامله ثبت شد.\nنتیجه در: {OUT_FILE}")
