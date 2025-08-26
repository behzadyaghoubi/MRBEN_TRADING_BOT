import pandas as pd
import numpy as np

# Load OHLC data with high, low, close columns
ohlc_df = pd.read_csv('ohlc_data.csv')
# Load signals data (must have 'time' and 'balanced_signal')
signals_df = pd.read_csv('lstm_balanced_signals_final.csv')

# Convert time columns to datetime for merge
ohlc_df['time'] = pd.to_datetime(ohlc_df['time'])
signals_df['time'] = pd.to_datetime(signals_df['time'])

# Merge dataframes on 'time' - inner join to keep matching rows only
df = pd.merge(ohlc_df, signals_df[['time', 'balanced_signal']], on='time', how='inner')

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

df['atr'] = calculate_atr(df)

initial_balance = 10000
risk_per_trade = 0.01

balance = initial_balance
equity_curve = []
position = None
entry_price = 0
sl = 0
tp = 0
lots = 0
win_trades = 0
loss_trades = 0

pip_value = 1  # adjust based on instrument

window = 14  # ATR period

for i in range(window + 1, len(df)):
    price = df['close'].iloc[i]
    signal = df['balanced_signal'].iloc[i-1]
    atr = df['atr'].iloc[i]

    if position is None:
        if signal == 1:
            entry_price = df['close'].iloc[i-1]
            sl = entry_price - atr * 1.5
            tp = entry_price + atr * 3
            risk_amount = balance * risk_per_trade
            lots = risk_amount / ((entry_price - sl) * pip_value)
            position = 'long'
        elif signal == -1:
            entry_price = df['close'].iloc[i-1]
            sl = entry_price + atr * 1.5
            tp = entry_price - atr * 3
            risk_amount = balance * risk_per_trade
            lots = risk_amount / ((sl - entry_price) * pip_value)
            position = 'short'
        equity_curve.append(balance)
        continue

    if position == 'long':
        if price <= sl:
            balance -= lots * (entry_price - sl) * pip_value
            position = None
            loss_trades += 1
        elif price >= tp:
            balance += lots * (tp - entry_price) * pip_value
            position = None
            win_trades += 1
        equity_curve.append(balance)
    elif position == 'short':
        if price >= sl:
            balance -= lots * (sl - entry_price) * pip_value
            position = None
            loss_trades += 1
        elif price <= tp:
            balance += lots * (entry_price - tp) * pip_value
            position = None
            win_trades += 1
        equity_curve.append(balance)
    else:
        equity_curve.append(balance)

if len(equity_curve) < len(df):
    equity_curve += [balance] * (len(df) - len(equity_curve))

df['equity'] = equity_curve
df.to_csv('ai_trading_results_atr.csv', index=False)

print(f"✅ Backtest with ATR-based SL/TP and dynamic position sizing completed and saved.")
print(f"Win trades: {win_trades} | Loss trades: {loss_trades}")
print(f"Max Equity: {df['equity'].max()} | Final Equity: {df['equity'].iloc[-1]}")