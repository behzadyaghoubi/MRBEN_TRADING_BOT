import pandas as pd

df = pd.read_csv('lstm_filtered_signals.csv')

initial_balance = 10000
risk_per_trade = 0.01
stop_loss_pips = 20
take_profit_pips = 40
pip_value = 1

balance = initial_balance
equity_curve = []
position = None
entry_price = 0
sl = 0
tp = 0
lots = 0
win_trades = 0
loss_trades = 0

for i in range(1, len(df)):
    price = df['close'].iloc[i]
    signal = df['filtered_signal'].iloc[i-1]

    if position is None:
        if signal == 1:
            entry_price = df['close'].iloc[i-1]
            sl = entry_price - stop_loss_pips * pip_value
            tp = entry_price + take_profit_pips * pip_value
            lots = (balance * risk_per_trade) / (stop_loss_pips * pip_value)
            position = 'long'
        elif signal == -1:
            entry_price = df['close'].iloc[i-1]
            sl = entry_price + stop_loss_pips * pip_value
            tp = entry_price - take_profit_pips * pip_value
            lots = (balance * risk_per_trade) / (stop_loss_pips * pip_value)
            position = 'short'
        equity_curve.append(balance)
        continue

    if position == 'long':
        if price <= sl:
            balance -= lots * stop_loss_pips * pip_value
            position = None
            loss_trades += 1
        elif price >= tp:
            balance += lots * take_profit_pips * pip_value
            position = None
            win_trades += 1
        equity_curve.append(balance)
    elif position == 'short':
        if price >= sl:
            balance -= lots * stop_loss_pips * pip_value
            position = None
            loss_trades += 1
        elif price <= tp:
            balance += lots * take_profit_pips * pip_value
            position = None
            win_trades += 1
        equity_curve.append(balance)
    else:
        equity_curve.append(balance)

if len(equity_curve) < len(df):
    equity_curve += [balance] * (len(df) - len(equity_curve))

df['equity'] = equity_curve
df.to_csv('backtest_filtered_results.csv', index=False)
print(f"✅ بک‌تست با سیگنال فیلتر شده انجام شد.")
print(f"Win trades: {win_trades} | Loss trades: {loss_trades}")
print(f"Max Equity: {df['equity'].max()} | Final Equity: {df['equity'].iloc[-1]}")