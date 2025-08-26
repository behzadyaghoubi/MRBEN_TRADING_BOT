import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('ai_trading_results.csv')
df = df.dropna().reset_index(drop=True)

# آستانه‌های جدید (کمتر کن که سیگنال بگیری)
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
TRANSACTION_COST = 1.0

trade_condition = (df['lstm_proba'] > BUY_THRESHOLD) | (df['lstm_proba'] < SELL_THRESHOLD)
df = df[trade_condition].reset_index(drop=True)

print(f"Rows after filtering: {len(df)}")

if df.empty:
    print("⛔️ بعد از فیلتر threshold دیتافریم خالی شد! آستانه را کمتر کن یا مدل را بررسی کن.")
    exit()

df['final_signal'] = np.where(df['lstm_proba'] > BUY_THRESHOLD, 1, 0)  # 1=BUY, 0=SELL
df['trade_return_net'] = df['trade_return'] - TRANSACTION_COST

df['cum_return'] = df['trade_return_net'].cumsum()
df['equity'] = 10000 + df['cum_return']
df['drawdown'] = df['equity'] - df['equity'].cummax()
df['drawdown_pct'] = df['drawdown'] / df['equity'].cummax() * 100

print("Max Equity:", df['equity'].max())
print("Min Equity:", df['equity'].min())
print("Max Drawdown ($):", df['drawdown'].min())
print("Max Drawdown (%):", df['drawdown_pct'].min())
print("Final Equity:", df['equity'].iloc[-1])

plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(df['time'], df['equity'], label='Equity Curve (Net, Filtered)')
plt.title('Equity Curve (After Transaction Cost & Threshold)')
plt.ylabel('Equity ($)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['time'], df['drawdown'], color='red', label='Drawdown ($)')
plt.title('Drawdown')
plt.ylabel('Drawdown')
plt.xlabel('Time')
plt.legend()

plt.tight_layout()
plt.savefig('equity_curve_drawdown_threshold.png')
plt.show()
print("✅ نمودار سود تجمعی و افت سرمایه ذخیره شد (equity_curve_drawdown_threshold.png)")

n_trades = len(df)
win_trades = (df['trade_return_net'] > 0).sum()
loss_trades = n_trades - win_trades
win_rate = win_trades / n_trades * 100
avg_return = df['trade_return_net'].mean()
profit = df['cum_return'].iloc[-1]

print("\n===== Performance Summary (With Threshold) =====")
print(f"Total Trades: {n_trades}")
print(f"Win Trades: {win_trades} | Loss Trades: {loss_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average Trade Return (Net): {avg_return:.4f}")
print(f"Total Profit/Loss (Net): {profit:.2f} $")
