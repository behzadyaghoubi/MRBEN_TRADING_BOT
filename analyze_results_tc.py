import pandas as pd

df = pd.read_csv('ai_trading_results.csv')
fee_rate = 0.0003  # معادل 0.03% در هر ترید

capital = 10000
equity_curve = [capital]
for i in range(1, len(df)):
    if df['final_signal'].iloc[i - 1] == 1:
        ret = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
        net_ret = ret - fee_rate  # کسر کارمزد از بازدهی
        capital *= 1 + net_ret
    equity_curve.append(capital)
df['equity_tc'] = equity_curve
df.to_csv('ai_trading_results_tc.csv', index=False)
print(f"Max Equity: {df['equity_tc'].max()}")
print(f"Min Equity: {df['equity_tc'].min()}")
print(f"Final Equity: {df['equity_tc'].iloc[-1]}")
