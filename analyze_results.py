import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('ai_trading_results.csv')

df['equity'].plot(title='Equity Curve', figsize=(12, 6))
plt.ylabel('Equity ($)')
plt.grid(True)
plt.savefig('equity_curve.png')
plt.show()

# محاسبه Drawdown
df['drawdown'] = df['equity'] - df['equity'].cummax()
max_drawdown = df['drawdown'].min()

print(f"Max Drawdown ($): {max_drawdown:.2f}")
print(f"Final Equity: {df['equity'].iloc[-1]:.2f}")
