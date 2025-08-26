# file: plot_equity_curve.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ai_trading_results_pro.csv')
plt.figure(figsize=(12,5))
plt.plot(df['equity'], label="Equity")
plt.title('LSTM Pro Equity Curve')
plt.xlabel('Trade')
plt.ylabel('Equity')
plt.legend()
plt.grid()
plt.savefig('equity_curve_lstm_pro.png')
plt.show()