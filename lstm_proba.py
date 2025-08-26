import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('ai_trading_results.csv')
print(df['lstm_proba'].describe())
plt.hist(df['lstm_proba'], bins=30)
plt.title("LSTM Probabilities Histogram")
plt.xlabel("lstm_proba")
plt.ylabel("count")
plt.show()
