# file: check_signals.py
import pandas as pd

df = pd.read_csv('lstm_balanced_signals_final.csv')
print(df[['time', 'close', 'balanced_signal']].tail(20))
print(df['balanced_signal'].value_counts())