import pandas as pd
import numpy as np
from book_strategy import generate_book_signals

# 1. دیتای تست رو بخون
df = pd.read_csv("mock_xauusd_m15.csv")

# 2. سیگنال‌ها رو بساز
df = generate_book_signals(df)

# 3. اجرای بک‌تست ساده
initial_balance = 10000
balance = initial_balance
lot_size = 0.1  # می‌تونی هرچی خواستی بذاری
profit_list = []
position = None
entry_price = 0

for i, row in df.iterrows():
    signal = row['signal']
    close = row['close']
    if position is None and signal == 'BUY':
        position = 'BUY'
        entry_price = close
    elif position == 'BUY' and signal == 'SELL':
        # فرض ساده: پوزیشن رو می‌بندیم
        profit = (close - entry_price) * 100 * lot_size  # سود به دلار
        balance += profit
        profit_list.append(profit)
        position = None
    elif position is None and signal == 'SELL':
        position = 'SELL'
        entry_price = close
    elif position == 'SELL' and signal == 'BUY':
        profit = (entry_price - close) * 100 * lot_size
        balance += profit
        profit_list.append(profit)
        position = None

# 4. گزارش بک‌تست
print(f"Initial balance: {initial_balance}")
print(f"Final balance: {balance}")
print(f"Total trades: {len(profit_list)}")
print(f"Win rate: {np.mean([p > 0 for p in profit_list])*100:.2f}%")
print(f"Total profit: {sum(profit_list):.2f} USD")