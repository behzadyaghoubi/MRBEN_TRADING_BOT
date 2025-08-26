import pandas as pd

# فایل لاگ معاملات
trades = pd.read_csv('live_trades_log.csv')

# فرض می‌کنیم معاملات به صورت خطی و بدون باز شدن همزمان چند معامله است
results = []
last_buy = None
last_sell = None

for idx, row in trades.iterrows():
    if row['signal'] == 'BUY':
        last_buy = (row['time'], row['price'])
    elif row['signal'] == 'SELL':
        last_sell = (row['time'], row['price'])
    # بررسی آیا معامله بسته شده و سود/زیان باید محاسبه شود
    if last_buy and row['signal'] == 'SELL':
        profit = row['price'] - float(last_buy[1])
        results.append({'open_time': last_buy[0], 'close_time': row['time'], 'direction': 'BUY', 'entry': last_buy[1], 'exit': row['price'], 'result': profit})
        last_buy = None
    elif last_sell and row['signal'] == 'BUY':
        profit = float(last_sell[1]) - row['price']
        results.append({'open_time': last_sell[0], 'close_time': row['time'], 'direction': 'SELL', 'entry': last_sell[1], 'exit': row['price'], 'result': profit})
        last_sell = None

# تبدیل به DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('trades_with_results.csv', index=False)
print("✅ فایل trades_with_results.csv ساخته شد!")
