import pandas as pd

# لاگ معاملات اولیه
trades = pd.read_csv('live_trades_log.csv')

results = []
open_trades = []  # لیست معاملات باز

for idx, row in trades.iterrows():
    signal = row['action']  # اصلاح شد!
    price = float(row['entry_price'])  # اصلاح شد!
    time = row['timestamp']
    symbol = row['symbol']

    # اگر سیگنال BUY یا SELL است، معامله باز اضافه می‌شود
    if signal == 'BUY':
        open_trades.append(
            {'open_time': time, 'direction': 'BUY', 'entry': price, 'symbol': symbol}
        )
    elif signal == 'SELL':
        open_trades.append(
            {'open_time': time, 'direction': 'SELL', 'entry': price, 'symbol': symbol}
        )

    # هر سیگنال بعدی برعکس (مثلاً SELL بعد از BUY)، همه معاملات خلاف را می‌بندد
    to_remove = []
    for i, ot in enumerate(open_trades):
        if ot['direction'] == 'BUY' and signal == 'SELL' and ot['symbol'] == symbol:
            profit = price - ot['entry']
            results.append(
                {
                    'open_time': ot['open_time'],
                    'close_time': time,
                    'direction': ot['direction'],
                    'entry': ot['entry'],
                    'exit': price,
                    'result': profit,
                    'symbol': symbol,
                }
            )
            to_remove.append(i)
        elif ot['direction'] == 'SELL' and signal == 'BUY' and ot['symbol'] == symbol:
            profit = ot['entry'] - price
            results.append(
                {
                    'open_time': ot['open_time'],
                    'close_time': time,
                    'direction': ot['direction'],
                    'entry': ot['entry'],
                    'exit': price,
                    'result': profit,
                    'symbol': symbol,
                }
            )
            to_remove.append(i)

    # حذف معاملات بسته شده
    for i in sorted(to_remove, reverse=True):
        del open_trades[i]

# معاملات باز مانده را هم با قیمت آخر ببند
if len(trades) > 0:
    last_row = trades.iloc[-1]
    last_price = float(last_row['entry_price'])
    last_time = last_row['timestamp']
    last_symbol = last_row['symbol']
    for ot in open_trades:
        if ot['symbol'] == last_symbol:
            profit = (
                (last_price - ot['entry'])
                if ot['direction'] == 'BUY'
                else (ot['entry'] - last_price)
            )
            results.append(
                {
                    'open_time': ot['open_time'],
                    'close_time': last_time,
                    'direction': ot['direction'],
                    'entry': ot['entry'],
                    'exit': last_price,
                    'result': profit,
                    'symbol': ot['symbol'],
                }
            )

results_df = pd.DataFrame(results)
results_df.to_csv('trades_with_results.csv', index=False)
print("✅ فایل trades_with_results.csv با نام‌گذاری درست ساخته شد!")
