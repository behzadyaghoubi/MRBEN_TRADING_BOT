import pandas as pd

input_file = "live_trades_log.csv"
output_file = "clean_live_trades_log.csv"

with open(input_file, encoding='utf-8') as f:
    header = f.readline().strip().split(',')

rows = []
with open(input_file, encoding='utf-8') as f:
    next(f)
    for line in f:
        if line.count(',') == len(header) - 1:
            rows.append(line.strip().split(','))

df = pd.DataFrame(rows, columns=header)
df.to_csv(output_file, index=False)
print("✅ clean_live_trades_log.csv ساخته شد (فقط ردیف‌های سالم)")
