import pandas as pd

log_file = "live_trades_log.csv"
out_file = "clean_live_trades_log.csv"

# پیدا کردن سطرهای معیوب و استخراج فقط فیلدهای مهم
rows = []
with open(log_file, "r", encoding="utf-8") as f:
    header = f.readline().strip().split(",")
    for line in f:
        parts = line.strip().split(",")
        # سعی کن ستون‌ها را نگه داری (اگر تعداد کم/زیاد بود فقط فیلدهای لازم را بردار)
        if len(parts) < 3:
            continue
        try:
            # پیدا کردن فیلدها بر اساس ترتیب یا اسم ستون (بسته به ساختار فایل لاگ تو)
            d = dict(zip(header, parts))
            rows.append({
                "timestamp": d.get("timestamp", ""),
                "symbol": d.get("symbol", ""),
                "action": d.get("action", ""),
                "profit": d.get("profit", d.get("result", "")),  # اگر profit نبود از result استفاده کن
            })
        except Exception as e:
            continue

df = pd.DataFrame(rows)
df.to_csv(out_file, index=False)
print(f"✅ Clean log saved as {out_file} - Rows: {len(df)}")