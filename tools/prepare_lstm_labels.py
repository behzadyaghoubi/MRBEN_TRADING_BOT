import pandas as pd

# بارگذاری دیتا (تایم فریم 5 دقیقه، یا هر دیتایی که داری)
df = pd.read_csv("ohlc_data.csv")

# ایجاد ستون هدف: اگر قیمت بسته شدن آینده بالاتر بود، BUY (1)، در غیر اینصورت SELL (0)
future_close = df['close'].shift(-1)
df['target'] = (future_close > df['close']).astype(int)

# حذف ردیف‌های آخر که مقدار هدفشون نال شده (به دلیل شیفت)
df = df.dropna().reset_index(drop=True)

# ذخیره دیتا برای آموزش مدل
df.to_csv("lstm_train_data.csv", index=False)

print(df['target'].value_counts())
print("✅ برچسب‌گذاری (Labeling) انجام شد و فایل lstm_train_data.csv ساخته شد.")