import pandas as pd

# فایل دیتایی که قبلاً ساختی (مثلاً lstm_train_data.csv یا هر دیتای خام تمیز که داری)
df = pd.read_csv('lstm_train_data.csv')  # نام فایل را اگر فرق داشت همینجا اصلاح کن

future_horizon = 6  # چند کندل جلوتر را بررسی کند (مثلاً 6 کندل 5 دقیقه‌ای = 30 دقیقه)
threshold = 0.002   # مثلاً 0.2 درصد حرکت (0.002 برای طلا = حدود 6 دلار در قیمت 3000)

# محاسبه بازده آینده
df['future_return'] = (df['close'].shift(-future_horizon) - df['close']) / df['close']

# ساخت برچسب هدف: فقط حرکت‌های معنی‌دار ۱ یا ۰
df['target'] = 0
df.loc[df['future_return'] > threshold, 'target'] = 1
df.loc[df['future_return'] < -threshold, 'target'] = -1  # اگر فقط buy می‌خوای این خط رو حذف کن

print(df['target'].value_counts())
df.to_csv('lstm_train_data_pro.csv', index=False)
print('✅ برچسب‌گذاری جدید انجام شد و ذخیره شد: lstm_train_data_pro.csv')