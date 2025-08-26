import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# خواندن داده سیگنال‌ها و لاگ معاملات
signals = pd.read_csv("final_signal.csv")
trades = pd.read_csv("trades_log.csv")

# تطبیق معاملات با سیگنال‌ها بر اساس زمان نزدیک و نوع سیگنال
signals['timestamp'] = pd.to_datetime(signals['time'])
trades['timestamp'] = pd.to_datetime(trades['timestamp'])

merged = pd.merge_asof(signals.sort_values('timestamp'),
                       trades.sort_values('timestamp'),
                       on='timestamp',
                       direction='forward',
                       tolerance=pd.Timedelta('3h'))

# برچسب‌گذاری موفقیت سیگنال (سود مثبت)
merged['label'] = merged['profit'].apply(lambda x: 1 if x > 0 else 0)

# انتخاب ویژگی‌ها
features = merged[['confidence']]
labels = merged['label']

# آموزش مدل
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ارزیابی مدل
y_pred = model.predict(X_test)
print("\n📊 گزارش عملکرد مدل:")
print(classification_report(y_test, y_pred))

# ذخیره مدل نهایی در صورت نیاز (با joblib یا pickle)
