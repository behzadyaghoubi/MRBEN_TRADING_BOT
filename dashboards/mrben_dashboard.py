import matplotlib.pyplot as plt
import pandas as pd

# مسیر لاگ معاملات (آخرین فایل بک‌تست مدیریت ریسک)
csv_file = "mrben_dynamic_risk_protect_trades.csv"
df = pd.read_csv(csv_file)

# 1. خلاصه عملکرد کلی
start_capital = df['capital'].iloc[0]
end_capital = df['capital'].iloc[-1]
profit = end_capital - start_capital
num_trades = df[df['action'].str.contains('CLOSE')].shape[0]
win_trades = df[(df['action'].str.contains('CLOSE')) & (df['capital'].diff() > 0)].shape[0]
loss_trades = num_trades - win_trades
win_rate = win_trades / num_trades * 100 if num_trades > 0 else 0
max_drawdown = (df['capital'].cummax() - df['capital']).max()
max_drawdown_pct = (
    (max_drawdown / df['capital'].cummax().max()) * 100 if df['capital'].cummax().max() > 0 else 0
)

print(f"سرمایه اولیه: {start_capital:.2f}")
print(f"سرمایه نهایی: {end_capital:.2f}")
print(f"سود خالص: {profit:.2f}")
print(f"تعداد معاملات بسته شده: {num_trades}")
print(f"تعداد معاملات سودده: {win_trades}")
print(f"تعداد معاملات ضررده: {loss_trades}")
print(f"نرخ برد: {win_rate:.1f}%")
print(f"بیشترین افت سرمایه (Drawdown): {max_drawdown:.2f} دلار ({max_drawdown_pct:.1f}%)")

# 2. نمودار سرمایه (Equity Curve)
plt.figure(figsize=(10, 5))
plt.plot(df['capital'].values, label="سرمایه")
plt.title("نمودار سرمایه ربات MR BEN")
plt.xlabel("تعداد معاملات")
plt.ylabel("سرمایه (دلار)")
plt.legend()
plt.grid()

# 3. نمایش لاگ ضررهای متوالی
if 'loss_streak' in df.columns:
    plt.figure(figsize=(10, 2))
    plt.plot(df['loss_streak'].values, color='red', label="ضررهای متوالی")
    plt.title("نمودار ضررهای متوالی")
    plt.xlabel("تعداد معاملات")
    plt.ylabel("تعداد ضرر پشت سر هم")
    plt.legend()
    plt.grid()

plt.show()

# 4. جدول خلاصه معاملات آخر
print("\nآخرین 10 معامله:")
print(df[['date', 'action', 'price', 'capital', 'lot', 'SL', 'TP', 'phase']].tail(10))
