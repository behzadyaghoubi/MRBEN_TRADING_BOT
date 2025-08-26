import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log_file = "live_trades_log.csv"
df = pd.read_csv(log_file)

# 1. مرتب‌سازی و تمیزکاری (در صورت نیاز)
if "timestamp" in df.columns:
    df = df.sort_values("timestamp")

# 2. ساخت ستون نتیجه (WIN/LOSE) و محاسبه PnL
if "result" not in df.columns or df["result"].isnull().all():
    # اگر لاگ فقط معاملات ورودی دارد، فرض کن هر سفارش بسته شده با TP یا SL (اینجا فقط جهت دمو)
    df["exit_price"] = np.where(
        np.random.rand(len(df)) > 0.5, df["tp"], df["sl"]
    )
    df["result"] = np.where(
        ((df["action"] == "BUY") & (df["exit_price"] > df["entry_price"])) |
        ((df["action"] == "SELL") & (df["exit_price"] < df["entry_price"])),
        "WIN", "LOSE"
    )

# 3. محاسبه سود/زیان هر معامله (PNL)
df["pnl"] = np.where(
    df["action"] == "BUY",
    df["exit_price"] - df["entry_price"],
    df["entry_price"] - df["exit_price"]
)

# 4. آمار دقیق
total_trades = len(df)
num_win = (df["result"] == "WIN").sum()
num_lose = (df["result"] == "LOSE").sum()
win_rate = num_win / total_trades * 100 if total_trades else 0
avg_profit = df[df["result"] == "WIN"]["pnl"].mean()
avg_loss = df[df["result"] == "LOSE"]["pnl"].mean()
net_profit = df["pnl"].sum()

print(f"\n📊 تعداد کل معاملات: {total_trades}")
print(f"✅ تعداد معاملات موفق: {num_win} | ❌ ناموفق: {num_lose}")
print(f"🔥 Win Rate: {win_rate:.2f}%")
print(f"💵 میانگین سود معامله موفق: {avg_profit:.2f}")
print(f"💸 میانگین زیان معامله ناموفق: {avg_loss:.2f}")
print(f"📈 سود خالص همه معاملات: {net_profit:.2f}\n")

# 5. نمودار Equity Curve (رشد سرمایه با هر معامله)
df["equity"] = 10000 + df["pnl"].cumsum()  # سرمایه اولیه 10,000 فرضی
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["equity"], label="Equity Curve", color="blue")
plt.title("رشد سرمایه ربات (Equity Curve)")
plt.xlabel("زمان")
plt.ylabel("سرمایه (فرضی)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 6. نمودار تعداد معاملات موفق/ناموفق
plt.figure(figsize=(5, 5))
plt.bar(["WIN", "LOSE"], [num_win, num_lose], color=["green", "red"])
plt.title("تعداد معاملات موفق و ناموفق")
plt.ylabel("تعداد")
plt.tight_layout()
plt.show()