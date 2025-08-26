import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# --- اطلاعات اکانت خودت رو از settings.json بخون ---
import json
with open("settings.json", "r") as f:
    s = json.load(f)
login = s["login"]
password = s["password"]
server = s["server"]
symbol = s.get("symbol", "XAUUSD")

if not mt5.initialize(login=login, password=password, server=server):
    print("❌ Connection failed:", mt5.last_error())
    exit()

trades = mt5.history_deals_get(datetime(2024, 1, 1), datetime.now(), group=symbol+"*")
if trades is None or len(trades) == 0:
    print("No trades found!")
    mt5.shutdown()
    exit()

df = pd.DataFrame(list(trades))
df["timestamp"] = pd.to_datetime(df["time"], unit="s")
df = df[df["symbol"] == symbol]
df = df[df["type"].isin([mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL])]

# خروجی با ساختار حرفه‌ای برای داشبورد
out = pd.DataFrame({
    "timestamp": df["timestamp"],
    "symbol": df["symbol"],
    "action": df["type"].map({mt5.ORDER_TYPE_BUY: "BUY", mt5.ORDER_TYPE_SELL: "SELL"}),
    "entry_price": df["price"],
    "sl": df.get("sl", None),   # اگر SL در دیتای شما هست
    "tp": df.get("tp", None),   # اگر TP در دیتای شما هست
    "lot": df["volume"],
    "balance": df["profit"].cumsum() + 100000,  # فرض اولیه
    "ai_decision": 1,  # دستی (یا از دیتا اگر داشتی)
    "result_code": 10009,
    "comment": df["comment"],
    "profit": df["profit"]
})

out.to_csv("live_trades_log.csv", index=False)
print("✅ فایل معاملات واقعی ساخته شد: live_trades_log.csv")
mt5.shutdown()