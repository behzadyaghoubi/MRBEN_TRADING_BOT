import json
import subprocess
import time

# زمان بین اجراها (مثلاً هر 12 ساعت)
interval_hours = 12
interval_seconds = interval_hours * 60 * 60

while True:
    print("🕒 اجرای خودکار MR BEN شروع شد...")

    # اجرای استراتژی و ساخت سیگنال جدید
    subprocess.run(["python", "ultimate_strategy.py"])

    # پیش‌بینی کیفیت سیگنال با AI
    subprocess.run(["python", "ai_predictor.py"])

    # بررسی احتمال موفقیت از ai_predictor
    try:
        with open("latest_signal.json") as f:
            latest = json.load(f)
        if latest.get("confidence", 0) >= 60:
            print("✅ سیگنال با اعتماد بالا شناسایی شد، اجرای معامله...")
            subprocess.run(["python", "live_trader.py"])
        else:
            print("⚠️ سیگنال ضعیف، اجرای معامله انجام نشد.")
    except Exception as e:
        print(f"⛔ خطا در بررسی اجرای خودکار: {e}")

    print(f"⏸️ انتظار برای {interval_hours} ساعت بعدی...")
    time.sleep(interval_seconds)
