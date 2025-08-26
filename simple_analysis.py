import os

import pandas as pd


def main():
    """Simple analysis that can run directly."""
    print("🚀 تحلیل ساده دیتاست مصنوعی")
    print("=" * 50)

    try:
        # Check if synthetic dataset exists
        dataset_path = 'data/mrben_ai_signal_dataset_synthetic_balanced.csv'

        if not os.path.exists(dataset_path):
            print(f"❌ فایل {dataset_path} یافت نشد")
            return False

        # Load dataset
        print("📊 بارگذاری دیتاست...")
        df = pd.read_csv(dataset_path)

        # Count signals
        buy_count = len(df[df['signal'] == 'BUY'])
        sell_count = len(df[df['signal'] == 'SELL'])
        hold_count = len(df[df['signal'] == 'HOLD'])
        total_count = len(df)

        print("\n📈 آمار دیتاست:")
        print(f"   کل نمونه‌ها: {total_count}")
        print(f"   BUY: {buy_count} ({buy_count/total_count*100:.1f}%)")
        print(f"   SELL: {sell_count} ({sell_count/total_count*100:.1f}%)")
        print(f"   HOLD: {hold_count} ({hold_count/total_count*100:.1f}%)")

        # Check balance
        if buy_count > 0 and sell_count > 0:
            ratio = buy_count / sell_count
            print(f"\n⚖️ نسبت BUY/SELL: {ratio:.2f}")

            if 0.8 <= ratio <= 1.2:
                print("   ✅ توزیع BUY/SELL متعادل است")
            else:
                print("   ⚠️ توزیع BUY/SELL نامتعادل است")

        # Compare with original
        original_hold = 97.7
        improvement = original_hold - (hold_count / total_count * 100)
        print(f"\n📊 بهبود: {improvement:.1f}% کاهش غلبه HOLD")

        print("\n✅ دیتاست مصنوعی آماده است!")
        print("✅ می‌توانیم LSTM را بازآموزی کنیم!")

        return True

    except Exception as e:
        print(f"❌ خطا: {e}")
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎯 مراحل بعدی:")
        print("   1. مشکل کیبورد را رفع کنید")
        print("   2. python train_lstm_balanced.py را اجرا کنید")
        print("   3. python live_trader_clean.py را اجرا کنید")
    else:
        print("\n❌ مشکل در تحلیل دیتاست")
