import pandas as pd


def test_synthetic_dataset():
    """Test the synthetic dataset directly."""
    try:
        # Load the synthetic dataset
        df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')

        # Count signals
        buy_count = len(df[df['signal'] == 'BUY'])
        sell_count = len(df[df['signal'] == 'SELL'])
        hold_count = len(df[df['signal'] == 'HOLD'])
        total_count = len(df)

        print("🔍 تست مستقیم دیتاست مصنوعی")
        print("=" * 50)
        print(f"کل نمونه‌ها: {total_count}")
        print(f"BUY: {buy_count} ({buy_count/total_count*100:.1f}%)")
        print(f"SELL: {sell_count} ({sell_count/total_count*100:.1f}%)")
        print(f"HOLD: {hold_count} ({hold_count/total_count*100:.1f}%)")

        # Check balance
        if buy_count > 0 and sell_count > 0:
            ratio = buy_count / sell_count
            print(f"\nنسبت BUY/SELL: {ratio:.2f}")

            if 0.8 <= ratio <= 1.2:
                print("✅ توزیع BUY/SELL متعادل است")
            else:
                print("⚠️ توزیع BUY/SELL نامتعادل است")

        # Check HOLD dominance
        hold_percentage = hold_count / total_count * 100
        print(f"\nغلبه HOLD: {hold_percentage:.1f}%")

        if hold_percentage < 70:
            print("✅ غلبه HOLD قابل قبول است")
        else:
            print("⚠️ غلبه HOLD زیاد است")

        # Compare with original
        original_hold = 97.7
        improvement = original_hold - hold_percentage
        print(f"\nبهبود نسبت به دیتاست اصلی: {improvement:.1f}% کاهش غلبه HOLD")

        return True

    except Exception as e:
        print(f"خطا: {e}")
        return False


if __name__ == "__main__":
    test_synthetic_dataset()
