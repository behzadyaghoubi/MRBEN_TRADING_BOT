import pandas as pd


def analyze_dataset():
    try:
        df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')

        buy_count = len(df[df['signal'] == 'BUY'])
        sell_count = len(df[df['signal'] == 'SELL'])
        hold_count = len(df[df['signal'] == 'HOLD'])
        total_count = len(df)

        print("Dataset Analysis:")
        print(f"Total samples: {total_count}")
        print(f"BUY: {buy_count} ({buy_count/total_count*100:.1f}%)")
        print(f"SELL: {sell_count} ({sell_count/total_count*100:.1f}%)")
        print(f"HOLD: {hold_count} ({hold_count/total_count*100:.1f}%)")

        if buy_count > 0 and sell_count > 0:
            ratio = buy_count / sell_count
            print(f"BUY/SELL ratio: {ratio:.2f}")

            if 0.8 <= ratio <= 1.2:
                print("✅ Balanced distribution")
            else:
                print("⚠️ Unbalanced distribution")

        print("✅ Dataset ready for LSTM training!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    analyze_dataset()
