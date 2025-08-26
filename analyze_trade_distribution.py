import pandas as pd


def analyze_trade_distribution():
    print("🔍 تحلیل توزیع معاملات در live_trades.csv")
    print("=" * 50)

    # Load trade log with error handling for inconsistent columns
    try:
        df = pd.read_csv('logs/live_trades.csv', on_bad_lines='skip')
    except:
        # If that fails, try reading with different parameters
        df = pd.read_csv('logs/live_trades.csv', engine='python', error_bad_lines=False)

    # Extract just the action column for analysis
    if 'action' in df.columns:
        action_counts = df['action'].value_counts()
        total_trades = len(df)

        print(f"کل معاملات: {total_trades}")
        print("\nتوزیع معاملات:")
        print(action_counts)

        print("\nدرصدها:")
        for action in action_counts.index:
            count = action_counts[action]
            percentage = (count / total_trades) * 100
            print(f"{action}: {count} ({percentage:.1f}%)")

        # Check for BUY bias
        buy_count = action_counts.get('BUY', 0)
        sell_count = action_counts.get('SELL', 0)

        print("\n🔍 تحلیل Bias:")
        if sell_count > 0:
            print(f"BUY/SELL ratio: {buy_count/sell_count:.2f}")
        else:
            print("BUY/SELL ratio: ∞ (no SELL trades)")

        if buy_count > sell_count * 2:
            print("⚠️  BUY bias detected!")
        elif sell_count > buy_count * 2:
            print("⚠️  SELL bias detected!")
        else:
            print("✅ نسبت BUY/SELL متعادل است")

        # Analyze confidence levels if available
        if 'confidence' in df.columns:
            print("\n📊 تحلیل سطح اطمینان:")
            print(f"میانگین اطمینان: {df['confidence'].mean():.3f}")
            print(f"حداقل اطمینان: {df['confidence'].min():.3f}")
            print(f"حداکثر اطمینان: {df['confidence'].max():.3f}")

        # Analyze source if available
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            print("\n📈 منبع سیگنال‌ها:")
            print(source_counts)
    else:
        print("❌ ستون 'action' در فایل یافت نشد")
        print("ستون‌های موجود:", df.columns.tolist())


if __name__ == "__main__":
    analyze_trade_distribution()
