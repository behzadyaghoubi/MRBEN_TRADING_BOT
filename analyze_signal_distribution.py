import pandas as pd
import numpy as np

def analyze_signal_distribution():
    print("🔍 تحلیل توزیع سیگنال‌ها در دیتاست")
    print("=" * 50)
    
    # Load dataset
    df = pd.read_csv('data/mrben_ai_signal_dataset.csv')
    
    # Analyze signal distribution
    signal_counts = df['signal'].value_counts()
    total_signals = len(df)
    
    print(f"کل سیگنال‌ها: {total_signals}")
    print("\nتوزیع سیگنال‌ها:")
    print(signal_counts)
    
    print("\nدرصدها:")
    for signal in signal_counts.index:
        count = signal_counts[signal]
        percentage = (count / total_signals) * 100
        print(f"{signal}: {count} ({percentage:.1f}%)")
    
    # Check for BUY bias
    buy_count = signal_counts.get('BUY', 0)
    sell_count = signal_counts.get('SELL', 0)
    hold_count = signal_counts.get('HOLD', 0)
    
    print(f"\n🔍 تحلیل Bias:")
    print(f"BUY/SELL ratio: {buy_count/sell_count:.2f}" if sell_count > 0 else "BUY/SELL ratio: ∞ (no SELL signals)")
    print(f"HOLD dominance: {hold_count/total_signals*100:.1f}%")
    
    if buy_count > sell_count * 2:
        print("⚠️  BUY bias detected!")
    elif sell_count > buy_count * 2:
        print("⚠️  SELL bias detected!")
    else:
        print("✅ نسبت BUY/SELL متعادل است")

if __name__ == "__main__":
    analyze_signal_distribution() 