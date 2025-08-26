import pandas as pd

def check_synthetic_distribution():
    """Check the distribution of the synthetic balanced dataset."""
    print("🔍 بررسی توزیع دیتاست مصنوعی متعادل")
    print("=" * 50)
    
    # Load synthetic dataset
    df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')
    
    # Get signal counts
    signal_counts = df['signal'].value_counts()
    total_samples = len(df)
    
    print(f"کل نمونه‌ها: {total_samples}")
    print("\nتوزیع سیگنال‌ها:")
    for signal in signal_counts.index:
        count = signal_counts[signal]
        percentage = (count / total_samples) * 100
        print(f"  {signal}: {count} ({percentage:.1f}%)")
    
    # Check balance
    if 'BUY' in signal_counts and 'SELL' in signal_counts:
        buy_sell_ratio = signal_counts['BUY'] / signal_counts['SELL']
        print(f"\nنسبت BUY/SELL: {buy_sell_ratio:.2f}")
        
        if 0.8 <= buy_sell_ratio <= 1.2:
            print("✅ توزیع BUY/SELL متعادل است")
        else:
            print("⚠️ توزیع BUY/SELL نامتعادل است")
    
    # Check HOLD dominance
    if 'HOLD' in signal_counts:
        hold_percentage = (signal_counts['HOLD'] / total_samples) * 100
        print(f"غلبه HOLD: {hold_percentage:.1f}%")
        
        if hold_percentage < 70:
            print("✅ غلبه HOLD قابل قبول است")
        else:
            print("⚠️ غلبه HOLD زیاد است")

if __name__ == "__main__":
    check_synthetic_distribution() 