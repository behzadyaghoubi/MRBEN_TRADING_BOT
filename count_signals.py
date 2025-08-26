import pandas as pd

# Load the synthetic dataset
df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')

# Count signals
signal_counts = df['signal'].value_counts()
total = len(df)

print(f"Total samples: {total}")
print("\nSignal distribution:")
for signal, count in signal_counts.items():
    percentage = (count / total) * 100
    print(f"{signal}: {count} ({percentage:.1f}%)")

# Check balance
if 'BUY' in signal_counts and 'SELL' in signal_counts:
    ratio = signal_counts['BUY'] / signal_counts['SELL']
    print(f"\nBUY/SELL ratio: {ratio:.2f}")

    if 0.8 <= ratio <= 1.2:
        print("✅ Balanced BUY/SELL distribution")
    else:
        print("⚠️ Unbalanced BUY/SELL distribution")
