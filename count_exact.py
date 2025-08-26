import pandas as pd

# Load the synthetic dataset
df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')

# Count each signal type
buy_count = len(df[df['signal'] == 'BUY'])
sell_count = len(df[df['signal'] == 'SELL'])
hold_count = len(df[df['signal'] == 'HOLD'])
total_count = len(df)

print(f"Total samples: {total_count}")
print(f"BUY: {buy_count}")
print(f"SELL: {sell_count}")
print(f"HOLD: {hold_count}")

print(f"\nPercentages:")
print(f"BUY: {buy_count/total_count*100:.1f}%")
print(f"SELL: {sell_count/total_count*100:.1f}%")
print(f"HOLD: {hold_count/total_count*100:.1f}%")

if buy_count > 0 and sell_count > 0:
    ratio = buy_count / sell_count
    print(f"\nBUY/SELL ratio: {ratio:.2f}")
    
    if 0.8 <= ratio <= 1.2:
        print("✅ Balanced BUY/SELL distribution")
    else:
        print("⚠️ Unbalanced BUY/SELL distribution") 