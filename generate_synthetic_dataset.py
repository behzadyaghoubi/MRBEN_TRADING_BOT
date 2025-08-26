import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


def generate_synthetic_dataset():
    """Generate synthetic balanced dataset for training."""
    print("🚀 Generating Synthetic Balanced Dataset...")
    print("=" * 60)

    # Load original dataset
    df = pd.read_csv('data/mrben_ai_signal_dataset.csv')

    # Analyze original distribution
    signal_counts = df['signal'].value_counts()
    print("Original distribution:")
    for signal in signal_counts.index:
        count = signal_counts[signal]
        percentage = (count / len(df)) * 100
        print(f"  {signal}: {count} ({percentage:.1f}%)")

    # Separate classes
    hold_data = df[df['signal'] == 'HOLD'].copy()
    buy_data = df[df['signal'] == 'BUY'].copy()
    sell_data = df[df['signal'] == 'SELL'].copy()

    print(f"\nOriginal counts: HOLD={len(hold_data)}, BUY={len(buy_data)}, SELL={len(sell_data)}")

    # Target balanced distribution (30% each for BUY/SELL, 40% HOLD)
    target_total = 3000  # Total dataset size
    target_buy = int(target_total * 0.3)  # 30% BUY
    target_sell = int(target_total * 0.3)  # 30% SELL
    target_hold = int(target_total * 0.4)  # 40% HOLD

    print(f"Target distribution: HOLD={target_hold}, BUY={target_buy}, SELL={target_sell}")

    # Generate synthetic data using multiple methods
    synthetic_df = generate_synthetic_samples(
        hold_data, buy_data, sell_data, target_hold, target_buy, target_sell
    )

    # Save synthetic dataset
    output_path = 'data/mrben_ai_signal_dataset_synthetic_balanced.csv'
    synthetic_df.to_csv(output_path, index=False)

    print(f"\n✅ Synthetic dataset saved to: {output_path}")
    print(f"Total samples: {len(synthetic_df)}")

    # Analyze final distribution
    final_counts = synthetic_df['signal'].value_counts()
    print("\nFinal balanced distribution:")
    for signal in final_counts.index:
        count = final_counts[signal]
        percentage = (count / len(synthetic_df)) * 100
        print(f"  {signal}: {count} ({percentage:.1f}%)")

    return synthetic_df


def generate_synthetic_samples(
    hold_data, buy_data, sell_data, target_hold, target_buy, target_sell
):
    """Generate synthetic samples using multiple techniques."""

    # Method 1: Oversampling with noise
    synthetic_hold = generate_oversampled_data(hold_data, target_hold, 'HOLD')
    synthetic_buy = generate_oversampled_data(buy_data, target_buy, 'BUY')
    synthetic_sell = generate_oversampled_data(sell_data, target_sell, 'SELL')

    # Method 2: SMOTE for minority classes
    if len(buy_data) > 0 and len(sell_data) > 0:
        synthetic_buy_smote = generate_smote_data(buy_data, target_buy, 'BUY')
        synthetic_sell_smote = generate_smote_data(sell_data, target_sell, 'SELL')

        # Combine SMOTE with oversampling
        synthetic_buy = pd.concat([synthetic_buy, synthetic_buy_smote]).drop_duplicates()
        synthetic_sell = pd.concat([synthetic_sell, synthetic_sell_smote]).drop_duplicates()

    # Method 3: Pattern-based generation
    synthetic_buy_pattern = generate_pattern_based_data('BUY', target_buy)
    synthetic_sell_pattern = generate_pattern_based_data('SELL', target_sell)

    # Combine all synthetic data
    combined_df = pd.concat(
        [
            synthetic_hold.head(target_hold),
            synthetic_buy.head(target_buy),
            synthetic_sell.head(target_sell),
            synthetic_buy_pattern.head(target_buy // 2),
            synthetic_sell_pattern.head(target_sell // 2),
        ],
        ignore_index=True,
    )

    # Shuffle and reset index
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return combined_df


def generate_oversampled_data(data, target_count, signal_type):
    """Generate oversampled data with noise."""
    if len(data) == 0:
        return pd.DataFrame()

    # Calculate how many times to repeat
    repeat_times = max(1, target_count // len(data))

    # Repeat data
    repeated_data = []
    for _ in range(repeat_times + 1):
        # Add small noise to avoid exact duplicates
        noise_data = data.copy()
        numeric_columns = [
            'open',
            'high',
            'low',
            'close',
            'SMA20',
            'SMA50',
            'RSI',
            'MACD',
            'MACD_signal',
            'MACD_hist',
        ]

        for col in numeric_columns:
            if col in noise_data.columns:
                # Add 0.1% noise
                noise_factor = np.random.normal(1, 0.001, len(noise_data))
                noise_data[col] = noise_data[col] * noise_factor

        repeated_data.append(noise_data)

    # Combine and sample to target count
    combined = pd.concat(repeated_data, ignore_index=True)
    return combined.sample(n=target_count, random_state=42)


def generate_smote_data(data, target_count, signal_type):
    """Generate synthetic data using SMOTE."""
    if len(data) < 2:
        return pd.DataFrame()

    try:
        # Prepare features for SMOTE
        feature_columns = [
            'open',
            'high',
            'low',
            'close',
            'SMA20',
            'SMA50',
            'RSI',
            'MACD',
            'MACD_signal',
            'MACD_hist',
        ]
        available_features = [col for col in feature_columns if col in data.columns]

        X = data[available_features].fillna(data[available_features].mean())
        y = np.full(len(X), 1)  # All samples are of the same class

        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, len(X) - 1))
        X_synthetic, y_synthetic = smote.fit_resample(X, y)

        # Create synthetic dataframe
        synthetic_df = pd.DataFrame(X_synthetic, columns=available_features)
        synthetic_df['signal'] = signal_type
        synthetic_df['time'] = pd.Timestamp.now()  # Placeholder time

        return synthetic_df.head(target_count)

    except Exception as e:
        print(f"SMOTE failed for {signal_type}: {e}")
        return pd.DataFrame()


def generate_pattern_based_data(signal_type, target_count):
    """Generate data based on known trading patterns."""
    synthetic_data = []

    if signal_type == 'BUY':
        # Bullish patterns
        patterns = [
            # RSI oversold + MACD positive
            {'RSI': 30, 'MACD': 0.5, 'MACD_signal': 0.3, 'MACD_hist': 0.2},
            # Price above SMA20 and SMA50
            {'SMA20': 3300, 'SMA50': 3290, 'RSI': 55, 'MACD': 0.2},
            # Strong momentum
            {'RSI': 65, 'MACD': 1.0, 'MACD_signal': 0.8, 'MACD_hist': 0.2},
        ]
    else:  # SELL
        # Bearish patterns
        patterns = [
            # RSI overbought + MACD negative
            {'RSI': 70, 'MACD': -0.5, 'MACD_signal': -0.3, 'MACD_hist': -0.2},
            # Price below SMA20 and SMA50
            {'SMA20': 3310, 'SMA50': 3320, 'RSI': 45, 'MACD': -0.2},
            # Weak momentum
            {'RSI': 35, 'MACD': -1.0, 'MACD_signal': -0.8, 'MACD_hist': -0.2},
        ]

    for i in range(target_count):
        # Select random pattern
        pattern = patterns[i % len(patterns)]

        # Generate base price
        base_price = 3300 + np.random.normal(0, 10)

        # Create synthetic sample
        sample = {
            'time': pd.Timestamp.now() + pd.Timedelta(minutes=i),
            'open': base_price + np.random.normal(0, 2),
            'high': base_price + np.random.normal(2, 1),
            'low': base_price + np.random.normal(-2, 1),
            'close': base_price + np.random.normal(0, 1),
            'SMA20': pattern.get('SMA20', base_price + np.random.normal(0, 5)),
            'SMA50': pattern.get('SMA50', base_price + np.random.normal(0, 5)),
            'RSI': pattern.get('RSI', 50) + np.random.normal(0, 5),
            'MACD': pattern.get('MACD', 0) + np.random.normal(0, 0.1),
            'MACD_signal': pattern.get('MACD_signal', 0) + np.random.normal(0, 0.1),
            'MACD_hist': pattern.get('MACD_hist', 0) + np.random.normal(0, 0.1),
            'signal': signal_type,
        }

        synthetic_data.append(sample)

    return pd.DataFrame(synthetic_data)


def main():
    """Main function to generate synthetic dataset."""
    try:
        synthetic_df = generate_synthetic_dataset()

        print("\n🎉 Synthetic Dataset Generation Completed!")
        print("=" * 60)
        print("Next steps:")
        print("1. Use the synthetic dataset for training new models")
        print("2. Test the models with balanced data")
        print("3. Compare performance with original models")

    except Exception as e:
        print(f"❌ Error during synthetic dataset generation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
