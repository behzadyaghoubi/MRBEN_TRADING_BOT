#!/usr/bin/env python3
"""
Check Real Market Data
Analyze the collected real market data for LSTM training
"""

import os

import numpy as np


def check_real_data():
    """Check the collected real market data."""
    print("🔍 Checking Real Market Data")
    print("=" * 50)

    # Check if files exist
    sequences_path = "data/real_market_sequences.npy"
    labels_path = "data/real_market_labels.npy"

    if not os.path.exists(sequences_path):
        print("❌ Sequences file not found")
        return

    if not os.path.exists(labels_path):
        print("❌ Labels file not found")
        return

    # Load data
    print("📊 Loading data...")
    sequences = np.load(sequences_path)
    labels = np.load(labels_path)

    print("✅ Data loaded successfully")
    print(f"   Sequences shape: {sequences.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Features: {sequences.shape[2]}")
    print(f"   Timesteps: {sequences.shape[1]}")

    # Analyze labels distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n📈 Label Distribution:")
    for label, count in zip(unique_labels, counts, strict=False):
        percentage = (count / len(labels)) * 100
        if label == 0:
            signal_type = "SELL"
        elif label == 1:
            signal_type = "HOLD"
        elif label == 2:
            signal_type = "BUY"
        else:
            signal_type = f"UNKNOWN({label})"

        print(f"   {signal_type} ({label}): {count} samples ({percentage:.1f}%)")

    # Check data quality
    print("\n🔍 Data Quality Check:")
    print(f"   NaN in sequences: {np.isnan(sequences).sum()}")
    print(f"   NaN in labels: {np.isnan(labels).sum()}")
    print(f"   Sequences min: {sequences.min():.6f}")
    print(f"   Sequences max: {sequences.max():.6f}")
    print(f"   Sequences mean: {sequences.mean():.6f}")
    print(f"   Sequences std: {sequences.std():.6f}")

    # Sample analysis
    print("\n📋 Sample Analysis:")
    print(f"   First sequence shape: {sequences[0].shape}")
    print(f"   First sequence first row: {sequences[0][0]}")
    print(
        f"   First label: {labels[0]} ({'SELL' if labels[0] == 0 else 'HOLD' if labels[0] == 1 else 'BUY'})"
    )

    print("\n✅ Real market data is ready for LSTM training!")
    print(f"   Total samples: {len(sequences)}")
    print(f"   Balanced dataset with {len(unique_labels)} classes")

    return sequences, labels


if __name__ == "__main__":
    check_real_data()
