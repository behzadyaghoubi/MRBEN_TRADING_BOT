#!/usr/bin/env python3
"""
Test Data Loading
"""

import os
from datetime import datetime

import numpy as np


def print_status(message, level="INFO"):
    """Print status with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def main():
    """Test data loading"""
    print_status("🧪 Testing Data Loading", "START")

    # Check data files
    sequences_path = "data/real_market_sequences.npy"
    labels_path = "data/real_market_labels.npy"

    print_status(f"Checking file: {sequences_path}", "INFO")
    if os.path.exists(sequences_path):
        print_status("✅ Sequences file found", "SUCCESS")
        size_mb = os.path.getsize(sequences_path) / (1024 * 1024)
        print_status(f"   Size: {size_mb:.1f} MB", "INFO")
    else:
        print_status("❌ Sequences file not found", "ERROR")
        return False

    print_status(f"Checking file: {labels_path}", "INFO")
    if os.path.exists(labels_path):
        print_status("✅ Labels file found", "SUCCESS")
        size_kb = os.path.getsize(labels_path) / 1024
        print_status(f"   Size: {size_kb:.1f} KB", "INFO")
    else:
        print_status("❌ Labels file not found", "ERROR")
        return False

    # Load data
    print_status("📊 Loading sequences...", "INFO")
    try:
        sequences = np.load(sequences_path)
        print_status(f"✅ Sequences loaded: {sequences.shape}", "SUCCESS")
    except Exception as e:
        print_status(f"❌ Error loading sequences: {e}", "ERROR")
        return False

    print_status("📊 Loading labels...", "INFO")
    try:
        labels = np.load(labels_path)
        print_status(f"✅ Labels loaded: {labels.shape}", "SUCCESS")
    except Exception as e:
        print_status(f"❌ Error loading labels: {e}", "ERROR")
        return False

    # Check data
    print_status("🔍 Analyzing data...", "INFO")
    print_status(f"   Sequences shape: {sequences.shape}", "INFO")
    print_status(f"   Labels shape: {labels.shape}", "INFO")
    print_status(f"   Data type: {sequences.dtype}", "INFO")
    print_status(f"   Memory usage: {sequences.nbytes / (1024*1024):.1f} MB", "INFO")

    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print_status("   Label distribution:", "INFO")
    for label, count in zip(unique_labels, counts, strict=False):
        percentage = (count / len(labels)) * 100
        signal_type = ["SELL", "HOLD", "BUY"][label]
        print_status(f"     {signal_type}: {count} ({percentage:.1f}%)", "INFO")

    print_status("✅ Data loading test completed successfully!", "SUCCESS")
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print_status("🎉 All tests passed!", "SUCCESS")
    else:
        print_status("❌ Tests failed!", "ERROR")

    print_status("Press Enter to continue...", "INFO")
    input()
