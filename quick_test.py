#!/usr/bin/env python3
"""
Quick Test
"""

import os
import numpy as np
from datetime import datetime

def print_status(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def main():
    print_status("🧪 Quick Test Started")
    
    # Check data files
    sequences_path = "data/real_market_sequences.npy"
    labels_path = "data/real_market_labels.npy"
    
    if os.path.exists(sequences_path):
        print_status(f"✅ Sequences file found: {sequences_path}")
        size_mb = os.path.getsize(sequences_path) / (1024 * 1024)
        print_status(f"   Size: {size_mb:.1f} MB")
    else:
        print_status(f"❌ Sequences file not found: {sequences_path}")
        return
    
    if os.path.exists(labels_path):
        print_status(f"✅ Labels file found: {labels_path}")
        size_kb = os.path.getsize(labels_path) / 1024
        print_status(f"   Size: {size_kb:.1f} KB")
    else:
        print_status(f"❌ Labels file not found: {labels_path}")
        return
    
    # Check model files
    model_path = "models/mrben_lstm_real_data.h5"
    scaler_path = "models/mrben_lstm_real_data_scaler.save"
    
    if os.path.exists(model_path):
        print_status(f"✅ Model file found: {model_path}")
        size_kb = os.path.getsize(model_path) / 1024
        print_status(f"   Size: {size_kb:.1f} KB")
    else:
        print_status(f"❌ Model file not found: {model_path}")
        return
    
    if os.path.exists(scaler_path):
        print_status(f"✅ Scaler file found: {scaler_path}")
        size_kb = os.path.getsize(scaler_path) / 1024
        print_status(f"   Size: {size_kb:.1f} KB")
    else:
        print_status(f"❌ Scaler file not found: {scaler_path}")
        return
    
    # Load data
    try:
        sequences = np.load(sequences_path)
        labels = np.load(labels_path)
        print_status(f"✅ Data loaded successfully")
        print_status(f"   Sequences: {sequences.shape}")
        print_status(f"   Labels: {labels.shape}")
        
        # Check label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print_status(f"   Label distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(labels)) * 100
            signal_type = ["SELL", "HOLD", "BUY"][label]
            print_status(f"     {signal_type}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        print_status(f"❌ Error loading data: {e}")
        return
    
    print_status("🎉 Quick test completed successfully!")

if __name__ == "__main__":
    main()
    input("Press Enter to continue...") 