#!/usr/bin/env python3
"""
MR BEN - Test Bias Fix Script
=============================
Simple test to verify bias fix process
"""

import pandas as pd
import numpy as np

def main():
    print("🧪 Testing Bias Fix Process...")
    
    # Load data
    try:
        data = pd.read_csv('data/mrben_ai_signal_dataset.csv')
        print(f"✅ Data loaded successfully: {len(data)} records")
        
        # Check signal distribution
        signal_counts = data['signal'].value_counts()
        print(f"\n📊 Signal Distribution:")
        print(signal_counts)
        
        # Check features
        feature_columns = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist']
        available_features = [col for col in feature_columns if col in data.columns]
        print(f"\n🔍 Available features: {available_features}")
        
        if len(available_features) >= 2:
            print("✅ Sufficient features for training")
            
            # Create balanced sample
            hold_data = data[data['signal'] == 'HOLD'].sample(n=100, random_state=42)
            buy_data = data[data['signal'] == 'BUY'].sample(n=100, replace=True, random_state=42)
            sell_data = data[data['signal'] == 'SELL'].sample(n=100, replace=True, random_state=42)
            
            balanced_data = pd.concat([hold_data, buy_data, sell_data])
            balanced_counts = balanced_data['signal'].value_counts()
            
            print(f"\n⚖️ Balanced sample created:")
            print(balanced_counts)
            print("✅ Bias fix process is ready!")
            
        else:
            print("❌ Insufficient features for training")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 