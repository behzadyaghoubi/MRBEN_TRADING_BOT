#!/usr/bin/env python3
"""
Start LSTM Retraining
Simple script to execute the LSTM retraining with real data
"""

import os
import sys
import subprocess

def main():
    print("🎯 Starting LSTM Retraining with Real Data")
    print("=" * 50)
    
    # Check if real data exists
    if not os.path.exists("data/real_market_sequences.npy"):
        print("❌ Real market sequences not found")
        return
    
    if not os.path.exists("data/real_market_labels.npy"):
        print("❌ Real market labels not found")
        return
    
    print("✅ Real market data found")
    print("🚀 Starting retraining process...")
    
    try:
        # Execute the retraining script
        result = subprocess.run([
            sys.executable, "retrain_lstm_with_real_data.py"
        ], capture_output=True, text=True, timeout=1800)  # 30 minutes timeout
        
        if result.returncode == 0:
            print("✅ Retraining completed successfully!")
            print("\n📊 Output:")
            print(result.stdout)
        else:
            print("❌ Retraining failed!")
            print("\n📊 Error:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Retraining timed out after 30 minutes")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 