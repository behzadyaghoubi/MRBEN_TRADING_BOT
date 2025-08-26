#!/usr/bin/env python3
"""
Run LSTM Retraining with Real Data
Simple script to execute the LSTM retraining process
"""

import subprocess
import sys
import os

def run_retraining():
    """Run the LSTM retraining process."""
    print("🎯 Starting LSTM Retraining with Real Data")
    print("=" * 50)
    
    # Check if real data exists
    if not os.path.exists("data/real_market_sequences.npy"):
        print("❌ Real market data not found. Please run collect_real_data_for_lstm.py first.")
        return False
    
    if not os.path.exists("data/real_market_labels.npy"):
        print("❌ Real market labels not found. Please run collect_real_data_for_lstm.py first.")
        return False
    
    print("✅ Real market data found. Starting retraining...")
    
    try:
        # Run the retraining script
        result = subprocess.run([
            sys.executable, "retrain_lstm_with_real_data.py"
        ], capture_output=True, text=True, timeout=1800)  # 30 minutes timeout
        
        if result.returncode == 0:
            print("✅ LSTM retraining completed successfully!")
            print("\n📊 Output:")
            print(result.stdout)
            return True
        else:
            print("❌ LSTM retraining failed!")
            print("\n📊 Error:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Retraining timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running retraining: {e}")
        return False

if __name__ == "__main__":
    success = run_retraining()
    
    if success:
        print("\n🎉 Retraining completed! Next steps:")
        print("1. Test the system with: test_complete_system_real_data.py")
        print("2. Run live trading with: live_trader_clean.py")
    else:
        print("\n⚠️ Retraining failed. Please check the errors above.") 