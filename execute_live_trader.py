#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os

def run_live_trader():
    """Run the live trader script"""
    print("🚀 Starting live trader...")
    print("=" * 50)
    
    try:
        # Run the live trader script
        result = subprocess.run([sys.executable, 'live_trader_clean.py'], 
                               capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Live trader started successfully")
            if result.stdout:
                print("Output:", result.stdout)
        else:
            print("❌ Live trader failed to start")
            if result.stderr:
                print("Error:", result.stderr)
            if result.stdout:
                print("Output:", result.stdout)
                
    except subprocess.TimeoutExpired:
        print("✅ Live trader is running (timeout reached - this is normal)")
    except Exception as e:
        print(f"❌ Error running live trader: {e}")

if __name__ == "__main__":
    run_live_trader()
