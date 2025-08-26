#!/usr/bin/env python3
"""
Quick Shadow Demo - 3 minutes to demonstrate key log patterns
"""
import time
from datetime import datetime
from live_trader_ai_enhanced import EnhancedAILiveTrader

def run_quick_demo():
    print("🎯 QUICK SHADOW DEMO - Looking for 3 key patterns:")
    print("  1. Conformal: accept=False (rejection)")
    print("  2. Risk Governor rejection") 
    print("  3. [SHADOW] Would execute (should execute)")
    print("=" * 50)
    
    trader = EnhancedAILiveTrader()
    print(f"Mode: {trader.config.get('ai_control', {}).get('mode', 'unknown')}")
    
    # Quick verification that conformal is working
    print("\n🧪 Testing Conformal Gate directly:")
    if trader.policy_brain and trader.policy_brain.conformal_gate:
        # Test with dummy features
        test_features = {
            "close": 2500.0, "ret": 0.001, "sma_20": 2499.0, "sma_50": 2498.0,
            "atr": 15.0, "rsi": 55.0, "macd": 0.5, "macd_signal": 0.3,
            "hour": 10.0, "dow": 1.0
        }
        accept, p_hat, nonconf = trader.policy_brain.conformal_gate.accept(test_features)
        print(f"Sample conformal test: accept={accept}, p={p_hat:.3f}, nonconf={nonconf:.3f}")
    
    print("\n🚀 Starting 3-minute live test...")
    trader.start()
    
    try:
        time.sleep(180)  # 3 minutes
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    finally:
        trader.stop()
        
        print("\n📊 Quick Results:")
        if hasattr(trader, 'session_stats'):
            stats = trader.session_stats
            for key, value in stats.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    run_quick_demo()
