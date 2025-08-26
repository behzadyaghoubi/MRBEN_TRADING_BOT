#!/usr/bin/env python3
"""
Quick Shadow Mode Test - Demonstrates AI decision-making without execution
"""
import logging
import time
from datetime import datetime
from live_trader_ai_enhanced import EnhancedAILiveTrader

def run_shadow_test(duration_minutes=5):
    """Run shadow mode test for specified duration"""
    
    print("🤖 SHADOW MODE TEST STARTING")
    print("=" * 50)
    
    # Create trader
    trader = EnhancedAILiveTrader()
    
    # Start the system
    trader.start()
    
    print(f"⏱️ Running for {duration_minutes} minutes in Shadow mode...")
    print("📊 Watch for these key logs:")
    print("  - 'Conformal gate loaded'")
    print("  - 'Conformal: accept=True/False p=...'") 
    print("  - '[SHADOW] Would execute...'")
    print("=" * 50)
    
    try:
        # Run for specified duration
        time.sleep(duration_minutes * 60)
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    
    finally:
        # Stop the system
        trader.stop()
        
        # Print session statistics
        print("\n" + "=" * 50)
        print("📊 SHADOW MODE TEST RESULTS")
        print("=" * 50)
        print(f"Signals Generated: {trader.session_stats['signals_generated']}")
        print(f"Conformal Accepted: {trader.session_stats['conformal_accepted']}")
        print(f"Conformal Rejected: {trader.session_stats['conformal_rejected']}")
        print(f"Risk Rejected: {trader.session_stats['risk_rejected']}")
        print(f"Trades (Shadow): {trader.session_stats['trades_executed']}")
        
        # Get policy brain statistics if available
        if trader.policy_brain:
            brain_stats = trader.policy_brain.get_statistics()
            print(f"Brain Acceptance Rate: {brain_stats.get('acceptance_rate', 0.0):.1%}")
        
        print("=" * 50)

if __name__ == "__main__":
    run_shadow_test(duration_minutes=2)  # Short test for demonstration
