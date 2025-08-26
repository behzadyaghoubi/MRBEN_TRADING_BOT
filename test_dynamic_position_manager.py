#!/usr/bin/env python3
"""
Test Dynamic Position Manager
Test the ATR-based trailing stop loss and dynamic take profit functionality
"""

import time
import json
from datetime import datetime
from dynamic_position_manager import DynamicPositionManager

def test_position_manager():
    """Test the dynamic position manager."""
    print("🧪 Testing Dynamic Position Manager")
    print("=" * 60)
    
    try:
        # Create position manager
        manager = DynamicPositionManager()
        
        if not manager.mt5_connected:
            print("❌ Failed to connect to MT5")
            return False
        
        print("✅ MT5 connection successful")
        
        # Test 1: Check current positions
        print("\n1️⃣ Testing position tracking...")
        manager.update_position_tracking()
        summary = manager.get_position_summary()
        print(f"📊 Current positions: {summary['total_positions']}")
        
        # Test 2: Check if new position can be opened
        print("\n2️⃣ Testing position limit check...")
        can_open = manager.can_open_new_position()
        print(f"📊 Can open new position: {can_open}")
        
        # Test 3: Get market data and calculate ATR
        print("\n3️⃣ Testing ATR calculation...")
        df = manager._get_market_data(100)
        if df is not None:
            atr = manager.calculate_atr(df)
            print(f"📊 Current ATR: {atr:.2f}")
            print(f"📊 ATR-based SL distance: {atr * manager.atr_multiplier:.2f}")
        else:
            print("❌ Failed to get market data")
        
        # Test 4: Simulate position management
        print("\n4️⃣ Testing position management logic...")
        if summary['total_positions'] > 0:
            print("📊 Managing existing positions...")
            manager.manage_positions()
        else:
            print("📊 No positions to manage")
        
        # Test 5: Configuration test
        print("\n5️⃣ Testing configuration...")
        print(f"📊 ATR Period: {manager.atr_period}")
        print(f"📊 ATR Multiplier: {manager.atr_multiplier}")
        print(f"📊 TP Trail Percent: {manager.tp_trail_percent}%")
        print(f"📊 Max Positions: {manager.max_positions}")
        print(f"📊 Min ATR Distance: {manager.min_atr_distance} points")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def run_quick_management_test():
    """Run a quick management test for 2 minutes."""
    print("\n🚀 Running Quick Management Test (2 minutes)")
    print("=" * 60)
    
    try:
        manager = DynamicPositionManager()
        
        if not manager.mt5_connected:
            print("❌ Failed to connect to MT5")
            return
        
        start_time = time.time()
        duration = 120  # 2 minutes
        
        while time.time() - start_time < duration:
            try:
                # Manage positions
                manager.manage_positions()
                
                # Print summary
                summary = manager.get_position_summary()
                if summary['total_positions'] > 0:
                    print(f"📊 {datetime.now().strftime('%H:%M:%S')} | "
                          f"Positions: {summary['total_positions']} | "
                          f"Profit: {summary['total_profit']:.2f}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                print("\n🛑 Test stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in test loop: {e}")
                time.sleep(30)
        
        print("\n✅ Quick management test completed!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

def main():
    """Main test function."""
    print("🎯 Dynamic Position Manager Test Suite")
    print("=" * 60)
    
    # Run basic tests
    if test_position_manager():
        print("\n" + "=" * 60)
        
        # Ask user if they want to run the quick management test
        response = input("\n🤔 Run quick management test for 2 minutes? (y/n): ")
        if response.lower() in ['y', 'yes']:
            run_quick_management_test()
    
    print("\n🏁 Test suite completed!")

if __name__ == "__main__":
    main() 