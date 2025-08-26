#!/usr/bin/env python3
"""
Test script for Enhanced MR BEN Trading System
Tests the new features: Dynamic TP/SL, Trailing Stops, and Adaptive Confidence
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_risk_manager():
    """Test Enhanced Risk Manager functionality"""
    print("🧪 Testing Enhanced Risk Manager...")
    
    try:
        from enhanced_risk_manager import EnhancedRiskManager
        
        # Create risk manager instance
        rm = EnhancedRiskManager(
            base_risk=0.02,
            min_lot=0.01,
            max_lot=0.1,
            max_open_trades=2,
            atr_period=14,
            sl_atr_multiplier=2.0,
            tp_atr_multiplier=4.0,
            trailing_atr_multiplier=1.5,
            base_confidence_threshold=0.5,
            adaptive_confidence=True
        )
        
        print("✅ Enhanced Risk Manager created successfully")
        
        # Test confidence threshold
        threshold = rm.get_current_confidence_threshold()
        print(f"✅ Current confidence threshold: {threshold:.3f}")
        
        # Test performance update
        rm.update_performance(100)  # Profitable trade
        rm.update_performance(-50)  # Loss trade
        rm.update_performance(75)   # Profitable trade
        
        new_threshold = rm.get_current_confidence_threshold()
        print(f"✅ Updated confidence threshold: {new_threshold:.3f}")
        
        # Test trade validation
        can_trade = rm.can_open_new_trade(10000, 10000, 1)
        print(f"✅ Can open new trade: {can_trade}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Risk Manager test failed: {e}")
        return False

def test_dynamic_sl_tp():
    """Test dynamic SL/TP calculation"""
    print("\n🧪 Testing Dynamic SL/TP Calculation...")
    
    try:
        from enhanced_risk_manager import EnhancedRiskManager
        
        rm = EnhancedRiskManager()
        
        # Mock ATR calculation (since we don't have MT5 connection in test)
        def mock_get_atr(symbol, timeframe=5, bars=100):
            return 0.0025  # Mock ATR value
        
        rm.get_atr = mock_get_atr
        
        # Test BUY signal
        sl, tp = rm.calculate_dynamic_sl_tp("XAUUSD", 2000.0, "BUY")
        print(f"✅ BUY Signal - Entry: 2000.0, SL: {sl:.5f}, TP: {tp:.5f}")
        
        # Test SELL signal
        sl, tp = rm.calculate_dynamic_sl_tp("XAUUSD", 2000.0, "SELL")
        print(f"✅ SELL Signal - Entry: 2000.0, SL: {sl:.5f}, TP: {tp:.5f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dynamic SL/TP test failed: {e}")
        return False

def test_trailing_stop_logic():
    """Test trailing stop logic"""
    print("\n🧪 Testing Trailing Stop Logic...")
    
    try:
        from enhanced_risk_manager import EnhancedRiskManager
        
        rm = EnhancedRiskManager()
        
        # Add mock trailing stop
        rm.add_trailing_stop(12345, 2000.0, 1995.0, True)  # BUY position
        
        print(f"✅ Added trailing stop for ticket 12345")
        print(f"✅ Active trailing stops: {len(rm.trailing_stops)}")
        
        # Test trailing stop removal
        rm.remove_trailing_stop(12345)
        print(f"✅ Removed trailing stop for ticket 12345")
        print(f"✅ Active trailing stops: {len(rm.trailing_stops)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trailing stop test failed: {e}")
        return False

def test_adaptive_confidence():
    """Test adaptive confidence threshold logic"""
    print("\n🧪 Testing Adaptive Confidence Thresholds...")
    
    try:
        from enhanced_risk_manager import EnhancedRiskManager
        
        rm = EnhancedRiskManager(
            base_confidence_threshold=0.5,
            adaptive_confidence=True,
            performance_window=10,
            confidence_adjustment_factor=0.1
        )
        
        initial_threshold = rm.get_current_confidence_threshold()
        print(f"✅ Initial threshold: {initial_threshold:.3f}")
        
        # Simulate good performance
        for _ in range(8):
            rm.update_performance(100)  # Profitable trades
        
        good_performance_threshold = rm.get_current_confidence_threshold()
        print(f"✅ After good performance: {good_performance_threshold:.3f}")
        
        # Simulate poor performance
        for _ in range(8):
            rm.update_performance(-50)  # Loss trades
        
        poor_performance_threshold = rm.get_current_confidence_threshold()
        print(f"✅ After poor performance: {poor_performance_threshold:.3f}")
        
        # Verify adaptive behavior
        if good_performance_threshold < initial_threshold and poor_performance_threshold > initial_threshold:
            print("✅ Adaptive confidence working correctly")
            return True
        else:
            print("❌ Adaptive confidence not working as expected")
            return False
        
    except Exception as e:
        print(f"❌ Adaptive confidence test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration...")
    
    try:
        with open('enhanced_config.json', 'r') as f:
            config = json.load(f)
        
        print("✅ Configuration loaded successfully")
        print(f"✅ Symbol: {config['trading']['symbol']}")
        print(f"✅ Base Risk: {config['risk_management']['base_risk']}")
        print(f"✅ Adaptive Confidence: {config['confidence_thresholds']['adaptive_confidence']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("\n📊 Creating Sample Data...")
    
    # Create sample OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    np.random.seed(42)
    
    base_price = 2000.0
    prices = []
    for i in range(100):
        change = np.random.normal(0, 0.5)
        base_price += change
        prices.append(base_price)
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.2)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.2)) for p in prices],
        'close': prices,
        'volume': np.random.randint(100, 1000, 100)
    })
    
    # Save sample data
    df.to_csv('sample_test_data.csv', index=False)
    print("✅ Sample data created: sample_test_data.csv")
    
    return df

def main():
    """Run all tests"""
    print("🚀 Enhanced MR BEN Trading System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Enhanced Risk Manager", test_enhanced_risk_manager),
        ("Dynamic SL/TP", test_dynamic_sl_tp),
        ("Trailing Stop Logic", test_trailing_stop_logic),
        ("Adaptive Confidence", test_adaptive_confidence)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced system is ready for use.")
        print("\n📋 Next Steps:")
        print("1. Run: python enhanced_live_runner.py")
        print("2. Monitor logs in logs/enhanced_live_runner.log")
        print("3. Check trade logs in enhanced_live_trades.csv")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    # Create sample data for further testing
    create_sample_data()

if __name__ == "__main__":
    main() 