#!/usr/bin/env python3
"""
Quick Tests for MR BEN Live Trader Improvements
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_lot_sizing():
    """Test lot sizing with different inputs."""
    print("🧪 Testing lot sizing...")
    
    from live_trader_clean import EnhancedRiskManager
    
    risk_manager = EnhancedRiskManager()
    
    # Test cases
    test_cases = [
        (10000, 0.02, 100, "XAUUSD.PRO"),  # Normal case
        (10000, 0.02, 50, "XAUUSD.PRO"),   # Closer SL
        (10000, 0.02, 200, "XAUUSD.PRO"),  # Further SL
        (5000, 0.01, 100, "XAUUSD.PRO"),   # Lower balance
    ]
    
    for balance, risk, sl_distance, symbol in test_cases:
        lot = risk_manager.calculate_lot_size(balance, risk, sl_distance, symbol)
        print(f"   Balance: {balance}, Risk: {risk}, SL: {sl_distance} -> Lot: {lot:.4f}")
        
        # Check if lot is in valid range
        if 0.01 <= lot <= 0.1:
            print("   ✅ Lot size in valid range")
        else:
            print("   ❌ Lot size out of range")
    
    print("✅ Lot sizing test completed")

def test_feature_schema():
    """Test if all required features exist in DataFrame."""
    print("\n🧪 Testing feature schema...")
    
    from live_trader_clean import MT5DataManager
    
    data_manager = MT5DataManager("XAUUSD.PRO")
    df = data_manager.get_latest_data(100)
    
    required_features = ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr']
    
    missing_features = []
    for feature in required_features:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        print(f"   ❌ Missing features: {missing_features}")
    else:
        print("   ✅ All required features present")
    
    print(f"   Available features: {list(df.columns)}")
    print("✅ Feature schema test completed")

def test_price_rounding():
    """Test price rounding function."""
    print("\n🧪 Testing price rounding...")
    
    from live_trader_clean import round_price
    
    test_prices = [3300.123456, 3300.987654, 3300.0, 3300.5]
    
    for price in test_prices:
        rounded = round_price("XAUUSD.PRO", price)
        print(f"   {price} -> {rounded}")
    
    print("✅ Price rounding test completed")

def test_config_loading():
    """Test configuration loading."""
    print("\n🧪 Testing configuration loading...")
    
    from live_trader_clean import MT5Config
    
    config = MT5Config()
    
    print(f"   Symbol: {config.SYMBOL}")
    print(f"   Volume: {config.VOLUME}")
    print(f"   Magic: {config.MAGIC}")
    print(f"   Max Daily Loss: {config.MAX_DAILY_LOSS}")
    print(f"   Max Trades Per Day: {config.MAX_TRADES_PER_DAY}")
    print(f"   Trading Sessions: {config.TRADING_SESSIONS}")
    print(f"   Sleep Seconds: {config.SLEEP_SECONDS}")
    
    print("✅ Configuration loading test completed")

def test_session_detection():
    """Test trading session detection."""
    print("\n🧪 Testing session detection...")
    
    from live_trader_clean import MT5LiveTrader
    
    trader = MT5LiveTrader()
    current_session = trader._current_session()
    
    print(f"   Current session: {current_session}")
    print(f"   Allowed sessions: {trader.config.TRADING_SESSIONS}")
    
    if current_session in trader.config.TRADING_SESSIONS:
        print("   ✅ Current session is allowed")
    else:
        print("   ⚠️ Current session is not allowed")
    
    print("✅ Session detection test completed")

if __name__ == "__main__":
    print("🎯 MR BEN Live Trader - Quick Tests")
    print("=" * 50)
    
    test_lot_sizing()
    test_feature_schema()
    test_price_rounding()
    test_config_loading()
    test_session_detection()
    
    print("\n🎉 All quick tests completed!")
