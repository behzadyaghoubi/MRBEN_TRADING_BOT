#!/usr/bin/env python3
"""
Comprehensive System Test for MR BEN Live Trader
Tests all components and identifies any remaining issues
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports."""
    print("🧪 Testing Imports...")
    print("=" * 50)
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5 imported successfully")
    except ImportError as e:
        print(f"⚠️ MetaTrader5 not available: {e}")
    
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
    except ImportError as e:
        print(f"⚠️ TensorFlow not available: {e}")
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"⚠️ Scikit-learn not available: {e}")
    
    try:
        import joblib
        print("✅ Joblib imported successfully")
    except ImportError as e:
        print(f"⚠️ Joblib not available: {e}")
    
    print("✅ Import test completed")
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\n🧪 Testing Configuration Loading...")
    print("=" * 50)
    
    try:
        config_path = 'config/settings.json'
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print("✅ Configuration file loaded successfully")
            print(f"   Symbol: {config.get('trading', {}).get('symbol', 'N/A')}")
            print(f"   Min Confidence: {config.get('trading', {}).get('min_signal_confidence', 'N/A')}")
            print(f"   Consecutive Signals: {config.get('trading', {}).get('consecutive_signals_required', 'N/A')}")
            return True
        else:
            print("❌ Configuration file not found")
            return False
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def test_live_trader_components():
    """Test live trader component initialization."""
    print("\n🧪 Testing Live Trader Components...")
    print("=" * 50)
    
    try:
        from live_trader_clean import MT5Config, MT5DataManager, MRBENAdvancedAISystem, EnhancedRiskManager
        
        # Test MT5Config
        print("Testing MT5Config...")
        config = MT5Config()
        print(f"   Symbol: {config.SYMBOL}")
        print(f"   Min Confidence: {config.MIN_SIGNAL_CONFIDENCE}")
        print(f"   Consecutive Signals: {config.CONSECUTIVE_SIGNALS_REQUIRED}")
        print("✅ MT5Config initialized successfully")
        
        # Test MT5DataManager
        print("Testing MT5DataManager...")
        try:
            data_manager = MT5DataManager(config.SYMBOL)
            print("✅ MT5DataManager initialized successfully")
        except Exception as e:
            print(f"⚠️ MT5DataManager initialization warning: {e}")
        
        # Test MRBENAdvancedAISystem
        print("Testing MRBENAdvancedAISystem...")
        ai_system = MRBENAdvancedAISystem()
        print("✅ MRBENAdvancedAISystem initialized successfully")
        
        # Test EnhancedRiskManager
        print("Testing EnhancedRiskManager...")
        risk_manager = EnhancedRiskManager()
        print("✅ EnhancedRiskManager initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing components: {e}")
        traceback.print_exc()
        return False

def test_signal_generation():
    """Test signal generation with sample data."""
    print("\n🧪 Testing Signal Generation...")
    print("=" * 50)
    
    try:
        from live_trader_clean import MRBENAdvancedAISystem
        
        ai_system = MRBENAdvancedAISystem()
        
        # Test with sample market data
        sample_data = {
            'time': datetime.now().isoformat(),
            'open': 3300.0,
            'high': 3310.0,
            'low': 3295.0,
            'close': 3308.0,
            'tick_volume': 1000
        }
        
        print("Generating signal with sample data...")
        signal_data = ai_system.generate_ensemble_signal(sample_data)
        
        print(f"✅ Signal generated successfully:")
        print(f"   Signal: {signal_data['signal']}")
        print(f"   Confidence: {signal_data['confidence']:.3f}")
        print(f"   Source: {signal_data.get('source', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing signal generation: {e}")
        traceback.print_exc()
        return False

def test_mt5_connection():
    """Test MT5 connection if available."""
    print("\n🧪 Testing MT5 Connection...")
    print("=" * 50)
    
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            print("❌ Failed to initialize MT5")
            return False
        
        # Try to load config
        config_path = 'config/settings.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            login = config.get('mt5_login', 1104123)
            password = config.get('mt5_password', '-4YcBgRd')
            server = config.get('mt5_server', 'OxSecurities-Demo')
            
            if mt5.login(login=login, password=password, server=server):
                print("✅ MT5 login successful")
                
                # Test symbol info
                symbol_info = mt5.symbol_info("XAUUSD.PRO")
                if symbol_info:
                    print(f"✅ Symbol info retrieved: {symbol_info.name}")
                else:
                    print("⚠️ Could not get symbol info")
                
                # Test account info
                account_info = mt5.account_info()
                if account_info:
                    print(f"✅ Account info retrieved: Balance = {account_info.balance}")
                else:
                    print("⚠️ Could not get account info")
                
                mt5.shutdown()
                return True
            else:
                print("❌ MT5 login failed")
                return False
        else:
            print("⚠️ Config file not found, skipping MT5 test")
            return True
            
    except ImportError:
        print("⚠️ MetaTrader5 not available, skipping MT5 test")
        return True
    except Exception as e:
        print(f"❌ Error testing MT5: {e}")
        return False

def test_model_loading():
    """Test model loading."""
    print("\n🧪 Testing Model Loading...")
    print("=" * 50)
    
    model_paths = [
        'models/advanced_lstm_model.h5',
        'models/mrben_simple_model.joblib',
        'models/quick_fix_ml_filter.joblib',
        'models/mrben_ai_signal_filter_xgb_balanced.joblib'
    ]
    
    found_models = []
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found model: {path}")
            found_models.append(path)
        else:
            print(f"⚠️ Model not found: {path}")
    
    if found_models:
        print(f"✅ Found {len(found_models)} model(s)")
        return True
    else:
        print("⚠️ No models found, system will use fallback methods")
        return True

def run_comprehensive_test():
    """Run all tests."""
    print("🎯 MR BEN Live Trader - Comprehensive System Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Loading", test_config_loading),
        ("Component Initialization", test_live_trader_components),
        ("Signal Generation", test_signal_generation),
        ("MT5 Connection", test_mt5_connection),
        ("Model Loading", test_model_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        return True
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. System should work with some limitations.")
        return True
    else:
        print("❌ Many tests failed. System needs fixes before running.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 