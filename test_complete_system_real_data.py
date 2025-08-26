#!/usr/bin/env python3
"""
Test Complete System with Real Data Model
Comprehensive test of the trading system with the new LSTM model trained on real data
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time

def test_mt5_connection():
    """Test MT5 connection."""
    print("🔌 Testing MT5 Connection...")
    
    try:
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return False
        
        # Load config
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        login = config.get('mt5_login', 1104123)
        password = config.get('mt5_password', '-4YcBgRd')
        server = config.get('mt5_server', 'OxSecurities-Demo')
        
        if not mt5.login(login=login, password=password, server=server):
            print("❌ MT5 login failed")
            return False
        
        print("✅ MT5 connected successfully")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"   Account: {account_info.login}")
            print(f"   Balance: {account_info.balance}")
            print(f"   Equity: {account_info.equity}")
        
        return True
        
    except Exception as e:
        print(f"❌ MT5 connection error: {e}")
        return False

def test_data_collection():
    """Test data collection from MT5."""
    print("\n📊 Testing Data Collection...")
    
    try:
        symbol = "XAUUSD.PRO"
        
        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to select symbol {symbol}")
            return None
        
        # Get recent data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        
        if rates is None or len(rates) == 0:
            print("❌ Failed to get market data")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"✅ Collected {len(df)} bars of data")
        print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"   Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Data collection error: {e}")
        return None

def test_real_data_files():
    """Test if real data files exist."""
    print("\n📁 Testing Real Data Files...")
    
    sequences_path = "data/real_market_sequences.npy"
    labels_path = "data/real_market_labels.npy"
    
    if not os.path.exists(sequences_path):
        print(f"❌ Sequences file not found: {sequences_path}")
        return False
    
    if not os.path.exists(labels_path):
        print(f"❌ Labels file not found: {labels_path}")
        return False
    
    # Load and check data
    try:
        sequences = np.load(sequences_path)
        labels = np.load(labels_path)
        
        print(f"✅ Real data files found:")
        print(f"   Sequences: {sequences.shape}")
        print(f"   Labels: {labels.shape}")
        
        # Check label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"   Label distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(labels)) * 100
            signal_type = ["SELL", "HOLD", "BUY"][label]
            print(f"     {signal_type}: {count} ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        return False

def test_model_files():
    """Test if model files exist."""
    print("\n🤖 Testing Model Files...")
    
    model_paths = [
        'models/mrben_lstm_real_data.h5',
        'models/mrben_lstm_real_data_scaler.save',
        'models/mrben_ai_signal_filter_xgb_balanced.joblib'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"✅ {path} ({size:.1f} KB)")
        else:
            print(f"❌ {path} not found")
    
    return True

def test_signal_generation():
    """Test signal generation with real data."""
    print("\n🎯 Testing Signal Generation...")
    
    try:
        # Import the trading system
        from live_trader_clean import MT5LiveTrader
        
        # Create trader instance
        trader = MT5LiveTrader()
        
        # Test data collection
        df = trader.data_manager.get_latest_data(bars=100)
        if df is None:
            print("❌ Failed to get data for signal generation")
            return False
        
        print(f"✅ Got {len(df)} bars for signal generation")
        
        # Test signal generation
        signal_data = trader.signal_generator.generate_enhanced_signal(df)
        
        print(f"✅ Signal generated:")
        print(f"   Signal: {signal_data['signal']} ({['SELL', 'HOLD', 'BUY'][signal_data['signal']]})")
        print(f"   Confidence: {signal_data['confidence']:.3f}")
        print(f"   Source: {signal_data.get('source', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal generation error: {e}")
        return False

def test_complete_pipeline():
    """Test the complete trading pipeline."""
    print("\n🚀 Testing Complete Pipeline...")
    
    try:
        # Import the trading system
        from live_trader_clean import MT5LiveTrader
        
        # Create trader instance
        trader = MT5LiveTrader()
        
        # Test data collection
        df = trader.data_manager.get_latest_data(bars=100)
        if df is None:
            print("❌ Failed to get data")
            return False
        
        # Test signal generation
        signal_data = trader.signal_generator.generate_enhanced_signal(df)
        
        # Test ATR calculation
        current_price = df['close'].iloc[-1]
        sl, tp = trader._calculate_atr_based_sl_tp(df, current_price, signal_data['signal'])
        
        print(f"✅ Complete pipeline test:")
        print(f"   Current Price: {current_price:.2f}")
        print(f"   Signal: {signal_data['signal']} ({['SELL', 'HOLD', 'BUY'][signal_data['signal']]})")
        print(f"   Confidence: {signal_data['confidence']:.3f}")
        print(f"   Stop Loss: {sl:.2f}")
        print(f"   Take Profit: {tp:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Complete System Test with Real Data Model")
    print("=" * 60)
    
    # Test 1: MT5 Connection
    mt5_ok = test_mt5_connection()
    
    # Test 2: Data Collection
    data_ok = test_data_collection()
    
    # Test 3: Real Data Files
    real_data_ok = test_real_data_files()
    
    # Test 4: Model Files
    model_ok = test_model_files()
    
    # Test 5: Signal Generation
    signal_ok = test_signal_generation()
    
    # Test 6: Complete Pipeline
    pipeline_ok = test_complete_pipeline()
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 30)
    print(f"MT5 Connection: {'✅' if mt5_ok else '❌'}")
    print(f"Data Collection: {'✅' if data_ok else '❌'}")
    print(f"Real Data Files: {'✅' if real_data_ok else '❌'}")
    print(f"Model Files: {'✅' if model_ok else '❌'}")
    print(f"Signal Generation: {'✅' if signal_ok else '❌'}")
    print(f"Complete Pipeline: {'✅' if pipeline_ok else '❌'}")
    
    if all([mt5_ok, data_ok, real_data_ok, model_ok, signal_ok, pipeline_ok]):
        print("\n🎉 All tests passed! System is ready for live trading.")
        print("\n🎯 Next Steps:")
        print("   1. Run live_trader_clean.py for live trading")
        print("   2. Monitor signal distribution")
        print("   3. Check trade performance")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")
    
    # Cleanup
    if mt5_ok:
        mt5.shutdown()

if __name__ == "__main__":
    main() 