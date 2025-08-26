#!/usr/bin/env python3
"""
Quick MT5 Test
Quick verification of MT5 data access and signal generation
"""

import time
from live_trader_mt5 import MT5DataManager, MT5Config, MT5SignalGenerator

def quick_test():
    """Quick test of MT5 components."""
    
    print("⚡ Quick MT5 System Test...")
    print("=" * 40)
    
    # Test data manager
    print("1️⃣ Testing MT5 Data Manager...")
    data_manager = MT5DataManager("XAUUSD.PRO")
    
    # Get latest data
    df = data_manager.get_latest_data(100)
    print(f"   ✅ Data retrieved: {len(df)} rows")
    print(f"   📅 Latest time: {df['time'].iloc[-1]}")
    print(f"   💰 Latest price: {df['close'].iloc[-1]:.2f}")
    
    # Check technical indicators
    required_features = ['rsi', 'macd', 'atr']
    available_features = [f for f in required_features if f in df.columns]
    print(f"   📊 Technical indicators: {len(available_features)}/3")
    
    # Test current tick
    tick = data_manager.get_current_tick()
    if tick:
        print(f"   🔄 Current tick: Bid={tick['bid']:.2f}, Ask={tick['ask']:.2f}")
    else:
        print("   ⚠️ No tick data available")
    
    # Test signal generation
    print("\n2️⃣ Testing Signal Generation...")
    config = MT5Config()
    
    # Load models (simplified)
    lstm_model = None
    lstm_scaler = None
    ml_filter = None
    
    try:
        from tensorflow.keras.models import load_model
        import joblib
        
        model_path = 'models/lstm_balanced_model.h5'
        scaler_path = 'models/lstm_balanced_scaler.save'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            lstm_model = load_model(model_path)
            lstm_scaler = joblib.load(scaler_path)
            print("   ✅ LSTM model loaded")
        else:
            print("   ⚠️ LSTM model not found")
    except Exception as e:
        print(f"   ⚠️ Error loading LSTM: {e}")
    
    # Create signal generator
    signal_gen = MT5SignalGenerator(config, lstm_model, lstm_scaler, ml_filter)
    
    # Generate signal
    signal_data = signal_gen.generate_enhanced_signal(df)
    print(f"   🎯 Signal: {signal_data['signal']}")
    print(f"   🎯 Confidence: {signal_data['confidence']:.3f}")
    print(f"   🎯 Source: {signal_data.get('source', 'Unknown')}")
    
    # Test configuration
    print("\n3️⃣ Testing Configuration...")
    print(f"   📊 Symbol: {config.SYMBOL}")
    print(f"   📊 Volume: {config.VOLUME}")
    print(f"   📊 Threshold: {config.MIN_SIGNAL_CONFIDENCE}")
    print(f"   📊 MT5 Enabled: {config.ENABLE_MT5}")
    
    # Cleanup
    data_manager.shutdown()
    
    print("\n✅ Quick test completed!")
    return True

if __name__ == "__main__":
    import os
    quick_test() 