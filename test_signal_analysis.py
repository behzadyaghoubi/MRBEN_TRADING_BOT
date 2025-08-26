#!/usr/bin/env python3
"""
Signal Analysis Test
Test different signal scenarios and analyze the system behavior
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5

# Import our modules
from live_trader_clean import MT5Config, MT5SignalGenerator, MT5DataManager

def test_signal_generation():
    """Test signal generation with different scenarios."""
    print("🔍 Testing Signal Generation")
    print("=" * 60)
    
    try:
        # Initialize components
        config = MT5Config()
        data_manager = MT5DataManager()
        
        # Load LSTM model
        lstm_model = None
        lstm_scaler = None
        if os.path.exists('models/mrben_lstm_model.h5'):
            from tensorflow.keras.models import load_model
            lstm_model = load_model('models/mrben_lstm_model.h5')
            print("✅ LSTM model loaded")
        
        if os.path.exists('models/mrben_lstm_scaler.save'):
            import joblib
            lstm_scaler = joblib.load('models/mrben_lstm_scaler.save')
            print("✅ LSTM scaler loaded")
        
        # Load ML filter
        ml_filter = None
        if os.path.exists('models/mrben_ai_signal_filter_xgb.joblib'):
            from ai_filter import AISignalFilter
            ml_filter = AISignalFilter('models/mrben_ai_signal_filter_xgb.joblib')
            print("✅ ML filter loaded")
        
        # Create signal generator
        signal_generator = MT5SignalGenerator(config, lstm_model, lstm_scaler, ml_filter)
        
        # Get current market data
        df = data_manager.get_latest_data(500)
        if df is None:
            print("❌ Failed to get market data")
            return
        
        print(f"📊 Data shape: {df.shape}")
        print(f"📊 Latest price: {df['close'].iloc[-1]:.2f}")
        
        # Test 1: Generate current signal
        print("\n1️⃣ Testing Current Signal Generation")
        print("-" * 40)
        
        signal = signal_generator.generate_enhanced_signal(df)
        print(f"Final Signal: {signal['signal']} (1=BUY, -1=SELL, 0=HOLD)")
        print(f"Confidence: {signal['confidence']:.3f}")
        print(f"Source: {signal.get('source', 'Unknown')}")
        
        # Test 2: Test LSTM signal generation
        print("\n2️⃣ Testing LSTM Signal Generation")
        print("-" * 40)
        
        lstm_signal = signal_generator._generate_lstm_signal(df)
        print(f"LSTM Signal: {lstm_signal['signal']}")
        print(f"LSTM Confidence: {lstm_signal['confidence']:.3f}")
        if 'raw_prediction' in lstm_signal:
            raw_pred = lstm_signal['raw_prediction']
            print(f"LSTM Raw Predictions: SELL={raw_pred[0]:.3f}, HOLD={raw_pred[1]:.3f}, BUY={raw_pred[2]:.3f}")
        
        # Test 3: Test Technical Analysis signal
        print("\n3️⃣ Testing Technical Analysis Signal")
        print("-" * 40)
        
        ta_signal = signal_generator._generate_technical_signal(df)
        print(f"TA Signal: {ta_signal['signal']}")
        print(f"TA Confidence: {ta_signal['confidence']:.3f}")
        print(f"RSI: {ta_signal.get('rsi', 'N/A'):.2f}")
        print(f"MACD: {ta_signal.get('macd', 'N/A'):.2f}")
        
        # Test 4: Test ML filter
        print("\n4️⃣ Testing ML Filter")
        print("-" * 40)
        
        if ml_filter:
            features = [lstm_signal['signal'], lstm_signal['confidence'], 
                       ta_signal['signal'], ta_signal['confidence']]
            ml_result = ml_filter.filter_signal_with_confidence(features)
            print(f"ML Filter Result: {ml_result}")
        else:
            print("ML filter not available")
        
        # Test 5: Market direction analysis
        print("\n5️⃣ Market Direction Analysis")
        print("-" * 40)
        
        # Calculate moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        print(f"Current Price: {current_price:.2f}")
        print(f"SMA 20: {sma_20:.2f}")
        print(f"SMA 50: {sma_50:.2f}")
        
        if current_price > sma_20 > sma_50:
            print("📈 Market Direction: UPTREND (Bullish)")
        elif current_price < sma_20 < sma_50:
            print("📉 Market Direction: DOWNTREND (Bearish)")
        else:
            print("➡️ Market Direction: SIDEWAYS (Neutral)")
        
        # RSI analysis
        rsi = df['rsi'].iloc[-1]
        print(f"RSI: {rsi:.2f}")
        if rsi > 70:
            print("🔴 RSI: Overbought (Potential SELL)")
        elif rsi < 30:
            print("🟢 RSI: Oversold (Potential BUY)")
        else:
            print("🟡 RSI: Neutral")
        
        # Test 6: Simulate different scenarios
        print("\n6️⃣ Simulating Different Scenarios")
        print("-" * 40)
        
        # Create synthetic data with different patterns
        scenarios = [
            ("Bullish", {"rsi": 25, "macd": 0.5, "close": current_price + 10}),
            ("Bearish", {"rsi": 75, "macd": -0.5, "close": current_price - 10}),
            ("Neutral", {"rsi": 50, "macd": 0.0, "close": current_price})
        ]
        
        for scenario_name, scenario_data in scenarios:
            print(f"\n📊 Testing {scenario_name} scenario:")
            
            # Create modified dataframe
            test_df = df.copy()
            test_df.iloc[-1, test_df.columns.get_loc('rsi')] = scenario_data['rsi']
            test_df.iloc[-1, test_df.columns.get_loc('macd')] = scenario_data['macd']
            test_df.iloc[-1, test_df.columns.get_loc('close')] = scenario_data['close']
            
            # Generate signal
            test_signal = signal_generator.generate_enhanced_signal(test_df)
            print(f"   Signal: {test_signal['signal']} | Confidence: {test_signal['confidence']:.3f}")
        
        print("\n✅ Signal analysis completed!")
        
    except Exception as e:
        print(f"❌ Error in signal analysis: {e}")
        import traceback
        traceback.print_exc()

def test_market_conditions():
    """Test current market conditions."""
    print("\n🌍 Testing Current Market Conditions")
    print("=" * 60)
    
    try:
        # Initialize MT5
        if not mt5.initialize():
            print("❌ Failed to initialize MT5")
            return
        
        # Load config
        config_path = 'config/settings.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            login = config.get('mt5_login', 1104123)
            password = config.get('mt5_password', '-4YcBgRd')
            server = config.get('mt5_server', 'OxSecurities-Demo')
        else:
            login = 1104123
            password = '-4YcBgRd'
            server = 'OxSecurities-Demo'
        
        if not mt5.login(login=login, password=password, server=server):
            print("❌ Failed to login to MT5")
            return
        
        symbol = "XAUUSD.PRO"
        
        # Get current tick
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print(f"📊 Current Bid: {tick.bid:.2f}")
            print(f"📊 Current Ask: {tick.ask:.2f}")
            print(f"📊 Spread: {(tick.ask - tick.bid):.2f}")
        
        # Get recent price history
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        if rates is not None:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate trends
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            print(f"\n📈 Price Analysis:")
            print(f"   Current Price: {current_price:.2f}")
            print(f"   SMA 20: {sma_20:.2f}")
            print(f"   SMA 50: {sma_50:.2f}")
            
            if current_price > sma_20 > sma_50:
                print("   📈 Trend: UPTREND")
            elif current_price < sma_20 < sma_50:
                print("   📉 Trend: DOWNTREND")
            else:
                print("   ➡️ Trend: SIDEWAYS")
            
            # Calculate price change
            price_change = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            print(f"   📊 Price Change: {price_change:.2f}%")
        
        mt5.shutdown()
        
    except Exception as e:
        print(f"❌ Error in market analysis: {e}")

def main():
    """Main function."""
    print("🎯 Signal Analysis Test Suite")
    print("=" * 60)
    
    # Test signal generation
    test_signal_generation()
    
    # Test market conditions
    test_market_conditions()
    
    print("\n🏁 Analysis completed!")

if __name__ == "__main__":
    main() 