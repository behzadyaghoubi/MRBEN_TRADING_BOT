#!/usr/bin/env python3
"""
Comprehensive Strategy Signal Test
Test all signal scenarios (BUY/SELL/HOLD) with the updated pipeline
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

def test_signal_scenarios():
    """Test different signal scenarios."""
    print("🎯 Testing Signal Scenarios")
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
        if os.path.exists('models/mrben_ai_signal_filter_xgb_balanced.joblib'):
            from ai_filter import AISignalFilter
            ml_filter = AISignalFilter('models/mrben_ai_signal_filter_xgb_balanced.joblib')
            print("✅ Balanced ML filter loaded")
        elif os.path.exists('models/mrben_ai_signal_filter_xgb.joblib'):
            from ai_filter import AISignalFilter
            ml_filter = AISignalFilter('models/mrben_ai_signal_filter_xgb.joblib')
            print("✅ Original ML filter loaded")
        
        # Create signal generator
        signal_generator = MT5SignalGenerator(config, lstm_model, lstm_scaler, ml_filter)
        
        # Get current market data
        df = data_manager.get_latest_data(500)
        if df is None:
            print("❌ Failed to get market data")
            return
        
        print(f"📊 Data shape: {df.shape}")
        print(f"📊 Latest price: {df['close'].iloc[-1]:.2f}")
        
        # Test scenarios
        scenarios = [
            ("Current Market", df),
            ("Bullish Market", create_bullish_scenario(df)),
            ("Bearish Market", create_bearish_scenario(df)),
            ("Neutral Market", create_neutral_scenario(df)),
            ("Overbought Market", create_overbought_scenario(df)),
            ("Oversold Market", create_oversold_scenario(df))
        ]
        
        results = []
        
        for scenario_name, test_df in scenarios:
            print(f"\n🔍 Testing {scenario_name}")
            print("-" * 40)
            
            # Generate signal
            signal = signal_generator.generate_enhanced_signal(test_df)
            
            # Store results
            results.append({
                'scenario': scenario_name,
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'source': signal.get('source', 'Unknown')
            })
            
            # Display result
            signal_name = {1: "BUY", 0: "HOLD", -1: "SELL"}[signal['signal']]
            print(f"   Final Signal: {signal_name} ({signal['signal']})")
            print(f"   Confidence: {signal['confidence']:.3f}")
            print(f"   Source: {signal.get('source', 'Unknown')}")
        
        # Summary
        print(f"\n📊 Summary of Results")
        print("=" * 60)
        
        buy_count = sum(1 for r in results if r['signal'] == 1)
        sell_count = sum(1 for r in results if r['signal'] == -1)
        hold_count = sum(1 for r in results if r['signal'] == 0)
        
        print(f"   BUY signals: {buy_count}/{len(results)} ({buy_count/len(results)*100:.1f}%)")
        print(f"   SELL signals: {sell_count}/{len(results)} ({sell_count/len(results)*100:.1f}%)")
        print(f"   HOLD signals: {hold_count}/{len(results)} ({hold_count/len(results)*100:.1f}%)")
        
        # Check for bias
        if buy_count > sell_count + 2:
            print("   ⚠️ Potential BUY bias detected")
        elif sell_count > buy_count + 2:
            print("   ⚠️ Potential SELL bias detected")
        else:
            print("   ✅ Balanced signal distribution")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in signal testing: {e}")
        import traceback
        traceback.print_exc()
        return []

def create_bullish_scenario(df):
    """Create bullish market scenario."""
    test_df = df.copy()
    # Modify last row to simulate bullish conditions
    test_df.iloc[-1, test_df.columns.get_loc('rsi')] = 60  # Neutral RSI
    test_df.iloc[-1, test_df.columns.get_loc('macd')] = 0.5  # Positive MACD
    test_df.iloc[-1, test_df.columns.get_loc('close')] = df['close'].iloc[-1] + 5  # Higher price
    return test_df

def create_bearish_scenario(df):
    """Create bearish market scenario."""
    test_df = df.copy()
    # Modify last row to simulate bearish conditions
    test_df.iloc[-1, test_df.columns.get_loc('rsi')] = 40  # Neutral RSI
    test_df.iloc[-1, test_df.columns.get_loc('macd')] = -0.5  # Negative MACD
    test_df.iloc[-1, test_df.columns.get_loc('close')] = df['close'].iloc[-1] - 5  # Lower price
    return test_df

def create_neutral_scenario(df):
    """Create neutral market scenario."""
    test_df = df.copy()
    # Modify last row to simulate neutral conditions
    test_df.iloc[-1, test_df.columns.get_loc('rsi')] = 50  # Neutral RSI
    test_df.iloc[-1, test_df.columns.get_loc('macd')] = 0.0  # Neutral MACD
    return test_df

def create_overbought_scenario(df):
    """Create overbought market scenario."""
    test_df = df.copy()
    # Modify last row to simulate overbought conditions
    test_df.iloc[-1, test_df.columns.get_loc('rsi')] = 85  # Overbought RSI
    test_df.iloc[-1, test_df.columns.get_loc('macd')] = 0.8  # High MACD
    return test_df

def create_oversold_scenario(df):
    """Create oversold market scenario."""
    test_df = df.copy()
    # Modify last row to simulate oversold conditions
    test_df.iloc[-1, test_df.columns.get_loc('rsi')] = 15  # Oversold RSI
    test_df.iloc[-1, test_df.columns.get_loc('macd')] = -0.8  # Low MACD
    return test_df

def test_ml_filter_directly():
    """Test ML filter directly with various inputs."""
    print("\n🧪 Testing ML Filter Directly")
    print("=" * 60)
    
    try:
        # Load ML filter
        if os.path.exists('models/mrben_ai_signal_filter_xgb_balanced.joblib'):
            from ai_filter import AISignalFilter
            ml_filter = AISignalFilter('models/mrben_ai_signal_filter_xgb_balanced.joblib')
            
            # Test cases
            test_cases = [
                # LSTM_signal, LSTM_conf, TA_signal, TA_conf, Expected
                ([1, 0.9, 1, 0.8], "Strong BUY", "BUY"),
                ([-1, 0.9, -1, 0.8], "Strong SELL", "SELL"),
                ([1, 0.7, -1, 0.6], "Mixed BUY/SELL", "Mixed"),
                ([-1, 0.7, 1, 0.6], "Mixed SELL/BUY", "Mixed"),
                ([0, 0.5, 0, 0.5], "Neutral HOLD", "Neutral"),
                ([1, 0.8, 0, 0.7], "LSTM BUY + TA Neutral", "BUY"),
                ([-1, 0.8, 0, 0.7], "LSTM SELL + TA Neutral", "SELL"),
                ([0, 0.6, 1, 0.8], "LSTM Neutral + TA BUY", "BUY"),
                ([0, 0.6, -1, 0.8], "LSTM Neutral + TA SELL", "SELL")
            ]
            
            print("📋 ML Filter Test Results:")
            print("-" * 40)
            
            for features, description, expected in test_cases:
                features_array = np.array(features).reshape(1, -1)
                result = ml_filter.filter_signal_with_confidence(features)
                
                prediction = result['prediction']
                confidence = result['confidence']
                signal = "BUY" if prediction == 1 else "SELL"
                
                print(f"   {description}:")
                print(f"     Expected: {expected}, Got: {signal}")
                print(f"     Confidence: {confidence:.3f}")
                print(f"     Features: LSTM({features[0]},{features[1]:.1f}), TA({features[2]},{features[3]:.1f})")
                print()
        
        else:
            print("❌ Balanced ML filter not found")
            
    except Exception as e:
        print(f"❌ Error testing ML filter: {e}")

def main():
    """Main function."""
    print("🎯 Comprehensive Strategy Signal Test")
    print("=" * 60)
    
    # Test signal scenarios
    results = test_signal_scenarios()
    
    # Test ML filter directly
    test_ml_filter_directly()
    
    print("\n✅ Testing completed!")
    
    if results:
        print("\n📝 Recommendations:")
        buy_count = sum(1 for r in results if r['signal'] == 1)
        sell_count = sum(1 for r in results if r['signal'] == -1)
        
        if buy_count > sell_count:
            print("   - Consider adjusting ML filter threshold for more SELL signals")
        elif sell_count > buy_count:
            print("   - Consider adjusting ML filter threshold for more BUY signals")
        else:
            print("   - Signal distribution looks balanced")

if __name__ == "__main__":
    main() 