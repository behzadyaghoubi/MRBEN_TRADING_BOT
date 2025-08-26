#!/usr/bin/env python3
"""
Check Models Status
Diagnose why no real signals are being generated
"""

import os
import sys
import pandas as pd
import numpy as np

def check_models_status():
    """Check the status of all models and data files."""
    
    print("🔍 Checking Models and Data Status...")
    print("=" * 50)
    
    # Check LSTM Model
    print("\n1️⃣ LSTM Model Status:")
    lstm_model_path = 'models/lstm_balanced_model.h5'
    lstm_scaler_path = 'models/lstm_balanced_scaler.save'
    
    print(f"   LSTM Model exists: {os.path.exists(lstm_model_path)}")
    print(f"   LSTM Scaler exists: {os.path.exists(lstm_scaler_path)}")
    
    if os.path.exists(lstm_model_path):
        try:
            from tensorflow.keras.models import load_model
            import joblib
            
            lstm_model = load_model(lstm_model_path)
            lstm_scaler = joblib.load(lstm_scaler_path)
            
            print(f"   ✅ LSTM Model loaded successfully")
            print(f"   📐 Model input shape: {lstm_model.input_shape}")
            print(f"   📐 Model output shape: {lstm_model.output_shape}")
            print(f"   🔧 Scaler features: {lstm_scaler.n_features_in_}")
            
        except Exception as e:
            print(f"   ❌ Error loading LSTM model: {e}")
    else:
        print("   ❌ LSTM model not found")
    
    # Check ML Filter
    print("\n2️⃣ ML Filter Status:")
    ml_filter_path = 'models/mrben_ai_signal_filter_xgb.joblib'
    print(f"   ML Filter exists: {os.path.exists(ml_filter_path)}")
    
    if os.path.exists(ml_filter_path):
        try:
            from ai_filter import AISignalFilter
            ml_filter = AISignalFilter(
                model_path=ml_filter_path,
                model_type="joblib",
                threshold=0.65
            )
            print("   ✅ ML Filter loaded successfully")
        except Exception as e:
            print(f"   ❌ Error loading ML Filter: {e}")
    else:
        print("   ❌ ML Filter not found")
    
    # Check Data Files
    print("\n3️⃣ Data Files Status:")
    data_files = [
        'data/XAUUSD_PRO_M5_enhanced.csv',
        'data/XAUUSD_PRO_M5_data.csv',
        'data/XAUUSD_PRO_M15_history.csv',
        'data/ohlc_data.csv',
        'data/lstm_signals_features.csv',
        'data/ml_training_data.csv'
    ]
    
    for file_path in data_files:
        exists = os.path.exists(file_path)
        print(f"   {file_path}: {'✅' if exists else '❌'}")
        
        if exists:
            try:
                df = pd.read_csv(file_path)
                print(f"      📊 Rows: {len(df)}, Columns: {list(df.columns)}")
                
                # Check if it has required features
                required_features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd', 'atr']
                available_features = [f for f in required_features if f in df.columns]
                print(f"      🎯 Required features: {len(available_features)}/8")
                
                if len(available_features) < 8:
                    missing = [f for f in required_features if f not in df.columns]
                    print(f"      ⚠️ Missing features: {missing}")
                
            except Exception as e:
                print(f"      ❌ Error reading file: {e}")
    
    # Check Log Files
    print("\n4️⃣ Log Files Status:")
    log_files = [
        'logs/gold_live_trader.log',
        'logs/gold_trades.csv'
    ]
    
    for file_path in log_files:
        exists = os.path.exists(file_path)
        print(f"   {file_path}: {'✅' if exists else '❌'}")
        
        if exists:
            try:
                if file_path.endswith('.log'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        print(f"      📝 Log lines: {len(lines)}")
                        if lines:
                            print(f"      📅 Last log: {lines[-1].strip()}")
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    print(f"      📊 Trade records: {len(df)}")
                    if len(df) > 0:
                        print(f"      📅 Last trade: {df.iloc[-1].to_dict()}")
            except Exception as e:
                print(f"      ❌ Error reading log: {e}")
    
    # Test Signal Generation
    print("\n5️⃣ Signal Generation Test:")
    try:
        # Try to load enhanced data
        test_data_path = 'data/XAUUSD_PRO_M5_enhanced.csv'
        if os.path.exists(test_data_path):
            df = pd.read_csv(test_data_path)
            print(f"   ✅ Test data loaded: {len(df)} rows")
            
            # Check if we have enough data
            if len(df) >= 50:
                print("   ✅ Sufficient data for LSTM (>= 50 rows)")
                
                # Check features
                required_features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd', 'atr']
                available_features = [f for f in required_features if f in df.columns]
                
                if len(available_features) == 8:
                    print("   ✅ All required features present")
                    
                    # Test LSTM prediction
                    if os.path.exists(lstm_model_path) and os.path.exists(lstm_scaler_path):
                        try:
                            from tensorflow.keras.models import load_model
                            import joblib
                            
                            lstm_model = load_model(lstm_model_path)
                            lstm_scaler = joblib.load(lstm_scaler_path)
                            
                            # Prepare data
                            data = df[available_features].values
                            scaled_data = lstm_scaler.transform(data)
                            
                            # Test prediction
                            sequence = scaled_data[-50:].reshape(1, 50, -1)
                            prediction = lstm_model.predict(sequence, verbose=0)
                            
                            signal_class = np.argmax(prediction[0])
                            confidence = np.max(prediction[0])
                            
                            signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
                            signal_name = signal_map[signal_class]
                            
                            print(f"   🎯 LSTM Test Prediction: {signal_name} (confidence: {confidence:.4f})")
                            print(f"   📊 Raw probabilities: SELL={prediction[0][0]:.4f}, HOLD={prediction[0][1]:.4f}, BUY={prediction[0][2]:.4f}")
                            
                        except Exception as e:
                            print(f"   ❌ Error in LSTM test: {e}")
                    else:
                        print("   ❌ LSTM model not available for test")
                else:
                    print(f"   ❌ Missing features: need 8, have {len(available_features)}")
            else:
                print(f"   ❌ Insufficient data: need >= 50, have {len(df)}")
        else:
            print("   ❌ Enhanced data file not found")
            
    except Exception as e:
        print(f"   ❌ Error in signal generation test: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 Status Check Complete!")

if __name__ == "__main__":
    check_models_status() 