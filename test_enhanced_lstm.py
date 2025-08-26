import pandas as pd
import numpy as np
import os
import joblib
from tensorflow import keras

def test_enhanced_lstm():
    """Test LSTM model with enhanced XAUUSD.PRO data."""
    
    print("🧪 Testing Enhanced LSTM with XAUUSD.PRO Data...")
    print("=" * 60)
    
    try:
        # Load enhanced data
        print("📊 Loading enhanced XAUUSD.PRO data...")
        df = pd.read_csv('data/XAUUSD_PRO_M5_enhanced.csv')
        print(f"✅ Loaded {len(df)} rows")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Load LSTM model
        print("\n🧠 Loading LSTM model...")
        lstm_model = keras.models.load_model('models/lstm_balanced_model.h5')
        print("✅ LSTM model loaded")
        
        # Load scaler
        print("\n📏 Loading scaler...")
        scaler = joblib.load('models/lstm_balanced_scaler.save')
        print(f"✅ Scaler loaded (expects {scaler.n_features_in_} features)")
        
        # Prepare data
        print("\n🔧 Preparing data...")
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd', 'atr']
        data = df[features].values
        print(f"📊 Data shape: {data.shape}")
        
        # Scale data
        scaled_data = scaler.transform(data)
        print("✅ Data scaled successfully")
        
        # Create sequence
        LSTM_TIMESTEPS = 50
        sequence = scaled_data[-LSTM_TIMESTEPS:].reshape(1, LSTM_TIMESTEPS, -1)
        print(f"✅ Sequence created: {sequence.shape}")
        
        # Make prediction
        print("\n🎯 Making LSTM prediction...")
        prediction = lstm_model.predict(sequence, verbose=0)
        print(f"✅ Prediction successful")
        print(f"📊 Raw prediction: {prediction[0]}")
        
        # Convert to signal
        signal_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        signal_map = {0: -1, 1: 0, 2: 1}  # SELL, HOLD, BUY
        signal = signal_map[signal_class]
        
        signal_names = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        
        print(f"\n🎯 LSTM Results:")
        print(f"📊 Signal: {signal_names[signal]} ({signal})")
        print(f"🎯 Confidence: {confidence:.3f}")
        print(f"💰 Current Price: ${df['close'].iloc[-1]:.2f}")
        
        # Check if this should trigger a trade
        if confidence >= 0.5:
            print("✅ Signal meets confidence threshold!")
            if signal != 0:
                print("🚀 POTENTIAL TRADE SIGNAL!")
            else:
                print("⏸️ HOLD signal - no trade")
        else:
            print("❌ Signal below confidence threshold")
            
        print(f"\n🎉 Test completed successfully!")
        print(f"💡 LSTM model is working with enhanced XAUUSD.PRO data")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_lstm() 