import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

def debug_lstm_model():
    print("🔍 Debugging LSTM Model - Comprehensive Analysis")
    print("=" * 60)
    
    try:
        # Load model and scaler
        print("1. Loading model and scaler...")
        model = load_model('models/lstm_balanced_model.h5')
        scaler = joblib.load('models/lstm_balanced_scaler.joblib')
        print("✅ Model and scaler loaded successfully")
        
        # Load dataset
        print("\n2. Loading dataset...")
        df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')
        print(f"✅ Dataset loaded: {len(df)} samples")
        
        # Check dataset distribution
        buy_count = len(df[df['signal'] == 'BUY'])
        sell_count = len(df[df['signal'] == 'SELL'])
        hold_count = len(df[df['signal'] == 'HOLD'])
        total_count = len(df)
        
        print(f"\n📊 Dataset Distribution:")
        print(f"BUY: {buy_count} ({buy_count/total_count*100:.1f}%)")
        print(f"SELL: {sell_count} ({sell_count/total_count*100:.1f}%)")
        print(f"HOLD: {hold_count} ({hold_count/total_count*100:.1f}%)")
        
        # Prepare features
        print("\n3. Preparing features...")
        feature_columns = ['open', 'high', 'low', 'close', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
        X = df[feature_columns].values
        
        # Check for NaN values
        nan_count = np.isnan(X).sum()
        print(f"NaN values in features: {nan_count}")
        
        # Normalize features
        X_scaled = scaler.transform(X)
        
        # Reshape for LSTM
        timesteps = 10
        X_reshaped = []
        y_reshaped = []
        
        for i in range(timesteps, len(X_scaled)):
            X_reshaped.append(X_scaled[i-timesteps:i])
            y_reshaped.append(df['signal'].iloc[i])
        
        X_reshaped = np.array(X_reshaped)
        y_reshaped = np.array(y_reshaped)
        
        print(f"✅ Reshaped data: {X_reshaped.shape}")
        
        # Test predictions on different samples
        print("\n4. Testing predictions on different samples...")
        
        # Test on first 10 samples
        test_samples = X_reshaped[:10]
        predictions = model.predict(test_samples)
        
        print(f"\n📊 Raw Predictions (first 10 samples):")
        for i, pred in enumerate(predictions):
            print(f"Sample {i+1}: {pred}")
        
        # Convert to classes
        predicted_classes = np.argmax(predictions, axis=1)
        print(f"\n📊 Predicted Classes: {predicted_classes}")
        
        # Test signal mapping
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        predicted_signals = [signal_map[cls] for cls in predicted_classes]
        print(f"📊 Predicted Signals: {predicted_signals}")
        
        # Test on larger sample
        print(f"\n5. Testing on larger sample (100 samples)...")
        large_test = X_reshaped[:100]
        large_predictions = model.predict(large_test)
        large_classes = np.argmax(large_predictions, axis=1)
        
        # Analyze distribution
        unique, counts = np.unique(large_classes, return_counts=True)
        total = len(large_classes)
        
        print(f"\n📊 Large Sample Distribution:")
        for class_id, count in zip(unique, counts):
            percentage = (count / total) * 100
            signal_name = signal_map[class_id]
            print(f"{signal_name}: {percentage:.1f}% ({count})")
        
        # Check model architecture
        print(f"\n6. Model Architecture:")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Number of layers: {len(model.layers)}")
        
        # Check model weights
        print(f"\n7. Model Weights Analysis:")
        for i, layer in enumerate(model.layers):
            if layer.weights:
                print(f"Layer {i} ({layer.name}): {len(layer.weights)} weight arrays")
        
        # Test with different input
        print(f"\n8. Testing with random input...")
        random_input = np.random.random((1, timesteps, len(feature_columns)))
        random_pred = model.predict(random_input)
        random_class = np.argmax(random_pred, axis=1)
        print(f"Random input prediction: {random_pred}")
        print(f"Random input class: {signal_map[random_class[0]]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_signal_mapping():
    """Fix signal mapping if needed"""
    print("\n🔧 Fixing Signal Mapping...")
    
    # Create corrected mapping function
    mapping_code = '''
def correct_signal_mapping(predictions):
    """
    Correct signal mapping function
    predictions: numpy array of shape (n_samples, 3) with probabilities
    returns: list of signal strings
    """
    signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    predicted_classes = np.argmax(predictions, axis=1)
    return [signal_map[cls] for cls in predicted_classes]

def get_signal_confidence(predictions):
    """
    Get confidence for each prediction
    """
    return np.max(predictions, axis=1)
'''
    
    with open('corrected_signal_mapping.py', 'w') as f:
        f.write(mapping_code)
    
    print("✅ Corrected signal mapping function created")

if __name__ == "__main__":
    success = debug_lstm_model()
    
    if success:
        fix_signal_mapping()
        print("\n🎯 Debugging completed. Check the results above.")
    else:
        print("\n❌ Debugging failed. Check the error messages above.") 