import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

def detailed_analysis():
    print("🔍 Detailed Analysis of Model Outputs")
    print("=" * 60)
    
    try:
        # Load model and scaler
        model = load_model('models/lstm_balanced_model.h5')
        scaler = joblib.load('models/lstm_balanced_scaler.joblib')
        
        # Load dataset
        df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')
        
        # Prepare features
        feature_columns = ['open', 'high', 'low', 'close', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
        X = df[feature_columns].values
        X_scaled = scaler.transform(X)
        
        # Reshape for LSTM
        timesteps = 10
        X_reshaped = []
        
        for i in range(timesteps, len(X_scaled)):
            X_reshaped.append(X_scaled[i-timesteps:i])
        
        X_reshaped = np.array(X_reshaped)
        
        # Test on first 20 samples
        test_samples = X_reshaped[:20]
        predictions = model.predict(test_samples)
        
        print("📊 Detailed Analysis of First 20 Samples:")
        print("-" * 60)
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        
        for i, pred in enumerate(predictions):
            print(f"\nSample {i+1}:")
            print(f"  Raw probabilities: {pred}")
            print(f"  SELL (class 0): {pred[0]:.6f}")
            print(f"  HOLD (class 1): {pred[1]:.6f}")
            print(f"  BUY (class 2): {pred[2]:.6f}")
            
            # Find max manually
            max_idx = np.argmax(pred)
            max_val = pred[max_idx]
            print(f"  Max probability: {max_val:.6f} at index {max_idx}")
            print(f"  Predicted signal: {signal_map[max_idx]}")
            
            # Check if there are ties
            max_indices = np.where(pred == max_val)[0]
            if len(max_indices) > 1:
                print(f"  ⚠️ TIE DETECTED! Multiple max values at indices: {max_indices}")
        
        # Analyze distribution
        predicted_classes = np.argmax(predictions, axis=1)
        unique, counts = np.unique(predicted_classes, return_counts=True)
        
        print(f"\n📊 Overall Distribution:")
        for class_id, count in zip(unique, counts):
            percentage = (count / len(predictions)) * 100
            signal_name = signal_map[class_id]
            print(f"{signal_name}: {percentage:.1f}% ({count})")
        
        # Check if all predictions are the same
        if len(unique) == 1:
            print(f"\n❌ PROBLEM: All predictions are {signal_map[unique[0]]}!")
            
            # Check if this is due to very small differences
            print(f"\n🔍 Checking for numerical precision issues...")
            for i, pred in enumerate(predictions[:5]):
                max_val = np.max(pred)
                min_val = np.min(pred)
                diff = max_val - min_val
                print(f"Sample {i+1}: Max={max_val:.6f}, Min={min_val:.6f}, Diff={diff:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_weights():
    """Check if model weights are properly initialized"""
    print("\n🔍 Checking Model Weights...")
    
    try:
        model = load_model('models/lstm_balanced_model.h5')
        
        print("Model layers:")
        for i, layer in enumerate(model.layers):
            print(f"  Layer {i}: {layer.name} - {type(layer).__name__}")
            if layer.weights:
                for j, weight in enumerate(layer.weights):
                    print(f"    Weight {j}: shape={weight.shape}, mean={np.mean(weight.numpy()):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking weights: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Detailed Analysis...")
    
    # Run detailed analysis
    analysis_success = detailed_analysis()
    
    if analysis_success:
        # Check model weights
        check_model_weights()
        
        print("\n🎯 Analysis completed. Check results above.")
    else:
        print("\n❌ Analysis failed.") 