#!/usr/bin/env python3
"""
Test New LSTM Model
"""

import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

def print_status(message, level="INFO"):
    """Print status with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def main():
    """Test the newly trained model"""
    print_status("🧪 Testing New LSTM Model", "START")
    
    # Check model files
    model_path = "models/mrben_lstm_real_data.h5"
    scaler_path = "models/mrben_lstm_real_data_scaler.save"
    
    if not os.path.exists(model_path):
        print_status(f"❌ Model file not found: {model_path}", "ERROR")
        return False
    
    if not os.path.exists(scaler_path):
        print_status(f"❌ Scaler file not found: {scaler_path}", "ERROR")
        return False
    
    print_status("✅ Model files found", "SUCCESS")
    
    # Load model and scaler
    print_status("📊 Loading model...", "INFO")
    try:
        model = load_model(model_path)
        print_status(f"✅ Model loaded successfully", "SUCCESS")
        print_status(f"   Model parameters: {model.count_params():,}", "INFO")
    except Exception as e:
        print_status(f"❌ Error loading model: {e}", "ERROR")
        return False
    
    print_status("📊 Loading scaler...", "INFO")
    try:
        scaler = joblib.load(scaler_path)
        print_status(f"✅ Scaler loaded successfully", "SUCCESS")
    except Exception as e:
        print_status(f"❌ Error loading scaler: {e}", "ERROR")
        return False
    
    # Test with sample data
    print_status("🧪 Testing model with sample data...", "INFO")
    try:
        # Load some real data for testing
        sequences = np.load("data/real_market_sequences.npy")
        sample_sequence = sequences[0:1]  # Take first sequence
        
        # Reshape for scaler
        n_samples, n_timesteps, n_features = sample_sequence.shape
        sample_reshaped = sample_sequence.reshape(-1, n_features)
        
        # Scale the data
        sample_scaled = scaler.transform(sample_reshaped)
        sample_scaled = sample_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Make prediction
        prediction = model.predict(sample_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        signal_type = ["SELL", "HOLD", "BUY"][predicted_class]
        print_status(f"✅ Prediction successful", "SUCCESS")
        print_status(f"   Signal: {signal_type}", "INFO")
        print_status(f"   Confidence: {confidence:.4f}", "INFO")
        print_status(f"   Raw probabilities: {prediction[0]}", "INFO")
        
    except Exception as e:
        print_status(f"❌ Error during prediction: {e}", "ERROR")
        return False
    
    print_status("🎉 Model test completed successfully!", "SUCCESS")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print_status("🎉 Model is ready for use!", "SUCCESS")
    else:
        print_status("❌ Model test failed!", "ERROR")
    
    print_status("Press Enter to continue...", "INFO")
    input() 