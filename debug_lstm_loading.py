#!/usr/bin/env python3
"""
LSTM Model Loading Debug Script
Comprehensive debugging of LSTM model loading and data compatibility
"""

import logging
import os
import sys

import joblib
import numpy as np
from tensorflow import keras

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)


class LSTMDebugger:
    """Comprehensive LSTM model debugger"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.test_results = {}

    def check_model_files(self):
        """Check if model files exist and are accessible"""
        logger.info("🔍 Checking model files...")

        model_files = [
            'models/mrben_lstm_real_data.h5',
            'models/mrben_lstm_real_data_scaler.save',
            'models/mrben_lstm_balanced_v2.h5',
            'models/mrben_lstm_balanced_new.h5',
            'models/mrben_lstm_model.h5',
        ]

        existing_files = []
        for file_path in model_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                existing_files.append((file_path, size))
                logger.info(f"✅ {file_path} ({size:.1f} KB)")
            else:
                logger.warning(f"❌ {file_path} - NOT FOUND")

        return existing_files

    def test_model_loading(self, model_path):
        """Test loading a specific model"""
        logger.info(f"🧪 Testing model loading: {model_path}")

        try:
            # Load model
            model = keras.models.load_model(model_path)
            logger.info("✅ Model loaded successfully")

            # Get model info
            model.summary()

            # Check model architecture
            input_shape = model.input_shape
            output_shape = model.output_shape
            logger.info("📊 Model Architecture:")
            logger.info(f"   Input shape: {input_shape}")
            logger.info(f"   Output shape: {output_shape}")

            return model, True

        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return None, False

    def test_scaler_loading(self, scaler_path):
        """Test loading scaler"""
        logger.info(f"🧪 Testing scaler loading: {scaler_path}")

        try:
            scaler = joblib.load(scaler_path)
            logger.info("✅ Scaler loaded successfully")

            # Get scaler info
            logger.info("📊 Scaler Info:")
            logger.info(f"   Type: {type(scaler)}")
            if hasattr(scaler, 'n_features_in_'):
                logger.info(f"   Features: {scaler.n_features_in_}")
            if hasattr(scaler, 'scale_'):
                logger.info(f"   Scale: {scaler.scale_.shape}")

            return scaler, True

        except Exception as e:
            logger.error(f"❌ Scaler loading failed: {e}")
            return None, False

    def test_data_compatibility(self, model, scaler):
        """Test data compatibility with model"""
        logger.info("🧪 Testing data compatibility...")

        try:
            # Create test data
            test_data = np.random.rand(50, 7)  # 50 timesteps, 7 features
            logger.info(f"📊 Test data shape: {test_data.shape}")

            # Test scaler
            if scaler is not None:
                try:
                    scaled_data = scaler.transform(test_data)
                    logger.info(f"✅ Scaler transform successful: {scaled_data.shape}")
                except Exception as e:
                    logger.error(f"❌ Scaler transform failed: {e}")
                    return False

            # Test model prediction
            if model is not None:
                try:
                    # Reshape for LSTM input
                    sequence = test_data.reshape(1, 50, 7)
                    logger.info(f"📊 Input sequence shape: {sequence.shape}")

                    # Make prediction
                    prediction = model.predict(sequence, verbose=0)
                    logger.info(f"✅ Model prediction successful: {prediction.shape}")
                    logger.info(f"📊 Prediction values: {prediction[0]}")

                    # Check prediction validity
                    if np.any(np.isnan(prediction)):
                        logger.error("❌ Prediction contains NaN values")
                        return False

                    if np.any(np.isinf(prediction)):
                        logger.error("❌ Prediction contains infinite values")
                        return False

                    # Get signal class and confidence
                    signal_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    logger.info(f"📊 Signal class: {signal_class}, Confidence: {confidence:.3f}")

                    return True

                except Exception as e:
                    logger.error(f"❌ Model prediction failed: {e}")
                    return False

            return False

        except Exception as e:
            logger.error(f"❌ Data compatibility test failed: {e}")
            return False

    def test_real_market_data(self, model, scaler):
        """Test with real market data"""
        logger.info("🧪 Testing with real market data...")

        try:
            # Load real market data
            sequences_path = 'data/real_market_sequences.npy'
            if not os.path.exists(sequences_path):
                logger.error(f"❌ Real market data not found: {sequences_path}")
                return False

            sequences = np.load(sequences_path)
            logger.info(f"📊 Real market sequences shape: {sequences.shape}")

            # Test with first sequence
            test_sequence = sequences[0:1]  # Shape: (1, 50, 7)
            logger.info(f"📊 Test sequence shape: {test_sequence.shape}")

            # Test scaler
            if scaler is not None:
                try:
                    # Reshape for scaler (expects 2D)
                    flat_sequence = test_sequence.reshape(-1, 7)
                    scaled_sequence = scaler.transform(flat_sequence)
                    scaled_sequence = scaled_sequence.reshape(1, 50, 7)
                    logger.info("✅ Real data scaling successful")
                except Exception as e:
                    logger.error(f"❌ Real data scaling failed: {e}")
                    return False
            else:
                scaled_sequence = test_sequence

            # Test model prediction
            if model is not None:
                try:
                    prediction = model.predict(scaled_sequence, verbose=0)
                    logger.info(f"✅ Real data prediction successful: {prediction.shape}")
                    logger.info(f"📊 Prediction values: {prediction[0]}")

                    # Get signal class and confidence
                    signal_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                    signal_name = signal_map[signal_class]

                    logger.info(f"📊 Signal: {signal_name}, Confidence: {confidence:.3f}")

                    return True

                except Exception as e:
                    logger.error(f"❌ Real data prediction failed: {e}")
                    return False

            return False

        except Exception as e:
            logger.error(f"❌ Real market data test failed: {e}")
            return False

    def run_comprehensive_debug(self):
        """Run comprehensive LSTM debugging"""
        logger.info("🚀 STARTING COMPREHENSIVE LSTM DEBUG")
        logger.info("=" * 60)

        # Check model files
        existing_files = self.check_model_files()

        if not existing_files:
            logger.error("❌ No model files found!")
            return False

        # Try loading the best model first
        best_model_path = 'models/mrben_lstm_real_data.h5'
        best_scaler_path = 'models/mrben_lstm_real_data_scaler.save'

        if not os.path.exists(best_model_path):
            logger.warning(f"⚠️ Best model not found: {best_model_path}")
            # Try alternative models
            for file_path, size in existing_files:
                if file_path.endswith('.h5'):
                    best_model_path = file_path
                    logger.info(f"🔄 Using alternative model: {best_model_path}")
                    break

        # Load model
        model, model_ok = self.test_model_loading(best_model_path)

        # Load scaler
        scaler, scaler_ok = self.test_scaler_loading(best_scaler_path)

        if not model_ok:
            logger.error("❌ Model loading failed - cannot proceed")
            return False

        # Test data compatibility
        compatibility_ok = self.test_data_compatibility(model, scaler)

        # Test with real market data
        real_data_ok = self.test_real_market_data(model, scaler)

        # Summary
        logger.info("=" * 60)
        logger.info("📋 DEBUG SUMMARY:")
        logger.info(f"   Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
        logger.info(f"   Scaler Loading: {'✅ PASS' if scaler_ok else '❌ FAIL'}")
        logger.info(f"   Data Compatibility: {'✅ PASS' if compatibility_ok else '❌ FAIL'}")
        logger.info(f"   Real Data Test: {'✅ PASS' if real_data_ok else '❌ FAIL'}")

        if model_ok and compatibility_ok and real_data_ok:
            logger.info("🎉 LSTM MODEL IS WORKING CORRECTLY!")
            return True
        else:
            logger.error("❌ LSTM MODEL HAS ISSUES - NEEDS FIXING")
            return False


def main():
    """Main function"""
    debugger = LSTMDebugger()
    success = debugger.run_comprehensive_debug()

    if success:
        logger.info("✅ LSTM debugging completed successfully")
        return True
    else:
        logger.error("❌ LSTM debugging failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
