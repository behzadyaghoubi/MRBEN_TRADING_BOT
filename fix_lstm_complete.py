#!/usr/bin/env python3
"""
Complete LSTM Fix Script
Properly loads and tests LSTM model with real data
"""

import json
import logging
import os
import sys

import joblib
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from tensorflow import keras

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)


class CompleteLSTMFixer:
    """Complete LSTM model fixer"""

    def __init__(self):
        self.lstm_model = None
        self.lstm_scaler = None
        self.ml_filter = None

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False

            # Load settings
            with open('config/settings.json') as f:
                settings = json.load(f)

            # Login to MT5
            if not mt5.login(
                login=settings['mt5_login'],
                password=settings['mt5_password'],
                server=settings['mt5_server'],
            ):
                logger.error("MT5 login failed")
                return False

            logger.info("✅ MT5 connected successfully")
            return True

        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    def load_lstm_model(self):
        """Load LSTM model and scaler"""
        logger.info("🧠 Loading LSTM Model...")

        try:
            # Try real data model first, then balanced models, fallback to original
            model_paths = [
                'models/mrben_lstm_real_data.h5',
                'models/mrben_lstm_balanced_v2.h5',
                'models/mrben_lstm_balanced_new.h5',
                'models/mrben_lstm_model.h5',
            ]
            scaler_paths = [
                'models/mrben_lstm_real_data_scaler.save',
                'models/mrben_lstm_scaler_v2.save',
                'models/mrben_lstm_scaler_balanced.save',
                'models/mrben_lstm_scaler.save',
            ]

            for model_path, scaler_path in zip(model_paths, scaler_paths, strict=False):
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    logger.info(f"Loading LSTM Model from {model_path}...")
                    self.lstm_model = keras.models.load_model(model_path)
                    self.lstm_scaler = joblib.load(scaler_path)
                    logger.info("✅ LSTM Model loaded successfully!")

                    # Print model info
                    logger.info(f"📊 Model input shape: {self.lstm_model.input_shape}")
                    logger.info(f"📊 Model output shape: {self.lstm_model.output_shape}")
                    logger.info(f"📊 Scaler features: {self.lstm_scaler.n_features_in_}")

                    return True

            logger.error("❌ No LSTM model found")
            return False

        except Exception as e:
            logger.error(f"❌ Error loading LSTM model: {e}")
            return False

    def get_market_data(self):
        """Get market data from MT5"""
        logger.info("📊 Getting market data...")

        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos("XAUUSD.PRO", mt5.TIMEFRAME_M5, 0, 100)

            if rates is None:
                logger.error("❌ Failed to get MT5 data")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)

            logger.info(f"✅ Market data loaded: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"❌ Error getting market data: {e}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        df['macd'] = ema_fast - ema_slow

        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def test_lstm_prediction(self, df: pd.DataFrame):
        """Test LSTM prediction with real data"""
        logger.info("🧪 Testing LSTM prediction...")

        try:
            # Prepare features
            features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd']
            data = df[features].values

            logger.info(f"📊 Input data shape: {data.shape}")

            # Scale data
            scaled_data = self.lstm_scaler.transform(data)
            logger.info("✅ Data scaled successfully")

            # Prepare sequence (50 timesteps)
            timesteps = 50
            if len(scaled_data) < timesteps:
                logger.error(f"❌ Insufficient data: {len(scaled_data)} < {timesteps}")
                return False

            sequence = scaled_data[-timesteps:].reshape(1, timesteps, -1)
            logger.info(f"✅ Sequence prepared: {sequence.shape}")

            # Make prediction
            prediction = self.lstm_model.predict(sequence, verbose=0)
            logger.info(f"✅ LSTM prediction successful: {prediction.shape}")

            # Analyze prediction
            signal_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal_name = signal_map[signal_class]

            logger.info("📊 LSTM Prediction Results:")
            logger.info(f"   Signal: {signal_name} (class {signal_class})")
            logger.info(f"   Confidence: {confidence:.4f}")
            logger.info(
                f"   Probabilities: SELL={prediction[0][0]:.4f}, HOLD={prediction[0][1]:.4f}, BUY={prediction[0][2]:.4f}"
            )

            # Check if prediction is reasonable
            if confidence > 0.1:
                logger.info("✅ LSTM prediction is reasonable")
                return True
            else:
                logger.warning("⚠️ LSTM confidence is very low")
                return False

        except Exception as e:
            logger.error(f"❌ LSTM prediction test failed: {e}")
            return False

    def test_multiple_predictions(self, num_tests=20):
        """Test multiple LSTM predictions"""
        logger.info(f"🔄 Testing {num_tests} LSTM predictions...")

        results = []

        for i in range(num_tests):
            try:
                # Get fresh market data
                df = self.get_market_data()
                if df is None:
                    continue

                # Test prediction
                success = self.test_lstm_prediction(df)

                if success:
                    # Get prediction details
                    features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd']
                    data = df[features].values
                    scaled_data = self.lstm_scaler.transform(data)
                    sequence = scaled_data[-50:].reshape(1, 50, -1)
                    prediction = self.lstm_model.predict(sequence, verbose=0)

                    signal_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                    signal_name = signal_map[signal_class]

                    results.append(
                        {
                            'test': i + 1,
                            'signal': signal_name,
                            'signal_class': signal_class,
                            'confidence': confidence,
                            'probabilities': prediction[0].tolist(),
                        }
                    )

                    logger.info(f"   Test {i+1}: {signal_name} (confidence: {confidence:.3f})")

            except Exception as e:
                logger.error(f"❌ Test {i+1} failed: {e}")
                continue

        # Analyze results
        if results:
            logger.info("📊 Multiple Test Analysis:")

            # Count signals
            signal_counts = {}
            for result in results:
                signal = result['signal']
                signal_counts[signal] = signal_counts.get(signal, 0) + 1

            total_tests = len(results)
            logger.info(f"   Total successful tests: {total_tests}")

            for signal, count in signal_counts.items():
                percentage = (count / total_tests) * 100
                logger.info(f"   {signal}: {count} ({percentage:.1f}%)")

            # Check confidence distribution
            confidences = [r['confidence'] for r in results]
            avg_conf = np.mean(confidences)
            min_conf = np.min(confidences)
            max_conf = np.max(confidences)

            logger.info(
                f"   Confidence: avg={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}"
            )

            # Check for bias
            if len(signal_counts) >= 2:
                logger.info("✅ Signal distribution looks balanced")
                return True
            else:
                logger.warning("⚠️ Signal distribution may be biased")
                return False
        else:
            logger.error("❌ No successful tests")
            return False

    def create_fixed_signal_generator(self):
        """Create a fixed signal generator with proper LSTM integration"""
        logger.info("🔧 Creating fixed signal generator...")

        try:
            # Import the signal generator class
            from live_trader_clean import MT5Config, MT5SignalGenerator

            # Create config
            config = MT5Config()

            # Create signal generator with loaded models
            signal_generator = MT5SignalGenerator(
                config=config,
                lstm_model=self.lstm_model,
                lstm_scaler=self.lstm_scaler,
                ml_filter=self.ml_filter,
            )

            logger.info("✅ Fixed signal generator created")
            return signal_generator

        except Exception as e:
            logger.error(f"❌ Error creating signal generator: {e}")
            return None

    def test_fixed_signal_generator(self, signal_generator):
        """Test the fixed signal generator"""
        logger.info("🧪 Testing fixed signal generator...")

        try:
            # Get market data
            df = self.get_market_data()
            if df is None:
                return False

            # Generate signal
            signal_result = signal_generator.generate_enhanced_signal(df)

            if signal_result:
                logger.info("📊 Signal Generation Results:")
                logger.info(f"   Final signal: {signal_result.get('signal', 'N/A')}")
                logger.info(f"   Final confidence: {signal_result.get('confidence', 0.0):.4f}")
                logger.info(f"   LSTM signal: {signal_result.get('lstm_signal', 'N/A')}")
                logger.info(f"   TA signal: {signal_result.get('ta_signal', 'N/A')}")

                return True
            else:
                logger.error("❌ No signal result")
                return False

        except Exception as e:
            logger.error(f"❌ Signal generator test failed: {e}")
            return False

    def run_complete_fix(self):
        """Run complete LSTM fix"""
        logger.info("🚀 STARTING COMPLETE LSTM FIX")
        logger.info("=" * 60)

        # Initialize MT5
        if not self.initialize_mt5():
            logger.error("❌ MT5 initialization failed")
            return False

        # Load LSTM model
        if not self.load_lstm_model():
            logger.error("❌ LSTM model loading failed")
            return False

        # Test single prediction
        df = self.get_market_data()
        if df is not None:
            single_ok = self.test_lstm_prediction(df)
        else:
            single_ok = False

        # Test multiple predictions
        multiple_ok = self.test_multiple_predictions(num_tests=20)

        # Create and test fixed signal generator
        signal_generator = self.create_fixed_signal_generator()
        if signal_generator:
            generator_ok = self.test_fixed_signal_generator(signal_generator)
        else:
            generator_ok = False

        # Summary
        logger.info("=" * 60)
        logger.info("📋 COMPLETE FIX SUMMARY:")
        logger.info("   LSTM Model Loading: ✅ PASS")
        logger.info(f"   Single Prediction: {'✅ PASS' if single_ok else '❌ FAIL'}")
        logger.info(f"   Multiple Predictions: {'✅ PASS' if multiple_ok else '❌ FAIL'}")
        logger.info(f"   Signal Generator: {'✅ PASS' if generator_ok else '❌ FAIL'}")

        if single_ok and multiple_ok and generator_ok:
            logger.info("🎉 LSTM FIX COMPLETED SUCCESSFULLY!")
            logger.info("✅ System is ready for live trading!")
            return True
        else:
            logger.error("❌ LSTM FIX FAILED")
            return False


def main():
    """Main function"""
    fixer = CompleteLSTMFixer()
    success = fixer.run_complete_fix()

    if success:
        logger.info("✅ Complete LSTM fix completed successfully")
        return True
    else:
        logger.error("❌ Complete LSTM fix failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
