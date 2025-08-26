#!/usr/bin/env python3
"""
Signal Pipeline Debug Script
Detailed debugging of LSTM, TA, and ML filter signal combination
"""

import logging
import sys
from collections import Counter

import numpy as np
import pandas as pd

# Import our trading system components
sys.path.append('.')
from live_trader_clean import MT5Config, MT5DataManager, MT5SignalGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)


class SignalPipelineDebugger:
    """Detailed signal pipeline debugger"""

    def __init__(self):
        self.config = MT5Config()
        self.data_manager = MT5DataManager()
        self.signal_generator = MT5SignalGenerator(self.config)
        self.debug_results = []

    def debug_lstm_input_data(self):
        """Debug LSTM input data format and preprocessing"""
        logger.info("🔍 Debugging LSTM Input Data...")

        try:
            # Get market data
            df = self.data_manager.get_latest_data(bars=100)
            if df is None or df.empty:
                logger.error("❌ No market data available")
                return False

            logger.info(f"📊 Market data shape: {df.shape}")
            logger.info(f"📊 Columns: {list(df.columns)}")

            # Check required features for LSTM
            required_features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd']
            available_features = [f for f in required_features if f in df.columns]

            logger.info(f"📊 Required features: {required_features}")
            logger.info(f"📊 Available features: {available_features}")

            if len(available_features) < 7:
                logger.error(
                    f"❌ Missing features: {set(required_features) - set(available_features)}"
                )
                return False

            # Prepare data for LSTM
            data = df[available_features].values
            logger.info(f"📊 Raw data shape: {data.shape}")
            logger.info("📊 Data sample (last 5 rows):")
            logger.info(data[-5:])

            # Check for NaN values
            nan_count = np.isnan(data).sum()
            logger.info(f"📊 NaN values: {nan_count}")

            if nan_count > 0:
                logger.warning("⚠️ Data contains NaN values - will be filled")
                data = pd.DataFrame(data, columns=available_features).fillna(method='ffill').values

            # Check data range
            logger.info("📊 Data range:")
            for i, feature in enumerate(available_features):
                min_val = np.min(data[:, i])
                max_val = np.max(data[:, i])
                mean_val = np.mean(data[:, i])
                logger.info(
                    f"   {feature}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}"
                )

            return True

        except Exception as e:
            logger.error(f"❌ LSTM input data debug failed: {e}")
            return False

    def debug_lstm_prediction_step_by_step(self):
        """Debug LSTM prediction step by step"""
        logger.info("🧠 Debugging LSTM Prediction Step by Step...")

        try:
            # Get market data
            df = self.data_manager.get_latest_data(bars=100)
            if df is None or df.empty:
                logger.error("❌ No market data available")
                return False

            # Prepare features
            features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd']
            data = df[features].values

            # Step 1: Check if LSTM model and scaler are loaded
            if self.signal_generator.lstm_model is None:
                logger.error("❌ LSTM model is None")
                return False

            if self.signal_generator.lstm_scaler is None:
                logger.error("❌ LSTM scaler is None")
                return False

            logger.info("✅ LSTM model and scaler are loaded")

            # Step 2: Scale the data
            try:
                scaled_data = self.signal_generator.lstm_scaler.transform(data)
                logger.info(f"✅ Data scaled successfully: {scaled_data.shape}")
                logger.info("📊 Scaled data sample (last 5 rows):")
                logger.info(scaled_data[-5:])
            except Exception as e:
                logger.error(f"❌ Data scaling failed: {e}")
                return False

            # Step 3: Prepare sequence
            timesteps = self.config.LSTM_TIMESTEPS
            if len(scaled_data) < timesteps:
                logger.error(f"❌ Insufficient data: {len(scaled_data)} < {timesteps}")
                return False

            sequence = scaled_data[-timesteps:].reshape(1, timesteps, -1)
            logger.info(f"✅ Sequence prepared: {sequence.shape}")

            # Step 4: Make prediction
            try:
                prediction = self.signal_generator.lstm_model.predict(sequence, verbose=0)
                logger.info(f"✅ LSTM prediction successful: {prediction.shape}")
                logger.info(f"📊 Raw prediction: {prediction[0]}")

                # Step 5: Analyze prediction
                signal_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                signal_name = signal_map[signal_class]

                logger.info("📊 LSTM Analysis:")
                logger.info(f"   Signal class: {signal_class} ({signal_name})")
                logger.info(f"   Confidence: {confidence:.4f}")
                logger.info(
                    f"   All probabilities: SELL={prediction[0][0]:.4f}, HOLD={prediction[0][1]:.4f}, BUY={prediction[0][2]:.4f}"
                )

                # Check if prediction is reasonable
                if confidence < 0.1:
                    logger.warning("⚠️ Very low confidence - model may be uncertain")

                if np.all(prediction[0] < 0.5):
                    logger.warning("⚠️ All probabilities are low - model may need retraining")

                return True

            except Exception as e:
                logger.error(f"❌ LSTM prediction failed: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ LSTM prediction debug failed: {e}")
            return False

    def debug_technical_analysis(self):
        """Debug technical analysis signal generation"""
        logger.info("📈 Debugging Technical Analysis...")

        try:
            # Get market data
            df = self.data_manager.get_latest_data(bars=100)
            if df is None or df.empty:
                logger.error("❌ No market data available")
                return False

            # Generate TA signal
            ta_signal = self.signal_generator._generate_technical_signal(df)

            logger.info("📊 TA Signal Result:")
            logger.info(f"   Signal: {ta_signal['signal']}")
            logger.info(f"   Confidence: {ta_signal['confidence']:.4f}")
            logger.info(f"   RSI: {ta_signal.get('rsi', 'N/A')}")
            logger.info(f"   MACD: {ta_signal.get('macd', 'N/A')}")

            return True

        except Exception as e:
            logger.error(f"❌ Technical analysis debug failed: {e}")
            return False

    def debug_ml_filter(self):
        """Debug ML filter signal combination"""
        logger.info("🤖 Debugging ML Filter...")

        try:
            # Get market data
            df = self.data_manager.get_latest_data(bars=100)
            if df is None or df.empty:
                logger.error("❌ No market data available")
                return False

            # Generate individual signals
            lstm_signal = self.signal_generator._generate_lstm_signal(df)
            ta_signal = self.signal_generator._generate_technical_signal(df)

            logger.info("📊 Individual Signals:")
            logger.info(
                f"   LSTM: signal={lstm_signal['signal']}, confidence={lstm_signal['confidence']:.4f}"
            )
            logger.info(
                f"   TA: signal={ta_signal['signal']}, confidence={ta_signal['confidence']:.4f}"
            )

            # Apply ML filter
            final_signal = self.signal_generator._apply_ml_filter(lstm_signal, ta_signal)

            logger.info("📊 ML Filter Result:")
            logger.info(f"   Final signal: {final_signal['signal']}")
            logger.info(f"   Final confidence: {final_signal['confidence']:.4f}")

            return True

        except Exception as e:
            logger.error(f"❌ ML filter debug failed: {e}")
            return False

    def debug_full_pipeline(self, num_tests=10):
        """Debug full signal pipeline multiple times"""
        logger.info(f"🔄 Debugging Full Pipeline ({num_tests} times)...")

        results = []

        for i in range(num_tests):
            try:
                # Get market data
                df = self.data_manager.get_latest_data(bars=100)
                if df is None or df.empty:
                    continue

                # Generate full signal
                signal_result = self.signal_generator.generate_enhanced_signal(df)

                if signal_result:
                    result = {
                        'test': i + 1,
                        'lstm_signal': signal_result.get('lstm_signal', 0),
                        'lstm_confidence': signal_result.get('lstm_confidence', 0.0),
                        'ta_signal': signal_result.get('ta_signal', 0),
                        'ta_confidence': signal_result.get('ta_confidence', 0.0),
                        'final_signal': signal_result.get('signal', 0),
                        'final_confidence': signal_result.get('confidence', 0.0),
                    }
                    results.append(result)

                    logger.info(
                        f"   Test {i+1}: LSTM({result['lstm_signal']},{result['lstm_confidence']:.3f}) + "
                        f"TA({result['ta_signal']},{result['ta_confidence']:.3f}) = "
                        f"Final({result['final_signal']},{result['final_confidence']:.3f})"
                    )

            except Exception as e:
                logger.error(f"❌ Test {i+1} failed: {e}")
                continue

        # Analyze results
        if results:
            logger.info("📊 Pipeline Analysis:")

            # Count signals
            lstm_signals = [r['lstm_signal'] for r in results]
            ta_signals = [r['ta_signal'] for r in results]
            final_signals = [r['final_signal'] for r in results]

            lstm_counts = Counter(lstm_signals)
            ta_counts = Counter(ta_signals)
            final_counts = Counter(final_signals)

            logger.info(f"   LSTM signals: {dict(lstm_counts)}")
            logger.info(f"   TA signals: {dict(ta_counts)}")
            logger.info(f"   Final signals: {dict(final_counts)}")

            # Check confidence patterns
            lstm_confidences = [r['lstm_confidence'] for r in results]
            avg_lstm_conf = np.mean(lstm_confidences)
            min_lstm_conf = np.min(lstm_confidences)
            max_lstm_conf = np.max(lstm_confidences)

            logger.info(
                f"   LSTM confidence: avg={avg_lstm_conf:.3f}, min={min_lstm_conf:.3f}, max={max_lstm_conf:.3f}"
            )

            if avg_lstm_conf < 0.1:
                logger.warning("⚠️ LSTM confidence is consistently very low!")

            if all(s == 0 for s in lstm_signals):
                logger.error("❌ LSTM always produces HOLD signals!")

            return True
        else:
            logger.error("❌ No successful pipeline tests")
            return False

    def run_comprehensive_debug(self):
        """Run comprehensive signal pipeline debugging"""
        logger.info("🚀 STARTING SIGNAL PIPELINE DEBUG")
        logger.info("=" * 60)

        # Debug LSTM input data
        input_ok = self.debug_lstm_input_data()

        # Debug LSTM prediction
        lstm_ok = self.debug_lstm_prediction_step_by_step()

        # Debug technical analysis
        ta_ok = self.debug_technical_analysis()

        # Debug ML filter
        ml_ok = self.debug_ml_filter()

        # Debug full pipeline
        pipeline_ok = self.debug_full_pipeline(num_tests=10)

        # Summary
        logger.info("=" * 60)
        logger.info("📋 PIPELINE DEBUG SUMMARY:")
        logger.info(f"   Input Data: {'✅ PASS' if input_ok else '❌ FAIL'}")
        logger.info(f"   LSTM Prediction: {'✅ PASS' if lstm_ok else '❌ FAIL'}")
        logger.info(f"   Technical Analysis: {'✅ PASS' if ta_ok else '❌ FAIL'}")
        logger.info(f"   ML Filter: {'✅ PASS' if ml_ok else '❌ FAIL'}")
        logger.info(f"   Full Pipeline: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")

        if input_ok and lstm_ok and ta_ok and ml_ok and pipeline_ok:
            logger.info("🎉 SIGNAL PIPELINE IS WORKING CORRECTLY!")
            return True
        else:
            logger.error("❌ SIGNAL PIPELINE HAS ISSUES - NEEDS FIXING")
            return False


def main():
    """Main function"""
    debugger = SignalPipelineDebugger()
    success = debugger.run_comprehensive_debug()

    if success:
        logger.info("✅ Signal pipeline debugging completed successfully")
        return True
    else:
        logger.error("❌ Signal pipeline debugging failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
