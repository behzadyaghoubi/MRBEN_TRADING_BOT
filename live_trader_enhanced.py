#!/usr/bin/env python3
"""
Enhanced Live Trader with Advanced LSTM + ML Filter Pipeline
MR BEN Trading System - Professional Version
"""

import json
import logging
import os
import sys
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =====================
# ADVANCED LSTM SIGNAL GENERATION
# =====================
try:
    import joblib
    from tensorflow.keras.models import load_model

    # Advanced LSTM Model Paths
    ADVANCED_LSTM_MODEL_PATH = 'models/advanced_lstm_model.h5'
    ADVANCED_LSTM_SCALER_PATH = 'models/advanced_lstm_scaler.save'

    # Load Advanced LSTM Model
    if os.path.exists(ADVANCED_LSTM_MODEL_PATH):
        print("🚀 Loading Enhanced Advanced LSTM Model...")
        advanced_lstm_model = load_model(ADVANCED_LSTM_MODEL_PATH)
        advanced_lstm_scaler = joblib.load(ADVANCED_LSTM_SCALER_PATH)
        print("✅ Enhanced Advanced LSTM Model loaded successfully!")
    else:
        print("⚠️ Advanced LSTM model not found")
        advanced_lstm_model = None
        advanced_lstm_scaler = None

except Exception as e:
    print(f"❌ Error loading LSTM models: {e}")
    advanced_lstm_model = None
    advanced_lstm_scaler = None

# =====================
# ML FILTER INTEGRATION
# =====================
try:
    from ai_filter import AISignalFilter

    # ML Filter Configuration
    ML_FILTER_PATH = 'models/mrben_ai_signal_filter_xgb.joblib'
    ML_FILTER_THRESHOLD = 0.65  # Enhanced threshold for better filtering

    if os.path.exists(ML_FILTER_PATH):
        print("🔍 Loading Enhanced ML Filter...")
        ml_filter = AISignalFilter(
            model_path=ML_FILTER_PATH, model_type="joblib", threshold=ML_FILTER_THRESHOLD
        )
        print("✅ Enhanced ML Filter loaded successfully!")
    else:
        print("⚠️ ML Filter not found")
        ml_filter = None

except Exception as e:
    print(f"❌ Error loading ML Filter: {e}")
    ml_filter = None


# =====================
# CONFIGURATION
# =====================
class EnhancedTradingConfig:
    """Enhanced configuration class for trading parameters."""

    def __init__(self):
        # Load config file
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.json')
        if not os.path.exists(config_path):
            config_path = os.path.join('config', 'settings.json')
        try:
            with open(config_path, encoding='utf-8') as f:
                self.config_data = json.load(f)
        except Exception:
            self.config_data = {}

        # MT5 Settings
        self.SYMBOL = self.config_data.get("trading", {}).get("symbol", "ADAUSD")
        self.VOLUME = self.config_data.get("trading", {}).get("min_lot", 1.0)
        self.MAGIC = self.config_data.get("trading", {}).get("magic_number", 654321)
        self.TIMEFRAME = self.config_data.get("trading", {}).get("timeframe", 5)
        self.BARS = self.config_data.get("trading", {}).get("bars", 500)

        # Enhanced Trading Parameters
        self.SLEEP_SECONDS = 30  # Faster response
        self.RETRY_DELAY = 5
        self.WINDOW_SIZE = 50
        self.LSTM_TIMESTEPS = 50

        # Enhanced Risk Management
        self.BASE_RISK = 0.02  # Slightly more aggressive
        self.MIN_LOT = 0.01
        self.MAX_LOT = 2.0
        self.MAX_OPEN_TRADES = 3

        # Enhanced Signal Filtering
        self.MIN_SIGNAL_CONFIDENCE = 0.7
        self.CONSECUTIVE_SIGNALS_REQUIRED = 2
        self.ML_FILTER_ENABLED = True
        self.LSTM_FILTER_ENABLED = True

        # Data Directory
        self.DATA_DIR = "data"


# =====================
# ENHANCED SIGNAL GENERATOR
# =====================
class EnhancedSignalGenerator:
    """Enhanced signal generator with LSTM + ML Filter pipeline."""

    def __init__(self, config: EnhancedTradingConfig):
        self.config = config
        self.lstm_model = advanced_lstm_model
        self.lstm_scaler = advanced_lstm_scaler
        self.ml_filter = ml_filter

    def generate_enhanced_signal(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate enhanced signal using LSTM + ML Filter pipeline
        """
        try:
            # Step 1: LSTM Signal Generation
            lstm_signal, lstm_confidence = self._generate_lstm_signal(df)

            # Step 2: Technical Analysis Confirmation
            ta_signal, ta_confidence = self._generate_technical_signal(df)

            # Step 3: ML Filter Validation
            final_signal, final_confidence = self._apply_ml_filter(
                lstm_signal, lstm_confidence, ta_signal, ta_confidence
            )

            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'lstm_signal': lstm_signal,
                'lstm_confidence': lstm_confidence,
                'ta_signal': ta_signal,
                'ta_confidence': ta_confidence,
                'source': 'Enhanced_LSTM_ML_Pipeline',
            }

        except Exception as e:
            logging.error(f"Error in enhanced signal generation: {e}")
            return {
                'signal': 0,
                'confidence': 0.0,
                'lstm_signal': 0,
                'lstm_confidence': 0.0,
                'ta_signal': 0,
                'ta_confidence': 0.0,
                'source': 'Error_Fallback',
            }

    def _generate_lstm_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """Generate LSTM signal with confidence."""
        if self.lstm_model is None or self.lstm_scaler is None:
            return 0, 0.0

        try:
            # Prepare features for LSTM
            features = ['open', 'high', 'low', 'close', 'tick_volume']
            available_features = [f for f in features if f in df.columns]

            if len(available_features) < 4:
                return 0, 0.0

            # Prepare sequence data
            data = df[available_features].values
            scaled_data = self.lstm_scaler.transform(data)

            # Create sequence
            if len(scaled_data) >= self.config.LSTM_TIMESTEPS:
                sequence = scaled_data[-self.config.LSTM_TIMESTEPS :].reshape(
                    1, self.config.LSTM_TIMESTEPS, -1
                )

                # Predict
                prediction = self.lstm_model.predict(sequence, verbose=0)

                # Convert to signal
                signal_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])

                # Map to signal: 0=HOLD, 1=BUY, 2=SELL
                signal = signal_class - 1  # Convert to -1, 0, 1

                return signal, confidence
            else:
                return 0, 0.0

        except Exception as e:
            logging.error(f"Error in LSTM signal generation: {e}")
            return 0, 0.0

    def _generate_technical_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """Generate technical analysis signal."""
        try:
            # Calculate technical indicators
            rsi = self._calculate_rsi(df['close'])
            macd, macd_signal = self._calculate_macd(df['close'])

            # Simple technical signal
            signal = 0
            confidence = 0.5

            if rsi < 30 and macd > macd_signal:
                signal = 1  # BUY
                confidence = 0.7
            elif rsi > 70 and macd < macd_signal:
                signal = -1  # SELL
                confidence = 0.7

            return signal, confidence

        except Exception as e:
            logging.error(f"Error in technical signal generation: {e}")
            return 0, 0.0

    def _apply_ml_filter(
        self, lstm_signal: int, lstm_conf: float, ta_signal: int, ta_conf: float
    ) -> tuple[int, float]:
        """Apply ML filter to combine signals."""
        if self.ml_filter is None:
            # Simple combination without ML filter
            if lstm_signal == ta_signal and lstm_signal != 0:
                return lstm_signal, (lstm_conf + ta_conf) / 2
            else:
                return 0, 0.0

        try:
            # Prepare features for ML filter
            features = {
                'RSI': self._calculate_rsi(df['close'])[-1] if len(df) > 0 else 50,
                'MACD': self._calculate_macd(df['close'])[0][-1] if len(df) > 0 else 0,
                'ATR': self._calculate_atr(df)[-1] if len(df) > 0 else 0,
                'Volume': df['tick_volume'].iloc[-1] if 'tick_volume' in df.columns else 0,
            }

            # Get ML filter prediction
            ml_prediction = self.ml_filter.predict(features)
            ml_confidence = self.ml_filter.get_confidence(features)

            # Combine signals
            if lstm_signal == ta_signal and lstm_signal != 0:
                # Both signals agree
                if ml_prediction > self.config.MIN_SIGNAL_CONFIDENCE:
                    return lstm_signal, (lstm_conf + ta_conf + ml_confidence) / 3
                else:
                    return 0, 0.0
            else:
                # Signals disagree - use ML filter as tiebreaker
                if ml_confidence > self.config.MIN_SIGNAL_CONFIDENCE:
                    return lstm_signal if lstm_conf > ta_conf else ta_signal, ml_confidence
                else:
                    return 0, 0.0

        except Exception as e:
            logging.error(f"Error in ML filter application: {e}")
            return 0, 0.0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd.values, macd_signal.values

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate ATR indicator."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.values


# =====================
# ENHANCED LIVE TRADER
# =====================
class EnhancedLiveTrader:
    """Enhanced live trader with advanced signal processing."""

    def __init__(self, config: EnhancedTradingConfig):
        self.config = config
        self.signal_generator = EnhancedSignalGenerator(config)
        self.running = False
        self.stop_event = threading.Event()

        # Setup logging
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup enhanced logging."""
        logger = logging.getLogger('EnhancedLiveTrader')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # File handler
            fh = logging.FileHandler('logs/enhanced_live_trader.log')
            fh.setLevel(logging.INFO)

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Formatter
            formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    def start(self):
        """Start enhanced live trading."""
        try:
            self.logger.info("🚀 Starting Enhanced Live Trading System...")

            # Check model availability
            if self.signal_generator.lstm_model is None:
                self.logger.warning("⚠️ LSTM model not available")

            if self.signal_generator.ml_filter is None:
                self.logger.warning("⚠️ ML Filter not available")

            # Start trading loop
            self.running = True
            self.stop_event.clear()

            trading_thread = threading.Thread(target=self._trading_loop)
            trading_thread.daemon = True
            trading_thread.start()

            self.logger.info("✅ Enhanced Live Trading System started successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error starting enhanced live trading: {e}")
            return False

    def stop(self):
        """Stop enhanced live trading."""
        self.logger.info("🛑 Stopping Enhanced Live Trading System...")
        self.running = False
        self.stop_event.set()
        self.logger.info("✅ Enhanced Live Trading System stopped")

    def _trading_loop(self):
        """Enhanced trading loop with advanced signal processing."""
        last_signal = 0
        consecutive_signals = 0

        while self.running and not self.stop_event.is_set():
            try:
                # Simulate market data (replace with actual MT5 data)
                df = self._get_simulation_data()

                if df is None or len(df) < self.config.WINDOW_SIZE:
                    self.logger.warning("⚠️ Insufficient market data")
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # Generate enhanced signal
                signal_data = self.signal_generator.generate_enhanced_signal(df)

                # Log signal information
                self.logger.info(
                    f"📊 Enhanced Signal: {signal_data['signal']} | "
                    f"Confidence: {signal_data['confidence']:.3f} | "
                    f"LSTM: {signal_data['lstm_signal']}({signal_data['lstm_confidence']:.3f}) | "
                    f"TA: {signal_data['ta_signal']}({signal_data['ta_confidence']:.3f}) | "
                    f"Source: {signal_data['source']}"
                )

                # Signal validation
                if signal_data['signal'] == last_signal and signal_data['signal'] != 0:
                    consecutive_signals += 1
                else:
                    consecutive_signals = 0
                    last_signal = signal_data['signal']

                # Execute trade if conditions are met
                if (
                    signal_data['signal'] != 0
                    and signal_data['confidence'] >= self.config.MIN_SIGNAL_CONFIDENCE
                    and consecutive_signals >= self.config.CONSECUTIVE_SIGNALS_REQUIRED
                ):

                    self.logger.info(f"🎯 Executing trade: Signal={signal_data['signal']}")
                    # Add actual trade execution here

                time.sleep(self.config.SLEEP_SECONDS)

            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(self.config.RETRY_DELAY)

    def _get_simulation_data(self) -> pd.DataFrame | None:
        """Get simulation data (replace with actual MT5 data)."""
        try:
            # Try to load real data
            data_files = [
                'data/ohlc_data.csv',
                'data/adausd_data.csv',
                'data/lstm_signals_features.csv',
            ]

            for file_path in data_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if len(df) >= self.config.WINDOW_SIZE:
                        return df.tail(self.config.WINDOW_SIZE)

            # Generate synthetic data if no real data available
            return self._generate_synthetic_data()

        except Exception as e:
            self.logger.error(f"Error getting simulation data: {e}")
            return None

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic trading data for testing."""
        np.random.seed(int(time.time()))

        n_samples = self.config.WINDOW_SIZE
        base_price = 2000.0

        prices = [base_price]
        for i in range(1, n_samples):
            change = np.random.normal(0, 0.01)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        df = pd.DataFrame(
            {
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'close': prices,
                'tick_volume': np.random.randint(100, 1000, n_samples),
            }
        )

        return df


# =====================
# MAIN FUNCTION
# =====================
def main():
    """Main function to run enhanced live trader."""
    print("🚀 MR BEN Enhanced Live Trading System")
    print("=" * 50)

    # Initialize configuration
    config = EnhancedTradingConfig()

    # Initialize enhanced live trader
    trader = EnhancedLiveTrader(config)

    try:
        # Start trading
        if trader.start():
            print("✅ Enhanced Live Trading System started successfully!")
            print("📊 Monitoring signals... Press Ctrl+C to stop")

            # Keep running
            while trader.running:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping Enhanced Live Trading System...")
        trader.stop()
        print("✅ Enhanced Live Trading System stopped successfully!")

    except Exception as e:
        print(f"❌ Error in main: {e}")
        trader.stop()


if __name__ == "__main__":
    main()
