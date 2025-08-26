"""
Live Trading System for MetaTrader 5
Professional auto-trading system with LSTM and ML signal filtering
"""

import csv
import logging
import os
import sys
import threading
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')  # Suppress general warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Additional suppress for TF
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress more TF warnings
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MetaTrader5 import را فقط در صورت نیاز انجام بده
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 not available - running in simulation mode only")

# TensorFlow imports
try:
    from tensorflow.keras.models import load_model

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - LSTM predictions disabled")

# Joblib import
try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Joblib not available - ML filter disabled")


# =====================
# CONFIGURATION
# =====================
class TradingConfig:
    """Configuration class for trading parameters."""

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
        self.SYMBOL = self.config_data.get('trading', {}).get('symbol', 'XAUUSD.PRO')
        if MT5_AVAILABLE:
            self.TIMEFRAME = mt5.TIMEFRAME_M5
        else:
            self.TIMEFRAME = 5  # 5 minutes as integer
        self.BARS = 500

        # Model Settings
        self.LSTM_MODEL_PATH = "models/lstm_trading_model.h5"  # Change to fixed model if re-saved
        self.ML_FILTER_MODEL_PATH = "models/mrben_ai_signal_filter_xgb.joblib"
        self.SCALER_PATH = "models/lstm_scaler.save"
        self.WINDOW_SIZE = 60  # Matched to model's trained shape
        self.NUM_FEATURES = 23  # Number of features for LSTM input shape

        # Trading Settings (dynamic)
        self.VOLUME = self.config_data.get('trading', {}).get('min_lot', 0.1)
        self.MAGIC = self.config_data.get('trading', {}).get('magic_number', 20250627)
        self.DEVIATION = self.config_data.get('trading', {}).get('deviation', 20)
        self.STOP_LOSS_PIPS = self.config_data.get('trading', {}).get('stop_loss_pips', 15)
        self.TAKE_PROFIT_PIPS = self.config_data.get('trading', {}).get('take_profit_pips', 30)

        # MT5 Credentials
        self.MT5_LOGIN = self.config_data.get('mt5', {}).get('login', 0)
        self.MT5_PASSWORD = self.config_data.get('mt5', {}).get('password', '')
        self.MT5_SERVER = self.config_data.get('mt5', {}).get('server', '')
        self.MT5_TIMEOUT = self.config_data.get('mt5', {}).get('timeout', 60000)

        # System Settings
        self.SLEEP_SECONDS = 60
        self.RETRY_DELAY = 30
        self.LOG_LEVEL = logging.INFO

        # Data paths
        self.DATA_DIR = "data"
        self.LOGS_DIR = "logs"

        # Simulation Mode
        self.SIMULATION_MODE = self.config_data.get('simulation_mode', False)


# =====================
# LOGGER SETUP
# =====================
def setup_logger(config):
    """Setup professional logging configuration."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.LOGS_DIR, 'live_trader.log')),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("LiveTrader")


def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr


def log_trade(time, signal, price, sl, tp, result, buy_proba, sell_proba, features):
    """Log trade details to CSV file."""
    log_file = os.path.join(config.DATA_DIR, "trade_log.csv")
    headers = ['time', 'signal', 'price', 'sl', 'tp', 'result', 'buy_proba', 'sell_proba'] + [
        f'feature_{i}' for i in range(len(features))
    ]

    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)  # Add header if file is new
        writer.writerow(
            [time, signal, price, sl, tp, result, buy_proba, sell_proba] + list(features)
        )


def update_trade_result(ticket, new_result):
    """Update trade result in log (e.g., after close)."""
    log_file = os.path.join(config.DATA_DIR, "trade_log.csv")
    if not os.path.exists(log_file):
        return
    df = pd.read_csv(log_file)
    # Assuming ticket is logged somewhere; for simplicity, skip for now
    # You can extend this to match by time or add ticket column
    df.to_csv(log_file, index=False)


def dynamic_sl_tp_from_history(trade_log_path, current_price, signal, default_atr):
    """Calculate dynamic SL/TP based on historical performance."""
    if not os.path.exists(trade_log_path):
        return (
            current_price - default_atr * 1.5 if signal == 1 else current_price + default_atr * 1.5
        ), (current_price + default_atr * 3 if signal == 1 else current_price - default_atr * 3)

    try:
        df = pd.read_csv(trade_log_path)
        if len(df) < 20:
            return (
                current_price - default_atr * 1.5
                if signal == 1
                else current_price + default_atr * 1.5
            ), (current_price + default_atr * 3 if signal == 1 else current_price - default_atr * 3)

        recent = df.tail(50)
        success_rate = (
            len(recent[recent['result'] == 'WIN']) / len(recent) if len(recent) > 0 else 0.5
        )

        if success_rate > 0.6:
            sl_multiplier = 1.2
            tp_multiplier = 2.5
        elif success_rate < 0.4:
            sl_multiplier = 1.8
            tp_multiplier = 3.5
        else:
            sl_multiplier = 1.5
            tp_multiplier = 3.0

        sl = (
            current_price - default_atr * sl_multiplier
            if signal == 1
            else current_price + default_atr * sl_multiplier
        )
        tp = (
            current_price + default_atr * tp_multiplier
            if signal == 1
            else current_price - default_atr * tp_multiplier
        )

        return sl, tp
    except Exception as e:
        print(f"Error calculating dynamic SL/TP: {e}")
        return (
            current_price - default_atr * 1.5 if signal == 1 else current_price + default_atr * 1.5
        ), (current_price + default_atr * 3 if signal == 1 else current_price - default_atr * 3)


def price_action_confirmation(df):
    """Simple price action confirmation."""
    if len(df) < 3:
        return True

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    # Check for bullish engulfing
    if (
        last_candle['close'] > last_candle['open']
        and prev_candle['close'] < prev_candle['open']
        and last_candle['close'] > prev_candle['open']
        and last_candle['open'] < prev_candle['close']
    ):
        return True

    # Check for bearish engulfing
    if (
        last_candle['close'] < last_candle['open']
        and prev_candle['close'] > prev_candle['open']
        and last_candle['close'] < prev_candle['open']
        and last_candle['open'] > prev_candle['close']
    ):
        return True

    return True  # Default to True for now


class TrailingStopManager(threading.Thread):
    """Manages trailing stops for open positions."""

    def __init__(
        self, symbol, ticket, direction, trailing_distance, logger, stop_event, mt5_manager
    ):
        super().__init__()
        self.symbol = symbol
        self.ticket = ticket
        self.direction = direction  # 1 for buy, -1 for sell
        self.trailing_distance = trailing_distance
        self.logger = logger
        self.stop_event = stop_event
        self.mt5_manager = mt5_manager  # Added for accessing MT5

    def run(self):
        """Run trailing stop logic."""
        while not self.stop_event.is_set():
            try:
                time.sleep(10)  # Check every 10 seconds
                if self.mt5_manager.config.SIMULATION_MODE:
                    self.logger.info(f"Simulated trailing stop for ticket {self.ticket}")
                    continue

                # Get current position
                position = mt5.positions_get(ticket=self.ticket)
                if not position:
                    break  # Position closed

                position = position[0]
                current_price = (
                    mt5.symbol_info_tick(self.symbol).bid
                    if self.direction == 1
                    else mt5.symbol_info_tick(self.symbol).ask
                )

                # Calculate new SL
                if self.direction == 1:  # Buy
                    new_sl = current_price - self.trailing_distance
                    if new_sl > position.sl:
                        self._modify_sl(new_sl)
                else:  # Sell
                    new_sl = current_price + self.trailing_distance
                    if new_sl < position.sl:
                        self._modify_sl(new_sl)
            except Exception as e:
                self.logger.error(f"Trailing stop error: {e}")

    def _modify_sl(self, new_sl):
        """Modify SL in MT5."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": self.ticket,
            "sl": new_sl,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"SL modified for ticket {self.ticket} to {new_sl}")
        else:
            self.logger.error(f"Failed to modify SL: {result.comment}")


class MT5Manager:
    """Manages MetaTrader 5 connection and operations."""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(config)
        self.connected = False

    def initialize(self):
        """Initialize MT5 connection."""
        if self.config.SIMULATION_MODE:
            self.logger.info("Running in simulation mode - MT5 initialization skipped")
            return True

        if not MT5_AVAILABLE:
            self.logger.info("MT5 not available - running in simulation mode")
            return True

        try:
            if not mt5.initialize(
                login=self.config.MT5_LOGIN,
                password=self.config.MT5_PASSWORD,
                server=self.config.MT5_SERVER,
            ):
                self.logger.error("MT5 initialization failed")
                return False

            self.connected = True
            self.logger.info("MT5 connected successfully")
            return True

        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")
            return False

    def shutdown(self):
        """Shutdown MT5 connection."""
        if MT5_AVAILABLE and self.connected:
            try:
                mt5.shutdown()
                self.logger.info("MT5 disconnected")
            except Exception as e:
                self.logger.error(f"MT5 shutdown error: {e}")

    def get_latest_data(self):
        """Get latest market data."""
        if not self.connected and not self.config.SIMULATION_MODE:
            return None

        try:
            if self.config.SIMULATION_MODE:
                return self._generate_simulation_data()

            rates = mt5.copy_rates_from_pos(
                self.config.SYMBOL, self.config.TIMEFRAME, 0, self.config.BARS
            )
            if rates is None:
                self.logger.error("Failed to get MT5 data")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None

    def _generate_simulation_data(self):
        """Generate simulation data for testing."""
        try:
            data_file = os.path.join(self.config.DATA_DIR, "XAUUSD_PRO_M15_history.csv")
            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                return df.tail(self.config.BARS).reset_index(drop=True)

            # Generate synthetic data with more realism (added trend)
            np.random.seed(42)
            dates = pd.date_range(start='2024-01-01', periods=self.config.BARS, freq='5T')
            base_price = 2000.0
            prices = [base_price]
            trend = 0.1  # Small upward trend
            for i in range(1, self.config.BARS):
                change = np.random.normal(0, 0.5) + trend
                new_price = prices[-1] + change
                prices.append(new_price)

            df = pd.DataFrame(
                {
                    'time': dates,
                    'open': [p - np.random.uniform(0, 1) for p in prices],
                    'high': [p + np.random.uniform(0, 2) for p in prices],
                    'low': [p - np.random.uniform(0, 2) for p in prices],
                    'close': prices,
                    'tick_volume': np.random.randint(100, 1000, self.config.BARS),
                }
            )

            return df

        except Exception as e:
            self.logger.error(f"Error generating simulation data: {e}")
            return None

    def has_open_position(self):
        """Check if there's an open position with our magic number."""
        if self.config.SIMULATION_MODE:
            return False  # Simulate no open positions
        positions = mt5.positions_get(symbol=self.config.SYMBOL)
        for pos in positions:
            if pos.magic == self.config.MAGIC:
                return True
        return False

    def send_order(self, signal, price, sl=None, tp=None):
        """Send trading order to MT5."""
        if self.has_open_position():
            self.logger.info("Open position exists - skipping new order")
            return None

        if not self.connected and not self.config.SIMULATION_MODE:
            self.logger.warning("Cannot send order - not connected to MT5")
            return None

        try:
            if self.config.SIMULATION_MODE:
                order_info = {
                    'ticket': int(time.time()),
                    'symbol': self.config.SYMBOL,
                    'type': signal,
                    'price': price,
                    'sl': sl,
                    'tp': tp,
                    'volume': self.config.VOLUME,
                    'magic': self.config.MAGIC,
                    'comment': 'LSTM_AI_Signal',
                }
                self.logger.info(f"Simulated order: {order_info}")
                return order_info

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.SYMBOL,
                "volume": self.config.VOLUME,
                "type": mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": self.config.DEVIATION,
                "magic": self.config.MAGIC,
                "comment": "LSTM_AI_Signal",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment}")
                return None

            self.logger.info(f"Order sent successfully: {result.order}")
            return result

        except Exception as e:
            self.logger.error(f"Error sending order: {e}")
            return None


class LSTMModel:
    """LSTM model for signal prediction."""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(config)
        self.model = None
        self.scaler = None
        self.num_classes = 3  # Default to 3 classes if detection fails

    def load_model(self):
        """Load LSTM model and scaler."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available - LSTM disabled")
            return False

        try:
            if os.path.exists(self.config.LSTM_MODEL_PATH):
                self.model = load_model(self.config.LSTM_MODEL_PATH)
                # Build the model if not already built
                input_shape = (None, self.config.WINDOW_SIZE, self.config.NUM_FEATURES)
                self.model.build(input_shape)
                # Call with dummy input to define output
                dummy_input = np.zeros((1, self.config.WINDOW_SIZE, self.config.NUM_FEATURES))
                _ = self.model(dummy_input, training=False)  # Call to define output
                self.num_classes = self.model.output.shape[-1]  # Check number of classes
                self.logger.info(
                    f"LSTM model loaded, built, and called successfully (classes: {self.num_classes})"
                )
            else:
                self.logger.error(f"LSTM model not found: {self.config.LSTM_MODEL_PATH}")
                return False

            if os.path.exists(self.config.SCALER_PATH):
                self.scaler = joblib.load(self.config.SCALER_PATH)
                self.logger.info("LSTM scaler loaded successfully")
            else:
                self.logger.warning(f"LSTM scaler not found: {self.config.SCALER_PATH}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {e}")
            self.logger.warning("Falling back to default signal generation")
            return False

    def predict(self, df):
        """Generate LSTM predictions."""
        if self.model is None:
            self.logger.warning("LSTM model not loaded - returning default signal")
            return 0, 0.5, 0.5  # Default: hold, 50% prob

        try:
            features = self._prepare_features(df)
            if features is None:
                return None, None, None

            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features

            features_reshaped = features_scaled.reshape(
                1, features_scaled.shape[0], features_scaled.shape[1]
            )

            prediction = self.model.predict(features_reshaped, verbose=0)

            if self.num_classes == 3:  # Use detected classes
                buy_proba = prediction[0][0]
                hold_proba = prediction[0][1]
                sell_proba = prediction[0][2]
                if buy_proba > sell_proba and buy_proba > 0.4:
                    signal = 1
                elif sell_proba > buy_proba and sell_proba > 0.4:
                    signal = -1
                else:
                    signal = 0
            else:
                buy_proba = prediction[0][0]
                sell_proba = 1 - buy_proba
                if buy_proba > 0.6:
                    signal = 1
                elif buy_proba < 0.4:
                    signal = -1
                else:
                    signal = 0

            return signal, buy_proba, sell_proba

        except Exception as e:
            self.logger.error(f"LSTM prediction error: {e}")
            return 0, 0.5, 0.5  # Fallback to default

    def _prepare_features(self, df):
        """Prepare features for LSTM model."""
        try:
            if len(df) < self.config.WINDOW_SIZE:
                return None

            df_window = df.tail(self.config.WINDOW_SIZE).copy()

            df_window['sma_20'] = df_window['close'].rolling(window=20).mean()
            df_window['ema_20'] = df_window['close'].ewm(span=20).mean()
            df_window['rsi'] = self._calculate_rsi(df_window['close'])
            df_window['macd'], df_window['macd_signal'] = self._calculate_macd(df_window['close'])
            df_window['bb_upper'], df_window['bb_lower'] = self._calculate_bollinger_bands(
                df_window['close']
            )
            df_window['atr'] = calculate_atr(df_window)

            df_window['sma_50'] = df_window['close'].rolling(window=50).mean()
            df_window['ema_50'] = df_window['close'].ewm(span=50).mean()
            df_window['stoch_k'], df_window['stoch_d'] = self._calculate_stochastic(df_window)
            df_window['williams_r'] = self._calculate_williams_r(df_window)
            df_window['cci'] = self._calculate_cci(df_window)

            df_window['price_change'] = df_window['close'].pct_change()
            df_window['high_low_ratio'] = df_window['high'] / df_window['low']
            df_window['volume_ma'] = df_window['tick_volume'].rolling(window=20).mean()
            df_window['volume_ratio'] = df_window['tick_volume'] / df_window['volume_ma']

            df_window['body_size'] = abs(df_window['close'] - df_window['open'])
            df_window['upper_shadow'] = df_window['high'] - np.maximum(
                df_window['open'], df_window['close']
            )
            df_window['lower_shadow'] = (
                np.minimum(df_window['open'], df_window['close']) - df_window['low']
            )

            df_window = df_window.dropna()

            if len(df_window) == 0:
                return None

            feature_columns = [
                'open',
                'high',
                'low',
                'close',
                'tick_volume',
                'sma_20',
                'ema_20',
                'rsi',
                'macd',
                'macd_signal',
                'bb_upper',
                'bb_lower',
                'atr',
                'price_change',
                'high_low_ratio',
                'volume_ma',
                'sma_50',
                'ema_50',
                'stoch_k',
                'stoch_d',
                'williams_r',
                'cci',
                'volume_ratio',
            ]

            # Select only existing columns and ensure 23
            available_columns = [col for col in feature_columns if col in df_window.columns]
            if len(available_columns) != self.config.NUM_FEATURES:
                self.logger.warning(
                    f"Feature mismatch: Expected {self.config.NUM_FEATURES}, got {len(available_columns)}. Adjust model or features."
                )
                return None  # Instead of padding, fail to avoid misleading model

            features = df_window[available_columns].values
            return features

        except Exception as e:
            self.logger.error(f"Feature preparation error: {e}")
            return None

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    def _calculate_williams_r(self, df, period=14):
        """Calculate Williams %R."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()

        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
        return williams_r

    def _calculate_cci(self, df, period=20):
        """Calculate Commodity Channel Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))

        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci


class MLFilter:
    """ML filter for signal validation."""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(config)
        self.model = None

    def load_model(self):
        """Load ML filter model."""
        if not JOBLIB_AVAILABLE:
            self.logger.warning("Joblib not available - ML filter disabled")
            return False

        try:
            if os.path.exists(self.config.ML_FILTER_MODEL_PATH):
                self.model = joblib.load(self.config.ML_FILTER_MODEL_PATH)
                self.logger.info("ML filter model loaded successfully")
                return True
            else:
                self.logger.warning(
                    f"ML filter model not found: {self.config.ML_FILTER_MODEL_PATH}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error loading ML filter: {e}")
            return False

    def filter_signal(self, buy_proba, sell_proba, signal):
        """Filter signal using ML model."""
        if self.model is None:
            return signal  # Return original signal if no filter

        try:
            # Prepare features - assuming model expects 4 features (add a dummy if needed)
            features = np.array(
                [buy_proba, sell_proba, signal, 0.0]
            )  # Dummy 4th feature (change if you know the real one)
            features = features.reshape(1, -1)

            # Predict
            filtered_signal = self.model.predict(features)[0]

            return filtered_signal

        except ValueError as ve:
            if "shape mismatch" in str(ve):
                self.logger.warning(
                    f"ML filter shape mismatch: {ve} - falling back to original signal"
                )
                return signal
            else:
                raise
        except Exception as e:
            self.logger.error(f"ML filter error: {e}")
            return signal  # Return original signal on error


# =====================
# LSTM SIGNAL GENERATION (ADDED: MR BEN NEW LSTM PIPELINE)
# =====================
# Load LSTM model and scaler at the top
try:
    LSTM_MODEL_PATH = 'models/mrben_lstm_model.h5'
    LSTM_SCALER_PATH = 'models/mrben_lstm_scaler.save'
    if TENSORFLOW_AVAILABLE and os.path.exists(LSTM_MODEL_PATH):
        lstm_model = load_model(LSTM_MODEL_PATH)
    else:
        lstm_model = None
    if JOBLIB_AVAILABLE and os.path.exists(LSTM_SCALER_PATH):
        lstm_scaler = joblib.load(LSTM_SCALER_PATH)
    else:
        lstm_scaler = None
except Exception:
    lstm_model = None
    lstm_scaler = None


class LiveTrader:
    """Main live trading class."""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(config)
        self.mt5_manager = MT5Manager(config)
        self.lstm_model = LSTMModel(config)
        self.ml_filter = MLFilter(config)
        self.running = False
        self.stop_event = threading.Event()
        self.trailing_threads = []  # To manage trailing stops

    def start(self):
        """Start live trading."""
        try:
            self.logger.info("Starting Live Trading System...")

            # Initialize MT5
            if not self.mt5_manager.initialize():
                self.logger.error("Failed to initialize MT5")
                return False

            # Load models
            if not self.lstm_model.load_model():
                self.logger.warning("LSTM model loading failed - continuing without LSTM")

            if not self.ml_filter.load_model():
                self.logger.warning("ML filter loading failed - continuing without filter")

            # Start trading loop
            self.running = True
            self.stop_event.clear()

            trading_thread = threading.Thread(target=self._trading_loop)
            trading_thread.daemon = True
            trading_thread.start()

            self.logger.info("Live Trading System started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error starting live trading: {e}")
            return False

    def stop(self):
        """Stop live trading."""
        self.logger.info("Stopping Live Trading System...")
        self.running = False
        self.stop_event.set()
        for thread in self.trailing_threads:
            thread.join()
        self.mt5_manager.shutdown()
        self.logger.info("Live Trading System stopped")

    def _trading_loop(self):
        """Main trading loop."""
        last_signal = 0
        consecutive_signals = 0
        lookback = 10
        while self.running and not self.stop_event.is_set():
            try:
                # Get market data
                df = self.mt5_manager.get_latest_data()
                if df is None or len(df) < lookback:
                    self.logger.warning("Insufficient market data")
                    time.sleep(self.config.RETRY_DELAY)
                    continue

                # --- LSTM Signal Section (NEW) ---
                signal_source = 'LSTM'
                signal, buy_proba, sell_proba = 0, 0.5, 0.5
                try:
                    if lstm_model is not None and lstm_scaler is not None:
                        # Build feature sequence for last lookback bars
                        feature_cols = ['close', 'sl', 'tp', 'buy_proba', 'sell_proba']
                        # Add any extra features if available
                        for col in ['ATR', 'feature_0', 'feature_1', 'feature_2']:
                            if col in df.columns and col not in feature_cols:
                                feature_cols.append(col)
                        df_seq = df.tail(lookback)[feature_cols].copy()
                        if df_seq.isnull().any().any() or len(df_seq) < lookback:
                            raise ValueError('Not enough data for LSTM sequence')
                        X_seq = lstm_scaler.transform(df_seq.values).reshape(
                            1, lookback, len(feature_cols)
                        )
                        pred = lstm_model.predict(X_seq, verbose=0)
                        pred_class = int(np.argmax(pred, axis=1)[0])
                        # Map class to signal: 0=hold, 1=buy, 2=sell
                        if pred_class == 1:
                            signal = 1
                        elif pred_class == 2:
                            signal = -1
                        else:
                            signal = 0
                        buy_proba = float(pred[0][1]) if pred.shape[1] > 1 else 0.5
                        sell_proba = float(pred[0][2]) if pred.shape[1] > 2 else 0.5
                    else:
                        raise Exception('LSTM model or scaler not loaded')
                except Exception as e:
                    self.logger.warning(f"LSTM prediction failed: {e}. Using fallback.")
                    signal_source = 'FALLBACK'
                    # Fallback: use ML filter or price action
                    # (You can call your ML filter or price action logic here)
                    # Example fallback:
                    signal, buy_proba, sell_proba = (
                        self.ml_filter.filter_signal(0.5, 0.5, 0),
                        0.5,
                        0.5,
                    )

                self.logger.info(f"Signal generated by: {signal_source} | Signal: {signal}")

                # Apply ML filter (if not already fallback)
                if signal_source == 'LSTM':
                    filtered_signal = self.ml_filter.filter_signal(buy_proba, sell_proba, signal)
                else:
                    filtered_signal = signal

                # Signal validation
                if filtered_signal == last_signal:
                    consecutive_signals += 1
                else:
                    consecutive_signals = 0
                    last_signal = filtered_signal

                # Execute trade if conditions are met
                if (
                    filtered_signal != 0
                    and consecutive_signals >= 2  # Require 2 consecutive signals
                    and price_action_confirmation(df)
                ):
                    current_price = df['close'].iloc[-1]
                    atr = calculate_atr(df).iloc[-1]
                    # Calculate SL/TP
                    sl, tp = dynamic_sl_tp_from_history(
                        os.path.join(self.config.DATA_DIR, "trade_log.csv"),
                        current_price,
                        filtered_signal,
                        atr,
                    )
                    # Send order
                    order_result = self.mt5_manager.send_order(
                        filtered_signal, current_price, sl, tp
                    )
                    if order_result:
                        # Log trade
                        features = [buy_proba, sell_proba, atr, current_price]
                        log_trade(
                            datetime.now(),
                            filtered_signal,
                            current_price,
                            sl,
                            tp,
                            "PENDING",
                            buy_proba,
                            sell_proba,
                            features,
                        )
                        self.logger.info(
                            f"Trade executed: Signal={filtered_signal}, "
                            f"Price={current_price:.2f}, SL={sl:.2f}, TP={tp:.2f}"
                        )
                        # Start trailing stop if order successful
                        if not self.config.SIMULATION_MODE:
                            trailing = TrailingStopManager(
                                self.config.SYMBOL,
                                order_result.order,
                                filtered_signal,
                                atr * 1.5,
                                self.logger,
                                self.stop_event,
                                self.mt5_manager,
                            )
                            trailing.start()
                            self.trailing_threads.append(trailing)
                # Sleep before next iteration
                time.sleep(self.config.SLEEP_SECONDS)
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                time.sleep(self.config.RETRY_DELAY)


def main():
    """Main function."""
    print("🤖 MR BEN Live Trading System")
    print("=" * 50)

    # Load configuration
    global config
    config = TradingConfig()

    # Create and start trader
    trader = LiveTrader(config)

    try:
        if trader.start():
            print("✅ Live trading started successfully!")
            print("Press Ctrl+C to stop...")

            # Keep main thread alive
            while trader.running:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping live trading...")
        trader.stop()
        print("✅ Live trading stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        trader.stop()


if __name__ == "__main__":
    main()
