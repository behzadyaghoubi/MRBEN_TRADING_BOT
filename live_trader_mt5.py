#!/usr/bin/env python3
"""
MR BEN Live Trading System v3.0 - MT5 Real-Time
Enhanced live trading system with real-time MT5 data integration
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

# MT5 Integration
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("⚠️ MetaTrader5 not available, using demo mode")
    MT5_AVAILABLE = False

# AI Models
try:
    from tensorflow.keras.models import load_model
    import joblib
    from ai_filter import AISignalFilter
    AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI models not available: {e}")
    AI_AVAILABLE = False

# Enhanced AI Components
try:
    from utils.regime import detect_regime
    from utils.conformal import ConformalGate
    ENHANCED_AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced AI components not available: {e}")
    ENHANCED_AI_AVAILABLE = False

class MT5Config:
    """Configuration class for MT5 live trader."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file."""
        default_config = {
            "trading": {
                "symbol": "XAUUSD.PRO",
                "timeframe": 5,
                "bars": 500,
                "sleep_seconds": 10,
                "retry_delay": 5,
                "consecutive_signals_required": 1
            },
            "risk": {
                "volume": 0.01,
                "min_signal_confidence": 0.6
            },
            "logging": {
                "logs_dir": "logs"
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Merge with defaults
                self.SYMBOL = config.get("trading", {}).get("symbol", default_config["trading"]["symbol"])
                self.TIMEFRAME = config.get("trading", {}).get("timeframe", default_config["trading"]["timeframe"])
                self.BARS = config.get("trading", {}).get("bars", default_config["trading"]["bars"])
                self.SLEEP_SECONDS = config.get("trading", {}).get("sleep_seconds", default_config["trading"]["sleep_seconds"])
                self.RETRY_DELAY = config.get("trading", {}).get("retry_delay", default_config["trading"]["retry_delay"])
                self.CONSECUTIVE_SIGNALS_REQUIRED = config.get("trading", {}).get("consecutive_signals_required", default_config["trading"]["consecutive_signals_required"])
                
                self.VOLUME = config.get("risk", {}).get("fixed_volume", default_config["risk"]["volume"])
                self.MIN_SIGNAL_CONFIDENCE = config.get("ml", {}).get("accept_threshold", default_config["risk"]["min_signal_confidence"])
                
                self.LOGS_DIR = config.get("logging", {}).get("log_file", default_config["logging"]["logs_dir"]).replace("/trading_bot.log", "")
                
                # MT5 specific
                self.ENABLE_MT5 = not config.get("flags", {}).get("demo_mode", True)
                
            except Exception as e:
                print(f"⚠️ Error loading config: {e}, using defaults")
                self._set_defaults(default_config)
        else:
            print(f"⚠️ Config file {self.config_path} not found, using defaults")
            self._set_defaults(default_config)
    
    def _set_defaults(self, default_config):
        """Set default configuration values."""
        self.SYMBOL = default_config["trading"]["symbol"]
        self.TIMEFRAME = default_config["trading"]["timeframe"]
        self.BARS = default_config["trading"]["bars"]
        self.SLEEP_SECONDS = default_config["trading"]["sleep_seconds"]
        self.RETRY_DELAY = default_config["trading"]["retry_delay"]
        self.CONSECUTIVE_SIGNALS_REQUIRED = default_config["trading"]["consecutive_signals_required"]
        self.VOLUME = default_config["risk"]["volume"]
        self.MIN_SIGNAL_CONFIDENCE = default_config["risk"]["min_signal_confidence"]
        self.LOGS_DIR = default_config["logging"]["logs_dir"]
        self.ENABLE_MT5 = False

class MT5DataManager:
    """Manages real-time data from MT5."""
    
    def __init__(self, symbol: str = "XAUUSD.PRO", timeframe: int = mt5.TIMEFRAME_M5):
        self.symbol = symbol
        self.timeframe = timeframe
        self.last_update = None
        self.current_data = None
        
        # Initialize MT5 if available
        if MT5_AVAILABLE:
            self._initialize_mt5()
    
    def _initialize_mt5(self):
        """Initialize MT5 connection."""
        if not mt5.initialize():
            print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
            return False
        
        # Load config
        config_path = 'config/settings.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Remove JSON comments
                    import re
                    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
                    config = json.loads(content)
            except Exception as e:
                print(f"⚠️ Error loading settings.json: {e}, using fallback config")
                config = {}
            
            login = config.get('mt5_login', 1104123)
            password = config.get('mt5_password', '-4YcBgRd')
            server = config.get('mt5_server', 'OxSecurities-Demo')
        else:
            login = 1104123
            password = '-4YcBgRd'
            server = 'OxSecurities-Demo'
        
        if not mt5.login(login=login, password=password, server=server):
            print(f"❌ Failed to login to MT5: {mt5.last_error()}")
            return False
        
        print(f"✅ MT5 connected: {self.symbol}")
        return True
    
    def get_latest_data(self, bars: int = 500) -> pd.DataFrame:
        """Get latest market data from MT5."""
        if not MT5_AVAILABLE:
            return self._get_synthetic_data()
        
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
            
            if rates is None:
                print("⚠️ Failed to get MT5 data, using synthetic data")
                return self._get_synthetic_data()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Update current data
            self.current_data = df
            self.last_update = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting MT5 data: {e}")
            return self._get_synthetic_data()
    
    def get_current_tick(self) -> Optional[Dict]:
        """Get current tick data."""
        if not MT5_AVAILABLE:
            return None
        
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is not None:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'time': datetime.fromtimestamp(tick.time),
                    'volume': tick.volume
                }
        except Exception as e:
            print(f"❌ Error getting tick data: {e}")
        
        return None
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def _get_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic data for testing."""
        print("🔄 Generating synthetic data...")
        
        num_bars = 500
        data = []
        base_price = 3300.0
        
        for i in range(num_bars):
            open_price = base_price + np.random.uniform(-10, 10)
            close_price = open_price + np.random.uniform(-5, 5)
            high = max(open_price, close_price) + np.random.uniform(0, 3)
            low = min(open_price, close_price) - np.random.uniform(0, 3)
            volume = np.random.randint(100, 1000)
            
            data.append({
                'time': datetime.now() - timedelta(minutes=i*5),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'tick_volume': volume
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('time').reset_index(drop=True)
        
        # Calculate technical indicators
        df = self._calculate_technical_indicators(df)
        
        return df
    
    def shutdown(self):
        """Shutdown MT5 connection."""
        if MT5_AVAILABLE:
            mt5.shutdown()

class MT5Config:
    """Configuration for MT5 live trading."""
    
    def __init__(self):
        # Load config file
        config_path = os.path.join('config', 'settings.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
        except Exception:
            self.config_data = {}

        # Trading Parameters
        self.SYMBOL = self.config_data.get("trading", {}).get("symbol", "XAUUSD.PRO")
        self.VOLUME = self.config_data.get("trading", {}).get("min_lot", 0.01)
        self.MAGIC = self.config_data.get("trading", {}).get("magic_number", 654321)
        self.TIMEFRAME = self.config_data.get("trading", {}).get("timeframe", 5)
        self.BARS = self.config_data.get("trading", {}).get("bars", 500)

        # Enhanced Parameters
        self.SLEEP_SECONDS = 30
        self.RETRY_DELAY = 5
        self.LSTM_TIMESTEPS = 50
        self.MIN_SIGNAL_CONFIDENCE = 0.6  # Increased threshold
        self.CONSECUTIVE_SIGNALS_REQUIRED = 2
        
        # Risk Management
        self.BASE_RISK = 0.01
        self.MIN_LOT = 0.01
        self.MAX_LOT = 0.1
        self.MAX_OPEN_TRADES = 2
        
        # Demo Mode Settings
        self.DEMO_MODE = True
        self.ENABLE_MT5 = MT5_AVAILABLE
        
        # Data Directory
        self.DATA_DIR = "data"
        self.LOGS_DIR = "logs"

class MT5SignalGenerator:
    """Enhanced signal generator with MT5 integration."""
    
    def __init__(self, config: MT5Config, lstm_model=None, lstm_scaler=None, ml_filter=None):
        self.config = config
        self.lstm_model = lstm_model
        self.lstm_scaler = lstm_scaler
        self.ml_filter = ml_filter
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging."""
        logger = logging.getLogger('MT5SignalGenerator')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    def generate_enhanced_signal(self, df: pd.DataFrame) -> Dict:
        """Generate enhanced signal using LSTM, TA, and ML Filter."""
        try:
            # LSTM Signal
            lstm_signal = self._generate_lstm_signal(df)
            
            # Technical Analysis Signal
            ta_signal = self._generate_technical_signal(df)
            
            # ML Filter
            final_signal = self._apply_ml_filter(lstm_signal, ta_signal)
            
            # Log pipeline
            self.logger.info(
                f"Signal Pipeline: LSTM({lstm_signal['signal']},{lstm_signal['confidence']:.3f}) + "
                f"TA({ta_signal['signal']},{ta_signal['confidence']:.3f}) = "
                f"Final({final_signal['signal']},{final_signal['confidence']:.3f})"
            )
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error in signal generation: {e}")
            return {'signal': 0, 'confidence': 0.0, 'source': 'Error'}
    
    def _generate_lstm_signal(self, df: pd.DataFrame) -> Dict:
        """Generate LSTM signal."""
        if self.lstm_model is None or self.lstm_scaler is None:
            return {'signal': 0, 'confidence': 0.0}
        
        try:
            # Prepare features for LSTM
            features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd', 'atr']
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) < 8:
                self.logger.warning(f"Available features: {available_features}, need 8 features for LSTM")
                return {'signal': 0, 'confidence': 0.0}
            
            # Prepare sequence data
            data = df[available_features].values
            
            if len(data) < self.config.LSTM_TIMESTEPS:
                return {'signal': 0, 'confidence': 0.0}
            
            # Scale the data
            scaled_data = self.lstm_scaler.transform(data)
            
            # Prepare sequence
            sequence = scaled_data[-self.config.LSTM_TIMESTEPS:].reshape(1, self.config.LSTM_TIMESTEPS, -1)
            
            # Make prediction
            prediction = self.lstm_model.predict(sequence, verbose=0)
            
            # Get signal class and confidence
            signal_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Map signal class to signal (-1, 0, 1)
            signal_map = {0: -1, 1: 0, 2: 1}  # SELL, HOLD, BUY
            signal = signal_map[signal_class]
            
            return {
                'signal': signal,
                'confidence': confidence,
                'raw_prediction': prediction[0]
            }
            
        except Exception as e:
            self.logger.error(f"Error in LSTM signal generation: {e}")
            return {'signal': 0, 'confidence': 0.0}
    
    def _generate_technical_signal(self, df: pd.DataFrame) -> Dict:
        """Generate technical analysis signal."""
        try:
            if len(df) < 20:
                return {'signal': 0, 'confidence': 0.5}
            
            current = df.iloc[-1]
            
            # RSI Analysis
            rsi = current['rsi']
            rsi_signal = 0
            if rsi < 30:
                rsi_signal = 1  # Oversold - BUY
            elif rsi > 70:
                rsi_signal = -1  # Overbought - SELL
            
            # MACD Analysis
            macd = current['macd']
            macd_signal = current.get('macd_signal', 0)
            macd_signal_value = 0
            if macd > macd_signal:
                macd_signal_value = 1  # Bullish
            elif macd < macd_signal:
                macd_signal_value = -1  # Bearish
            
            # Combine signals
            total_signal = rsi_signal + macd_signal_value
            signal = 0
            if total_signal > 0:
                signal = 1
            elif total_signal < 0:
                signal = -1
            
            # Calculate confidence based on signal strength
            confidence = 0.5 + abs(total_signal) * 0.25
            
            return {
                'signal': signal,
                'confidence': min(confidence, 0.9),
                'rsi': rsi,
                'macd': macd
            }
            
        except Exception as e:
            self.logger.error(f"Error in technical signal generation: {e}")
            return {'signal': 0, 'confidence': 0.5}
    
    def _apply_ml_filter(self, lstm_signal: Dict, ta_signal: Dict) -> Dict:
        """Apply ML filter to combine signals."""
        try:
            if self.ml_filter is None:
                # Simple combination without ML filter
                lstm_weight = 0.7
                ta_weight = 0.3
                
                combined_signal = (lstm_signal['signal'] * lstm_weight + 
                                 ta_signal['signal'] * ta_weight)
                
                combined_confidence = (lstm_signal['confidence'] * lstm_weight + 
                                     ta_signal['confidence'] * ta_weight)
                
                # Determine final signal
                if combined_signal > 0.3:
                    final_signal = 1
                elif combined_signal < -0.3:
                    final_signal = -1
                else:
                    final_signal = 0
                
                return {
                    'signal': final_signal,
                    'confidence': combined_confidence,
                    'source': 'LSTM_TA_Combined'
                }
            
            # Use ML filter
            features = [
                lstm_signal['signal'],
                lstm_signal['confidence'],
                ta_signal['signal'],
                ta_signal['confidence']
            ]
            
            ml_result = self.ml_filter.filter_signal(features)
            
            return {
                'signal': ml_result['signal'],
                'confidence': ml_result['confidence'],
                'source': 'LSTM_TA_ML_Pipeline'
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML filter: {e}")
            return {'signal': 0, 'confidence': 0.0, 'source': 'Error'}

class MT5LiveTrader:
    """Enhanced live trader with MT5 integration."""
    
    def __init__(self):
        self.config = MT5Config()
        self.data_manager = MT5DataManager(self.config.SYMBOL)
        
        # Load AI models
        self.lstm_model, self.lstm_scaler = self._load_lstm_model()
        self.ml_filter = self._load_ml_filter()
        
        # Initialize signal generator
        self.signal_generator = MT5SignalGenerator(
            self.config, self.lstm_model, self.lstm_scaler, self.ml_filter
        )
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize enhanced AI components
        self.conformal = None
        if ENHANCED_AI_AVAILABLE:
            try:
                self.conformal = ConformalGate("models/meta_filter.joblib", "models/conformal.json")
                if self.conformal.is_available():
                    self.logger.info("✅ Conformal gate loaded successfully")
                else:
                    self.logger.warning("⚠️ Conformal gate not available")
                    self.conformal = None
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load conformal gate: {e}")
                self.conformal = None
        
        # Enhanced trading configuration
        self.tp_policy = {
            "split": True,
            "tp1_r": 0.8,
            "tp2_r": 1.5,
            "tp1_share": 0.5,
            "breakeven_after_tp1": True
        }
        self.min_R_after_round = 1.2
        self.k_tp = 0.20
        self.max_spread_atr_frac = 0.10
        
        # Trading state
        self.running = False
        self.consecutive_signals = 0
        self.last_signal = 0
        
    def _setup_logger(self):
        """Setup logging with UTF-8 encoding."""
        os.makedirs(self.config.LOGS_DIR, exist_ok=True)
        
        logger = logging.getLogger('MT5LiveTrader')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler with UTF-8 encoding
        log_file = os.path.join(self.config.LOGS_DIR, 'mt5_live_trader.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_lstm_model(self):
        """Load LSTM model and scaler."""
        try:
            # Try balanced model first
            model_path = 'models/lstm_balanced_model.h5'
            scaler_path = 'models/lstm_balanced_scaler.save'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print("Loading Balanced LSTM Model...")
                lstm_model = load_model(model_path)
                lstm_scaler = joblib.load(scaler_path)
                print("✅ Balanced LSTM Model loaded successfully!")
                return lstm_model, lstm_scaler
            
            # Fallback to advanced model
            model_path = 'models/advanced_lstm_model.h5'
            scaler_path = 'models/advanced_lstm_scaler.save'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print("Loading Advanced LSTM Model...")
                lstm_model = load_model(model_path)
                lstm_scaler = joblib.load(scaler_path)
                print("✅ Advanced LSTM Model loaded successfully!")
                return lstm_model, lstm_scaler
            
            print("⚠️ No LSTM model found")
            return None, None
            
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
            return None, None
    
    def _load_ml_filter(self):
        """Load ML filter."""
        try:
            ml_filter_path = 'models/mrben_ai_signal_filter_xgb.joblib'
            
            if os.path.exists(ml_filter_path):
                print("Loading ML Filter...")
                ml_filter = AISignalFilter(
                    model_path=ml_filter_path,
                    model_type="joblib",
                    threshold=0.65
                )
                print("✅ ML Filter loaded successfully!")
                return ml_filter
            
            print("⚠️ ML Filter not found")
            return None
            
        except Exception as e:
            print(f"❌ Error loading ML Filter: {e}")
            return None
    
    def start(self):
        """Start the live trading system."""
        self.logger.info("🚀 Starting MT5 Live Trading System v3.0...")
        self.logger.info(f"📊 Symbol: {self.config.SYMBOL}")
        self.logger.info(f"📊 Timeframe: M{self.config.TIMEFRAME}")
        self.logger.info(f"📊 Volume: {self.config.VOLUME}")
        self.logger.info(f"📊 Threshold: {self.config.MIN_SIGNAL_CONFIDENCE}")
        self.logger.info(f"📊 Consecutive Signals: {self.config.CONSECUTIVE_SIGNALS_REQUIRED}")
        self.logger.info(f"📊 MT5 Enabled: {self.config.ENABLE_MT5}")
        
        self.running = True
        
        # Start trading loop in a separate thread
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        self.logger.info("✅ Trading system started successfully!")
    
    def stop(self):
        """Stop the live trading system."""
        self.logger.info("🛑 Stopping MT5 Live Trading System...")
        self.running = False
        
        if hasattr(self, 'trading_thread'):
            self.trading_thread.join(timeout=5)
        
        self.data_manager.shutdown()
        self.logger.info("✅ Trading system stopped")
    
    def _trading_loop(self):
        """Main trading loop."""
        while self.running:
            try:
                # Get latest market data
                df = self.data_manager.get_latest_data(self.config.BARS)
                
                if df is None or len(df) < 50:
                    self.logger.warning("Insufficient data, skipping iteration")
                    time.sleep(self.config.RETRY_DELAY)
                    continue
                
                # Generate signal
                signal_data = self.signal_generator.generate_enhanced_signal(df)
                
                # --- Build feature vector for Meta/Conformal ---
                last = df.iloc[-1]
                meta_feats = {
                    "close": float(last['close']),
                    "ret": float(df['close'].pct_change().iloc[-1]) if len(df) > 1 else 0.0,
                    "sma_20": float(last.get('sma_20', 0.0)),
                    "sma_50": float(last.get('sma_50', 0.0)),
                    "atr": float(last.get('atr', 0.0)),
                    "rsi": float(last.get('rsi', 50.0)),
                    "macd": float(last.get('macd', 0.0)),
                    "macd_signal": float(last.get('macd_signal', 0.0)),
                    "hour": float(last['time'].hour if 'time' in df.columns else datetime.now().hour),
                    "dow": float(last['time'].dayofweek if 'time' in df.columns else datetime.now().weekday())
                }

                # Check consecutive signals
                if signal_data['signal'] == self.last_signal:
                    self.consecutive_signals += 1
                else:
                    self.consecutive_signals = 1
                    self.last_signal = signal_data['signal']

                # If signal is zero, skip conformal check
                if signal_data['signal'] == 0:
                    time.sleep(self.config.SLEEP_SECONDS)
                    continue

                # Conformal accept?
                conformal_ok = True
                conformal_p = None
                regime = "UNKNOWN"
                
                if ENHANCED_AI_AVAILABLE:
                    # Detect market regime
                    regime = detect_regime(last) if hasattr(last, 'get') else "UNKNOWN"
                    
                    # Apply conformal filter
                    if self.conformal is not None:
                        conformal_ok, p_hat, nonconf = self.conformal.accept(meta_feats)
                        conformal_p = p_hat
                        self.logger.info(f"Conformal: accept={conformal_ok} p={p_hat:.3f} regime={regime}")
                        if not conformal_ok:
                            time.sleep(self.config.SLEEP_SECONDS)
                            continue

                # Adjust threshold based on conformal probability
                thr = self.config.MIN_SIGNAL_CONFIDENCE
                adj_thr = max(thr, 0.55 if conformal_p is not None else thr)

                # Check if we should execute trade
                should_execute = (
                    signal_data['confidence'] >= adj_thr and
                    self.consecutive_signals >= self.config.CONSECUTIVE_SIGNALS_REQUIRED and
                    signal_data['signal'] != 0 and
                    conformal_ok
                )
                
                # Log signal
                self.logger.info(
                    f"Signal: {signal_data['signal']} | "
                    f"Confidence: {signal_data['confidence']:.3f} | "
                    f"Consecutive: {self.consecutive_signals} | "
                    f"Source: {signal_data.get('source', 'Unknown')}"
                )
                
                # Execute trade if conditions met
                if should_execute:
                    self.logger.info(f"Executing trade: Signal={signal_data['signal']}")
                    self._execute_trade(signal_data, df)
                
                # Sleep before next iteration
                time.sleep(self.config.SLEEP_SECONDS)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(self.config.RETRY_DELAY)
    
    def _execute_trade(self, signal_data: Dict, df: pd.DataFrame):
        """Execute enhanced trade with split TP and conformal insights."""
        try:
            # Get current price from MT5 or data
            current_price = df['close'].iloc[-1]
            
            # Get current tick for more accurate pricing
            tick_data = self.data_manager.get_current_tick()
            if tick_data:
                if signal_data['signal'] == 1:  # BUY
                    current_price = tick_data['ask']
                else:  # SELL
                    current_price = tick_data['bid']
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Enhanced trade execution with split TP
            if signal_data['signal'] == 1:  # BUY
                action = "BUY"
                entry_price = current_price
                side = 1
            elif signal_data['signal'] == -1:  # SELL
                action = "SELL"
                entry_price = current_price
                side = -1
            else:
                return

            # Calculate dynamic SL/TP based on ATR if available
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 50
            risk = atr * 1.5  # 1.5x ATR for stop loss
            
            sl_price = entry_price - (side * risk)
            
            # Calculate split TP levels
            if self.tp_policy["split"]:
                tp1 = entry_price + (side * risk * self.tp_policy["tp1_r"])
                tp2 = entry_price + (side * risk * max(self.tp_policy["tp2_r"], self.min_R_after_round))
                
                # Log split TP trade
                volume_total = self.config.VOLUME
                v1 = volume_total * self.tp_policy["tp1_share"]
                v2 = volume_total - v1
                
                self.logger.info(
                    f"Enhanced {action} executed: {self.config.SYMBOL} at {entry_price:.2f} "
                    f"Split TP: v1={v1:.2f}@{tp1:.2f}, v2={v2:.2f}@{tp2:.2f} "
                    f"SL: {sl_price:.2f} | Risk: {risk:.2f} "
                    f"Confidence: {signal_data['confidence']:.3f}"
                )
            else:
                tp_price = entry_price + (side * risk * 2.0)  # 2:1 R/R ratio
                
                self.logger.info(
                    f"Enhanced {action} executed: {self.config.SYMBOL} at {entry_price:.2f} "
                    f"(SL: {sl_price:.2f}, TP: {tp_price:.2f}) "
                    f"Risk: {risk:.2f} | Confidence: {signal_data['confidence']:.3f}"
                )
            
            # Enhanced trade log with conformal and regime data
            trade_log = {
                'timestamp': timestamp,
                'symbol': self.config.SYMBOL,
                'action': action,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp2 if self.tp_policy["split"] else tp_price,
                'tp1_price': tp1 if self.tp_policy["split"] else None,
                'volume': self.config.VOLUME,
                'confidence': signal_data['confidence'],
                'source': signal_data.get('source', 'Unknown'),
                'mt5_data': tick_data is not None,
                'conformal_accepted': True,  # Since we only execute if accepted
                'regime': signal_data.get('regime', 'UNKNOWN'),
                'atr_risk': risk,
                'split_tp': self.tp_policy["split"]
            }
            
            # Save to demo trades file
            demo_file = os.path.join(self.config.LOGS_DIR, 'mt5_trades.csv')
            trade_df = pd.DataFrame([trade_log])
            
            if os.path.exists(demo_file):
                trade_df.to_csv(demo_file, mode='a', header=False, index=False)
            else:
                trade_df.to_csv(demo_file, index=False)
            
            self.logger.info(
                f"Demo {action} executed: {self.config.SYMBOL} at {entry_price:.2f} "
                f"(SL: {sl_price:.2f}, TP: {tp_price:.2f}) "
                f"Confidence: {signal_data['confidence']:.3f} "
                f"MT5: {'Yes' if tick_data else 'No'}"
            )
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")

def main():
    """Main function."""
    print("🎯 MR BEN MT5 Live Trading System v3.0")
    print("=" * 50)
    
    # Create and start trader
    trader = MT5LiveTrader()
    
    try:
        trader.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal...")
        trader.stop()
        print("✅ System shutdown complete")

if __name__ == "__main__":
    main() 