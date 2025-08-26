#!/usr/bin/env python3
"""
MR BEN Enhanced Live Trader v2.0
Advanced LSTM + ML Filter Pipeline
Professional Trading System
"""

import os
import sys
import time
import json
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =====================
# ADVANCED LSTM MODEL LOADING
# =====================
try:
    from tensorflow.keras.models import load_model
    import joblib
    
    # Model paths
    # Try balanced model first, then fallback to others
    LSTM_MODEL_PATH = 'models/lstm_balanced_model.h5'  # New balanced model
    LSTM_SCALER_PATH = 'models/lstm_balanced_scaler.save'
    
    # Load LSTM Model
    if os.path.exists(LSTM_MODEL_PATH):
        print("Loading Balanced LSTM Model...")
        lstm_model = load_model(LSTM_MODEL_PATH)
        lstm_scaler = joblib.load(LSTM_SCALER_PATH)
        print("Balanced LSTM Model loaded successfully!")
        print(f"Model structure: {lstm_model.input_shape} -> {lstm_model.output_shape}")
    else:
        # Try advanced model path
        alt_model_path = 'models/advanced_lstm_model.h5'
        if os.path.exists(alt_model_path):
            print("Loading Advanced LSTM Model...")
            lstm_model = load_model(alt_model_path)
            lstm_scaler = joblib.load('models/advanced_lstm_scaler.save')
            print("Advanced LSTM Model loaded successfully!")
        else:
            print("LSTM model not found, will use fallback")
            lstm_model = None
            lstm_scaler = None
        
except Exception as e:
    print(f"❌ Error loading LSTM model: {e}")
    lstm_model = None
    lstm_scaler = None

# =====================
# ML FILTER LOADING
# =====================
try:
    from ai_filter import AISignalFilter
    
    ML_FILTER_PATH = 'models/mrben_ai_signal_filter_xgb.joblib'
    ML_FILTER_THRESHOLD = 0.65
    
    if os.path.exists(ML_FILTER_PATH):
        print("🔍 Loading ML Filter...")
        ml_filter = AISignalFilter(
            model_path=ML_FILTER_PATH,
            model_type="joblib",
            threshold=ML_FILTER_THRESHOLD
        )
        print("✅ ML Filter loaded successfully!")
    else:
        print("⚠️ ML Filter not found")
        ml_filter = None
        
except Exception as e:
    print(f"❌ Error loading ML Filter: {e}")
    ml_filter = None

# =====================
# ENHANCED CONFIGURATION
# =====================
class EnhancedConfig:
    """Enhanced configuration for v2.0 trading system."""
    
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
        self.VOLUME = self.config_data.get("trading", {}).get("min_lot", 0.01)  # DEMO: Low volume
        self.MAGIC = self.config_data.get("trading", {}).get("magic_number", 654321)
        self.TIMEFRAME = self.config_data.get("trading", {}).get("timeframe", 5)
        self.BARS = self.config_data.get("trading", {}).get("bars", 500)

        # Enhanced Parameters
        self.SLEEP_SECONDS = 30
        self.RETRY_DELAY = 5
        self.LSTM_TIMESTEPS = 50
        self.MIN_SIGNAL_CONFIDENCE = 0.5  # افزایش به 0.5 برای دقت بیشتر
        self.CONSECUTIVE_SIGNALS_REQUIRED = 2  # افزایش به 2 برای کاهش overtrading
        
        # Risk Management - DEMO MODE
        self.BASE_RISK = 0.01  # Very low risk for demo
        self.MIN_LOT = 0.01
        self.MAX_LOT = 0.1  # Low max lot for demo
        self.MAX_OPEN_TRADES = 2  # Fewer trades for demo
        
        # Demo Mode Settings
        self.DEMO_MODE = True  # Set to False for live trading
        self.ENABLE_MT5 = False  # Set to True to enable MT5
        
        # Data Directory
        self.DATA_DIR = "data"
        self.LOGS_DIR = "logs"

# =====================
# ENHANCED SIGNAL GENERATOR
# =====================
class EnhancedSignalGenerator:
    """Enhanced signal generator with LSTM + ML Filter pipeline."""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.lstm_model = lstm_model
        self.lstm_scaler = lstm_scaler
        self.ml_filter = ml_filter
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging."""
        logger = logging.getLogger('EnhancedSignalGenerator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
            logger.addHandler(ch)
        
        return logger
    
    def generate_enhanced_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate enhanced signal using LSTM + ML Filter pipeline
        """
        try:
            # Step 1: LSTM Signal Generation
            lstm_result = self._generate_lstm_signal(df)
            
            # Step 2: Technical Analysis Confirmation
            ta_result = self._generate_technical_signal(df)
            
            # Step 3: ML Filter Validation
            final_result = self._apply_ml_filter(lstm_result, ta_result, df)
            
            # Log detailed information
            self.logger.info(
                f"📊 Signal Pipeline: LSTM({lstm_result['signal']},{lstm_result['confidence']:.3f}) + "
                f"TA({ta_result['signal']},{ta_result['confidence']:.3f}) + "
                f"ML({final_result['ml_confidence']:.3f}) = "
                f"Final({final_result['signal']},{final_result['confidence']:.3f})"
            )
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced signal generation: {e}")
            return {
                'signal': 0,
                'confidence': 0.0,
                'lstm_signal': 0,
                'lstm_confidence': 0.0,
                'ta_signal': 0,
                'ta_confidence': 0.0,
                'ml_confidence': 0.0,
                'source': 'Error_Fallback'
            }
    
    def _generate_lstm_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate LSTM signal with confidence."""
        if self.lstm_model is None or self.lstm_scaler is None:
            return {'signal': 0, 'confidence': 0.0}
            
        try:
            # Prepare features for LSTM - Use 8 features to match the trained model
            features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd', 'atr']
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) < 8:
                self.logger.warning(f"Available features: {available_features}, need 8 features for LSTM")
                return {'signal': 0, 'confidence': 0.0}
                
            # Prepare sequence data
            data = df[available_features].values
            scaled_data = self.lstm_scaler.transform(data)
            
            # Create sequence
            if len(scaled_data) >= self.config.LSTM_TIMESTEPS:
                sequence = scaled_data[-self.config.LSTM_TIMESTEPS:].reshape(1, self.config.LSTM_TIMESTEPS, -1)
                
                # Predict
                prediction = self.lstm_model.predict(sequence, verbose=0)
                
                # Convert to signal
                signal_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                # Map to signal: 0=SELL, 1=HOLD, 2=BUY
                signal_map = {0: -1, 1: 0, 2: 1}  # SELL, HOLD, BUY
                signal = signal_map[signal_class]
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'raw_prediction': prediction[0]
                }
            else:
                return {'signal': 0, 'confidence': 0.0}
                
        except Exception as e:
            self.logger.error(f"Error in LSTM signal generation: {e}")
            return {'signal': 0, 'confidence': 0.0}
    
    def _generate_technical_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate technical analysis signal."""
        try:
            # Calculate technical indicators
            rsi = self._calculate_rsi(df['close'])
            macd, macd_signal = self._calculate_macd(df['close'])
            bb_upper, bb_lower = self._calculate_bollinger_bands(df['close'])
            
            # Enhanced technical signal logic
            signal = 0
            confidence = 0.5
            
            current_price = df['close'].iloc[-1]
            current_rsi = rsi[-1] if len(rsi) > 0 else 50
            current_macd = macd[-1] if len(macd) > 0 else 0
            current_macd_signal = macd_signal[-1] if len(macd_signal) > 0 else 0
            
            # RSI conditions
            rsi_buy = current_rsi < 30
            rsi_sell = current_rsi > 70
            
            # MACD conditions
            macd_buy = current_macd > current_macd_signal and current_macd > 0
            macd_sell = current_macd < current_macd_signal and current_macd < 0
            
            # Bollinger Bands conditions
            bb_buy = current_price < bb_lower[-1] if len(bb_lower) > 0 else False
            bb_sell = current_price > bb_upper[-1] if len(bb_upper) > 0 else False
            
            # Combine signals
            buy_signals = sum([rsi_buy, macd_buy, bb_buy])
            sell_signals = sum([rsi_sell, macd_sell, bb_sell])
            
            if buy_signals >= 2:
                signal = 1
                confidence = 0.6 + (buy_signals * 0.1)
            elif sell_signals >= 2:
                signal = -1
                confidence = 0.6 + (sell_signals * 0.1)
            else:
                signal = 0
                confidence = 0.5
                
            return {
                'signal': signal,
                'confidence': min(confidence, 0.9),
                'indicators': {
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'macd_signal': current_macd_signal,
                    'bb_upper': bb_upper[-1] if len(bb_upper) > 0 else 0,
                    'bb_lower': bb_lower[-1] if len(bb_lower) > 0 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in technical signal generation: {e}")
            return {'signal': 0, 'confidence': 0.0}
    
    def _apply_ml_filter(self, lstm_result: Dict, ta_result: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply ML filter to combine signals."""
        try:
            # Prepare features for ML filter
            features = {
                'RSI': self._calculate_rsi(df['close'])[-1] if len(df) > 0 else 50,
                'MACD': self._calculate_macd(df['close'])[0][-1] if len(df) > 0 else 0,
                'ATR': self._calculate_atr(df)[-1] if len(df) > 0 else 0,
                'Volume': df['tick_volume'].iloc[-1] if 'tick_volume' in df.columns else 0
            }
            
            # Get ML filter prediction
            if self.ml_filter is not None:
                ml_prediction = self.ml_filter.predict(features)
                ml_confidence = self.ml_filter.get_confidence(features)
            else:
                ml_prediction = 0.5
                ml_confidence = 0.5
            
            # Combine signals with weighted approach
            lstm_weight = 0.4
            ta_weight = 0.3
            ml_weight = 0.3
            
            # Calculate weighted confidence
            weighted_confidence = (
                lstm_result['confidence'] * lstm_weight +
                ta_result['confidence'] * ta_weight +
                ml_confidence * ml_weight
            )
            
            # Determine final signal
            if lstm_result['signal'] == ta_result['signal'] and lstm_result['signal'] != 0:
                # Both signals agree
                final_signal = lstm_result['signal']
                if ml_confidence > self.config.MIN_SIGNAL_CONFIDENCE:
                    final_confidence = weighted_confidence
                else:
                    final_signal = 0
                    final_confidence = 0.0
            else:
                # Signals disagree - use highest confidence
                if lstm_result['confidence'] > ta_result['confidence']:
                    final_signal = lstm_result['signal']
                    final_confidence = lstm_result['confidence']
                else:
                    final_signal = ta_result['signal']
                    final_confidence = ta_result['confidence']
                
                # Apply ML filter as tiebreaker
                if ml_confidence < self.config.MIN_SIGNAL_CONFIDENCE:
                    final_signal = 0
                    final_confidence = 0.0
            
            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'lstm_signal': lstm_result['signal'],
                'lstm_confidence': lstm_result['confidence'],
                'ta_signal': ta_result['signal'],
                'ta_confidence': ta_result['confidence'],
                'ml_confidence': ml_confidence,
                'source': 'Enhanced_LSTM_ML_Pipeline_v2'
            }
                    
        except Exception as e:
            self.logger.error(f"Error in ML filter application: {e}")
            return {
                'signal': 0,
                'confidence': 0.0,
                'lstm_signal': lstm_result['signal'],
                'lstm_confidence': lstm_result['confidence'],
                'ta_signal': ta_result['signal'],
                'ta_confidence': ta_result['confidence'],
                'ml_confidence': 0.0,
                'source': 'Error_Fallback'
            }
    
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
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.values, lower_band.values
    
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
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.signal_generator = EnhancedSignalGenerator(config)
        self.running = False
        self.stop_event = threading.Event()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Trading state
        self.last_signal = 0
        self.consecutive_signals = 0
        self.trade_count = 0
        
    def _setup_logger(self):
        """Setup enhanced logging."""
        logger = logging.getLogger('EnhancedLiveTrader')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        os.makedirs(self.config.LOGS_DIR, exist_ok=True)
        
        if not logger.handlers:
            # File handler with UTF-8 encoding
            fh = logging.FileHandler(f'{self.config.LOGS_DIR}/gold_live_trader.log', encoding='utf-8')
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter without emojis for Windows compatibility
            formatter = logging.Formatter(
                '[%(asctime)s][%(levelname)s] %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger
    
    def start(self):
        """Start enhanced live trading."""
        try:
            self.logger.info("Starting MR BEN Enhanced Live Trading System v2.0...")
            
            # Check model availability
            if self.signal_generator.lstm_model is None:
                self.logger.warning("LSTM model not available")
            
            if self.signal_generator.ml_filter is None:
                self.logger.warning("ML Filter not available")
            
            # Start trading loop
            self.running = True
            self.stop_event.clear()
            
            trading_thread = threading.Thread(target=self._trading_loop)
            trading_thread.daemon = True
            trading_thread.start()
            
            self.logger.info("Enhanced Live Trading System started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting enhanced live trading: {e}")
            return False
    
    def stop(self):
        """Stop enhanced live trading."""
        self.logger.info("Stopping Enhanced Live Trading System...")
        self.running = False
        self.stop_event.set()
        self.logger.info("Enhanced Live Trading System stopped")
    
    def _trading_loop(self):
        """Enhanced trading loop with advanced signal processing."""
        while self.running and not self.stop_event.is_set():
            try:
                # Get market data
                df = self._get_market_data()
                
                if df is None or len(df) < self.config.LSTM_TIMESTEPS:
                    self.logger.warning("Insufficient market data")
                    time.sleep(self.config.RETRY_DELAY)
                    continue
                
                # Generate enhanced signal
                signal_data = self.signal_generator.generate_enhanced_signal(df)
                
                # Signal validation
                if signal_data['signal'] == self.last_signal and signal_data['signal'] != 0:
                    self.consecutive_signals += 1
                else:
                    self.consecutive_signals = 0
                    self.last_signal = signal_data['signal']
                
                # Log signal information
                self.logger.info(
                    f"Signal: {signal_data['signal']} | "
                    f"Confidence: {signal_data['confidence']:.3f} | "
                    f"Consecutive: {self.consecutive_signals} | "
                    f"Source: {signal_data['source']}"
                )
                
                # Execute trade if conditions are met
                if (signal_data['signal'] != 0 and 
                    signal_data['confidence'] >= self.config.MIN_SIGNAL_CONFIDENCE and
                    self.consecutive_signals >= self.config.CONSECUTIVE_SIGNALS_REQUIRED):
                    
                    self.logger.info(f"Executing trade: Signal={signal_data['signal']}")
                    self._execute_trade(signal_data, df)
                
                time.sleep(self.config.SLEEP_SECONDS)
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(self.config.RETRY_DELAY)
    
    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Get market data with priority for XAUUSD.PRO."""
        try:
            # Priority 1: Try to load enhanced XAUUSD.PRO data
            xauusd_files = [
                'data/XAUUSD_PRO_M5_enhanced.csv',
                'data/XAUUSD_PRO_M5_data.csv',
                'data/XAUUSD_PRO_M15_history.csv'
            ]
            
            for file_path in xauusd_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if len(df) >= self.config.LSTM_TIMESTEPS:
                        self.logger.info(f"📊 Loaded XAUUSD.PRO data from {file_path}")
                        return df.tail(self.config.LSTM_TIMESTEPS)
            
            # Priority 2: Try to load other available data
            data_files = [
                'data/ohlc_data.csv',
                'data/adausd_data.csv',
                'data/lstm_signals_features.csv'
            ]
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if len(df) >= self.config.LSTM_TIMESTEPS:
                        self.logger.info(f"📊 Loaded fallback data from {file_path}")
                        return df.tail(self.config.LSTM_TIMESTEPS)
            
            # Priority 3: Generate synthetic data if no real data available
            self.logger.warning("⚠️ No real data found, using synthetic data")
            return self._generate_synthetic_data()
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic trading data for testing."""
        np.random.seed(int(time.time()))
        
        n_samples = self.config.LSTM_TIMESTEPS
        base_price = 2000.0
        
        prices = [base_price]
        for i in range(1, n_samples):
            change = np.random.normal(0, 0.01)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'tick_volume': np.random.randint(100, 1000, n_samples)
        })
        
        return df
    
    def _execute_trade(self, signal_data: Dict, df: pd.DataFrame):
        """Execute trade with demo/live mode support."""
        try:
            self.trade_count += 1
            current_price = df['close'].iloc[-1]
            
            # Calculate position size based on risk
            position_size = self.config.VOLUME
            
            # Log trade details
            self.logger.info(
                f"💰 Trade #{self.trade_count}: "
                f"Signal={signal_data['signal']} | "
                f"Price={current_price:.2f} | "
                f"Confidence={signal_data['confidence']:.3f} | "
                f"Volume={position_size} | "
                f"Mode={'DEMO' if self.config.DEMO_MODE else 'LIVE'}"
            )
            
            # Demo Mode: Just log the trade
            if self.config.DEMO_MODE:
                self.logger.info(f"🎮 DEMO MODE: Simulated trade executed")
                
                # Save trade to demo log
                trade_log = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'trade_id': self.trade_count,
                    'signal': signal_data['signal'],
                    'price': current_price,
                    'volume': position_size,
                    'confidence': signal_data['confidence'],
                    'mode': 'DEMO',
                    'status': 'EXECUTED'
                }
                
                # Save to demo trades file
                demo_file = f"{self.config.LOGS_DIR}/gold_trades.csv"
                trade_df = pd.DataFrame([trade_log])
                if os.path.exists(demo_file):
                    trade_df.to_csv(demo_file, mode='a', header=False, index=False)
                else:
                    trade_df.to_csv(demo_file, index=False)
                
            # Live Mode: Execute actual MT5 trade
            else:
                if self.config.ENABLE_MT5:
                    self.logger.info(f"🚀 LIVE MODE: Executing MT5 trade...")
                    # Add MT5 trade execution here
                    # self._execute_mt5_trade(signal_data, current_price, position_size)
                else:
                    self.logger.warning("⚠️ LIVE MODE enabled but MT5 not configured")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")

# =====================
# MAIN FUNCTION
# =====================
def main():
    """Main function to run enhanced live trader."""
    print("🚀 MR BEN Enhanced Live Trading System v2.0")
    print("=" * 60)
    print("Advanced LSTM + ML Filter Pipeline")
    print("Professional Trading System")
    print("=" * 60)
    
    # Initialize configuration
    config = EnhancedConfig()
    
    # Initialize enhanced live trader
    trader = EnhancedLiveTrader(config)
    
    try:
        # Start trading
        if trader.start():
            print("✅ Enhanced Live Trading System started successfully!")
            print("📊 Monitoring signals... Press Ctrl+C to stop")
            print("🔍 Check logs/enhanced_live_trader_v2.log for details")
            
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