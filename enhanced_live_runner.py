import time
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any

# Import enhanced components
from enhanced_risk_manager import EnhancedRiskManager
from enhanced_trade_executor import EnhancedTradeExecutor

# Import existing components
try:
    from lstm_live_signal_generator import generate_lstm_live_signal
except ImportError:
    from strategies.lstm_live_signal_generator import generate_lstm_live_signal

from ai_filter import AISignalFilter
import os

class EnhancedLiveRunner:
    """
    Enhanced Live Trading Runner with:
    - Dynamic ATR-based TP/SL
    - Trailing stop management
    - Adaptive confidence thresholds
    - Comprehensive risk management
    """
    
    def __init__(self, symbol: str = "XAUUSD", timeframe=mt5.TIMEFRAME_M5):
        self.symbol = symbol
        self.timeframe = timeframe
        self.last_signal = None
        self.start_balance = None
        self.last_trailing_update = datetime.now()
        self.trailing_update_interval = 30  # seconds
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.risk_manager = EnhancedRiskManager(
            base_risk=0.02,
            min_lot=0.01,
            max_lot=0.1,
            max_open_trades=2,
            max_drawdown=0.10,
            atr_period=14,
            sl_atr_multiplier=2.0,
            tp_atr_multiplier=4.0,
            trailing_atr_multiplier=1.5,
            base_confidence_threshold=0.5,
            adaptive_confidence=True,
            performance_window=20,
            confidence_adjustment_factor=0.1
        )
        
        self.trade_executor = EnhancedTradeExecutor(self.risk_manager)
        
        # Initialize AI filter
        try:
            # Try to load from models directory first, then current directory
            model_paths = [
                "models/mrben_ai_signal_filter_xgb.joblib",
                "mrben_ai_signal_filter_xgb.joblib"
            ]
            
            ai_filter_loaded = False
            for model_path in model_paths:
                try:
                    self.ai_filter = AISignalFilter(
                        model_path=model_path, 
                        model_type="joblib", 
                        threshold=0.55
                    )
                    self.logger.info(f"AI filter loaded successfully from: {model_path}")
                    ai_filter_loaded = True
                    break
                except Exception as e:
                    self.logger.debug(f"Failed to load AI filter from {model_path}: {e}")
                    continue
            
            if not ai_filter_loaded:
                raise Exception("Could not load AI filter from any location")
                
        except Exception as e:
            self.logger.warning(f"AI filter initialization failed: {e}")
            self.ai_filter = None
        
        # Setup trade logging
        self.setup_trade_logging()
        
        self.logger.info("Enhanced Live Runner initialized successfully")

    def setup_logging(self):
        """Setup comprehensive logging"""
        self.logger = logging.getLogger("EnhancedLiveRunner")
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        file_handler = logging.FileHandler('logs/enhanced_live_runner.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        # Set encoding for console output
        import sys
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def setup_trade_logging(self):
        """Setup trade logging file"""
        self.log_path = "enhanced_live_trades.csv"
        if not os.path.exists(self.log_path):
            columns = ["time", "symbol", "signal", "price", "confidence", "sl", "tp", "volume"]
            pd.DataFrame(columns=columns).to_csv(self.log_path, index=False)

    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        if not mt5.initialize():
            self.logger.error(f"❌ MT5 connection failed: {mt5.last_error()}")
            return False
            
        if not mt5.symbol_select(self.symbol, True):
            self.logger.error(f"❌ Symbol {self.symbol} selection failed")
            return False
            
        # Get initial balance
        account_info = mt5.account_info()
        if account_info:
            self.start_balance = account_info.balance
            self.logger.info(f"✅ Connected to MT5. Balance: {self.start_balance}")
        else:
            self.logger.error("❌ Failed to get account info")
            return False
            
        return True

    def get_price_data(self, bars: int = 100) -> Optional[pd.DataFrame]:
        """Get price data from MT5"""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None:
            self.logger.error("❌ Failed to get price data")
            return None
            
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_current_confidence_threshold(self) -> float:
        """Get current adaptive confidence threshold"""
        return self.risk_manager.get_current_confidence_threshold()

    def should_execute_signal(self, signal: int, confidence: float) -> bool:
        """Check if signal should be executed based on confidence and rules"""
        # Get current adaptive threshold
        threshold = self.get_current_confidence_threshold()
        
        # Check minimum confidence
        if confidence < threshold:
            self.logger.info(f"Signal confidence {confidence:.3f} below threshold {threshold:.3f}")
            return False
        
        # Check if signal is different from last signal
        if signal == self.last_signal:
            self.logger.info("Signal is same as last signal, skipping")
            return False
        
        # Check if we can open new trade
        account_info = self.trade_executor.get_account_info()
        open_positions = self.risk_manager.get_open_positions(self.symbol)
        open_trades_count = len(open_positions)
        
        if not self.risk_manager.can_open_new_trade(
            account_info['balance'], 
            self.start_balance, 
            open_trades_count
        ):
            return False
        
        return True

    def log_trade(self, signal: str, price: float, confidence: float, sl: float, tp: float, volume: float):
        """Log trade execution"""
        new_row = {
            "time": datetime.now(),
            "symbol": self.symbol,
            "signal": signal,
            "price": price,
            "confidence": confidence,
            "sl": sl,
            "tp": tp,
            "volume": volume
        }
        
        df = pd.read_csv(self.log_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.log_path, index=False)

    def update_trailing_stops(self):
        """Update trailing stops for all open positions"""
        current_time = datetime.now()
        if (current_time - self.last_trailing_update).seconds >= self.trailing_update_interval:
            updated_count = self.trade_executor.update_trailing_stops(self.symbol)
            if updated_count > 0:
                self.logger.info(f"🔄 Updated {updated_count} trailing stops")
            self.last_trailing_update = current_time

    def check_closed_positions(self):
        """Check for closed positions and update performance"""
        # Get recent trade history
        history = self.trade_executor.get_trade_history(self.symbol, days=1)
        
        if not history.empty:
            # Check for closed positions (profit != 0)
            closed_trades = history[history['profit'] != 0]
            
            for _, trade in closed_trades.iterrows():
                # Update performance tracking
                self.risk_manager.update_performance(trade['profit'])
                
                # Remove from trailing stop monitoring if still there
                self.risk_manager.remove_trailing_stop(trade['ticket'])
                
                self.logger.info(f"Position {trade['ticket']} closed with profit: {trade['profit']}")

    def run_single_cycle(self) -> bool:
        """Run a single trading cycle"""
        try:
            # Update trailing stops
            self.update_trailing_stops()
            
            # Check for closed positions
            self.check_closed_positions()
            
            # Get market data
            df = self.get_price_data()
            if df is None:
                return False
            
            # Generate LSTM signal
            signal = generate_lstm_live_signal(df)
            
            # Apply AI filter if available
            if self.ai_filter:
                try:
                    filtered_signal = self.ai_filter.filter_signal(signal)
                    # Use AI filter result as confidence
                    confidence = self.ai_filter.filter_signal(signal, as_label=False)
                except Exception as e:
                    self.logger.warning(f"AI filter failed: {e}")
                    filtered_signal = signal
                    confidence = 0.6  # Default confidence
            else:
                filtered_signal = signal
                confidence = 0.6  # Default confidence
            
            # Convert signal to string
            signal_str = "BUY" if filtered_signal == 2 else "SELL" if filtered_signal == 0 else "HOLD"
            
            # Check if we should execute the signal
            if signal_str != "HOLD" and self.should_execute_signal(filtered_signal, confidence):
                # Execute trade
                ticket = self.trade_executor.send_order(self.symbol, signal_str)
                
                if ticket:
                    self.last_signal = filtered_signal
                    
                    # Get trade details for logging
                    positions = self.risk_manager.get_open_positions(self.symbol)
                    latest_position = next((p for p in positions if p['ticket'] == ticket), None)
                    
                    if latest_position:
                        self.log_trade(
                            signal_str,
                            latest_position['price_open'],
                            confidence,
                            latest_position['sl'],
                            latest_position['tp'],
                            latest_position['volume']
                        )
                    
                    self.logger.info(f"🚀 Executed {signal_str} signal with confidence {confidence:.3f}")
                else:
                    self.logger.error(f"❌ Failed to execute {signal_str} signal")
            else:
                if signal_str != "HOLD":
                    self.logger.info(f"Signal {signal_str} not executed (confidence: {confidence:.3f})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in trading cycle: {e}")
            return False

    def run(self):
        """Main trading loop"""
        self.logger.info("🚀 Starting Enhanced Live Trading System")
        
        if not self.connect_mt5():
            self.logger.error("❌ Failed to connect to MT5")
            return
        
        # Log initial status
        self.risk_manager.log_status()
        
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                
                # Run trading cycle
                success = self.run_single_cycle()
                
                if not success:
                    self.logger.warning("Trading cycle failed, retrying...")
                
                # Log status every 10 cycles
                if cycle_count % 10 == 0:
                    self.risk_manager.log_status()
                    
                    # Log account status
                    account_info = self.trade_executor.get_account_info()
                    self.logger.info(f"Account Balance: {account_info['balance']:.2f}, "
                                   f"Equity: {account_info['equity']:.2f}")
                
                # Sleep between cycles
                time.sleep(15)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Trading stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Unexpected error: {e}")
                time.sleep(30)  # Wait longer on error
        
        # Cleanup
        self.trade_executor.shutdown()
        self.logger.info("✅ Trading system shutdown complete")

def main():
    """Main entry point"""
    runner = EnhancedLiveRunner(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5)
    runner.run()

if __name__ == "__main__":
    main() 