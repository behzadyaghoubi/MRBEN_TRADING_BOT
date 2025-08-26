import logging
import os
import time
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd

# Import enhanced components
from enhanced_risk_manager import EnhancedRiskManager
from enhanced_trade_executor import EnhancedTradeExecutor

# Import existing components
try:
    from lstm_live_signal_generator import generate_lstm_live_signal
except ImportError:
    from strategies.lstm_live_signal_generator import generate_lstm_live_signal

from ai_filter import AISignalFilter


class EnhancedLiveRunnerDemo:
    """
    Enhanced Live Trading Runner (Demo Mode) with:
    - Dynamic ATR-based TP/SL
    - Trailing stop management
    - Adaptive confidence thresholds
    - Comprehensive risk management
    - Demo mode for testing without MT5
    """

    def __init__(self, symbol: str = "XAUUSD", timeframe=mt5.TIMEFRAME_M5, demo_mode=True):
        self.symbol = symbol
        self.timeframe = timeframe
        self.demo_mode = demo_mode
        self.last_signal = None
        self.start_balance = 10000.0  # Demo balance
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
            confidence_adjustment_factor=0.1,
        )

        self.trade_executor = EnhancedTradeExecutor(self.risk_manager)

        # Initialize AI filter
        try:
            # Try to load from models directory first, then current directory
            model_paths = [
                "models/mrben_ai_signal_filter_xgb.joblib",
                "mrben_ai_signal_filter_xgb.joblib",
            ]

            ai_filter_loaded = False
            for model_path in model_paths:
                try:
                    self.ai_filter = AISignalFilter(
                        model_path=model_path, model_type="joblib", threshold=0.55
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

        self.logger.info("Enhanced Live Runner (Demo Mode) initialized successfully")

    def setup_logging(self):
        """Setup comprehensive logging"""
        self.logger = logging.getLogger("EnhancedLiveRunnerDemo")
        self.logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        if not os.path.exists('logs'):
            os.makedirs('logs')

        file_handler = logging.FileHandler('logs/enhanced_live_runner_demo.log', encoding='utf-8')
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def setup_trade_logging(self):
        """Setup trade logging file"""
        self.log_path = "enhanced_live_trades_demo.csv"
        if not os.path.exists(self.log_path):
            columns = [
                "time",
                "symbol",
                "signal",
                "price",
                "confidence",
                "sl",
                "tp",
                "volume",
                "demo_mode",
            ]
            pd.DataFrame(columns=columns).to_csv(self.log_path, index=False)

    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5 or use demo mode"""
        if self.demo_mode:
            self.logger.info("DEMO MODE: Running without MT5 connection")
            return True

        if not mt5.initialize():
            self.logger.error(f"MT5 connection failed: {mt5.last_error()}")
            return False

        if not mt5.symbol_select(self.symbol, True):
            self.logger.error(f"Symbol {self.symbol} selection failed")
            return False

        # Get initial balance
        account_info = mt5.account_info()
        if account_info:
            self.start_balance = account_info.balance
            self.logger.info(f"Connected to MT5. Balance: {self.start_balance}")
        else:
            self.logger.error("Failed to get account info")
            return False

        return True

    def get_price_data(self, bars: int = 100) -> pd.DataFrame | None:
        """Get price data from MT5 or use demo data"""
        if self.demo_mode:
            # Generate demo price data
            return self.generate_demo_data(bars)

        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None:
            self.logger.error("Failed to get price data")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def generate_demo_data(self, bars: int = 100) -> pd.DataFrame:
        """Generate demo price data for testing"""
        import numpy as np

        # Generate realistic demo data
        base_price = 2000.0  # Base price for XAUUSD
        np.random.seed(int(datetime.now().timestamp()) % 1000)  # Different seed each time

        # Generate time series
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=bars * 5)  # 5-minute bars

        times = pd.date_range(start=start_time, end=end_time, periods=bars)

        # Generate price data with some volatility
        price_changes = np.random.normal(0, 0.5, bars)  # Small random changes
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] + change
            prices.append(max(new_price, base_price * 0.95))  # Prevent too low prices

        # Generate OHLC data
        data = []
        for i, (timestamp, price) in enumerate(zip(times, prices, strict=False)):
            # Generate realistic OHLC from base price
            high = price + abs(np.random.normal(0, 0.3))
            low = price - abs(np.random.normal(0, 0.3))
            open_price = price + np.random.normal(0, 0.1)
            close_price = price + np.random.normal(0, 0.1)

            # Ensure OHLC relationship
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            data.append(
                {
                    'time': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'tick_volume': int(np.random.uniform(100, 1000)),
                }
            )

        return pd.DataFrame(data)

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

        # In demo mode, always allow trading
        if self.demo_mode:
            return True

        # Check if we can open new trade
        account_info = self.trade_executor.get_account_info()
        open_positions = self.risk_manager.get_open_positions(self.symbol)
        open_trades_count = len(open_positions)

        if not self.risk_manager.can_open_new_trade(
            account_info['balance'], self.start_balance, open_trades_count
        ):
            return False

        return True

    def log_trade(
        self, signal: str, price: float, confidence: float, sl: float, tp: float, volume: float
    ):
        """Log trade execution"""
        new_row = {
            "time": datetime.now(),
            "symbol": self.symbol,
            "signal": signal,
            "price": price,
            "confidence": confidence,
            "sl": sl,
            "tp": tp,
            "volume": volume,
            "demo_mode": self.demo_mode,
        }

        df = pd.read_csv(self.log_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.log_path, index=False)

    def run_single_cycle(self) -> bool:
        """Run a single trading cycle"""
        try:
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
                    # Use AI filter result as confidence (simplified)
                    confidence = 0.7  # Default confidence when AI filter is used
                except Exception as e:
                    self.logger.warning(f"AI filter failed: {e}")
                    filtered_signal = signal
                    confidence = 0.6  # Default confidence
            else:
                filtered_signal = signal
                confidence = 0.6  # Default confidence

            # Convert signal to string
            signal_str = (
                "BUY" if filtered_signal == 2 else "SELL" if filtered_signal == 0 else "HOLD"
            )

            # Check if we should execute the signal
            if signal_str != "HOLD" and self.should_execute_signal(filtered_signal, confidence):
                if self.demo_mode:
                    # Simulate trade execution in demo mode
                    current_price = df['close'].iloc[-1]
                    sl, tp = self.risk_manager.calculate_dynamic_sl_tp(
                        self.symbol, current_price, signal_str
                    )
                    volume = 0.01  # Demo volume

                    self.log_trade(signal_str, current_price, confidence, sl, tp, volume)
                    self.last_signal = filtered_signal

                    self.logger.info(
                        f"DEMO: Executed {signal_str} signal with confidence {confidence:.3f}"
                    )
                    self.logger.info(
                        f"DEMO: Price: {current_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}"
                    )
                else:
                    # Execute real trade
                    ticket = self.trade_executor.send_order(self.symbol, signal_str)

                    if ticket:
                        self.last_signal = filtered_signal

                        # Get trade details for logging
                        positions = self.risk_manager.get_open_positions(self.symbol)
                        latest_position = next(
                            (p for p in positions if p['ticket'] == ticket), None
                        )

                        if latest_position:
                            self.log_trade(
                                signal_str,
                                latest_position['price_open'],
                                confidence,
                                latest_position['sl'],
                                latest_position['tp'],
                                latest_position['volume'],
                            )

                        self.logger.info(
                            f"Executed {signal_str} signal with confidence {confidence:.3f}"
                        )
                    else:
                        self.logger.error(f"Failed to execute {signal_str} signal")
            else:
                if signal_str != "HOLD":
                    self.logger.info(
                        f"Signal {signal_str} not executed (confidence: {confidence:.3f})"
                    )

            return True

        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            return False

    def run(self):
        """Main trading loop"""
        mode_text = "DEMO MODE" if self.demo_mode else "LIVE MODE"
        self.logger.info(f"Starting Enhanced Live Trading System ({mode_text})")

        if not self.connect_mt5():
            self.logger.error("Failed to connect to MT5")
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

                    if not self.demo_mode:
                        # Log account status
                        account_info = self.trade_executor.get_account_info()
                        self.logger.info(
                            f"Account Balance: {account_info['balance']:.2f}, "
                            f"Equity: {account_info['equity']:.2f}"
                        )
                    else:
                        self.logger.info(f"DEMO: Cycle {cycle_count} completed")

                # Sleep between cycles
                time.sleep(15)

            except KeyboardInterrupt:
                self.logger.info("Trading stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                time.sleep(30)  # Wait longer on error

        # Cleanup
        if not self.demo_mode:
            self.trade_executor.shutdown()
        self.logger.info("Trading system shutdown complete")


def main():
    """Main entry point"""
    # Run in demo mode first
    runner = EnhancedLiveRunnerDemo(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, demo_mode=True)
    runner.run()


if __name__ == "__main__":
    main()
