"""
Main Trading Engine for MR BEN Trading Bot.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import MetaTrader5 as mt5
import pandas as pd

from ai_filter import AISignalFilter
from config.settings import settings
from core.database import db_manager
from core.logger import get_logger, logger
from strategies import BookStrategy

from .position_manager import PositionManager
from .risk_manager import risk_manager
from .trade_executor import TradeExecutor

logger = get_logger("trading.engine")


@dataclass
class TradeSignal:
    """Trade signal data structure."""

    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    timestamp: datetime
    strategy: str
    ai_decision: bool
    ai_confidence: float
    features: dict[str, Any]
    metadata: dict[str, Any]


class TradingEngine:
    """
    Main trading engine that orchestrates all components:
    - Market data collection
    - Signal generation
    - AI filtering
    - Risk management
    - Trade execution
    - Position management
    """

    def __init__(self):
        """Initialize the trading engine."""
        self.logger = get_logger("trading_engine")
        self.executor = TradeExecutor()
        self.position_manager = PositionManager()
        self.strategy = BookStrategy()

        # Trading state
        self.is_running = False
        self.last_signal_time = None
        self.signal_cooldown = 300  # 5 minutes between signals

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0

        self.logger.info("Trading Engine initialized")

    def start(self) -> bool:
        """Start the trading engine."""
        try:
            # Validate settings
            if not settings.validate_settings():
                self.logger.error("Invalid settings configuration")
                return False

            # Initialize MT5 connection
            if not self._connect_mt5():
                return False

            # Initialize components
            if not self._initialize_components():
                return False

            self.is_running = True
            self.logger.info("Trading Engine started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start trading engine: {e}")
            return False

    def stop(self) -> None:
        """Stop the trading engine."""
        self.is_running = False
        self.logger.info("Trading Engine stopped")

        # Close MT5 connection
        if mt5.terminal_info():
            mt5.shutdown()

    def run_single_cycle(self) -> bool:
        """Run a single trading cycle."""
        if not self.is_running:
            return False

        try:
            # Check if enough time has passed since last signal
            if self._is_signal_cooldown_active():
                return True

            # Get market data
            market_data = self._get_market_data()
            if market_data is None or len(market_data) < 50:
                self.logger.warning("Insufficient market data")
                return True

            # Generate signals
            signal_result = self.strategy.get_latest_signal(market_data)

            # Skip if no signal
            if signal_result.signal == 'HOLD':
                return True

            # AI filtering
            ai_filter_instance = AISignalFilter()
            ai_decision = ai_filter_instance.filter_signal(signal_result.features)
            ai_confidence = ai_filter_instance.filter_signal(signal_result.features, as_label=False)

            # Skip if AI rejects signal
            if not ai_decision:
                self.logger.info(f"Signal rejected by AI: {signal_result.signal}")
                return True

            # Get current market prices
            current_price = self._get_current_price(signal_result.signal)
            if current_price is None:
                return True

            # Calculate trade parameters
            trade_signal = self._prepare_trade_signal(
                signal_result, current_price, ai_decision, ai_confidence
            )

            if trade_signal is None:
                return True

            # Execute trade
            success = self._execute_trade(trade_signal)

            if success:
                self.last_signal_time = datetime.now()
                self.logger.info(
                    f"Trade executed successfully: {trade_signal.action} {trade_signal.symbol}"
                )

            return True

        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            return False

    def run_continuous(self, interval_seconds: int = 60) -> None:
        """Run the trading engine continuously."""
        self.logger.info(f"Starting continuous trading with {interval_seconds}s interval")

        while self.is_running:
            try:
                self.run_single_cycle()
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous trading: {e}")
                time.sleep(interval_seconds)

        self.stop()

    def _connect_mt5(self) -> bool:
        """Connect to MetaTrader 5."""
        try:
            if not mt5.initialize(
                login=settings.mt5.login, password=settings.mt5.password, server=settings.mt5.server
            ):
                self.logger.error(f"MT5 connection failed: {mt5.last_error()}")
                return False

            # Check symbol availability
            symbol_info = mt5.symbol_info(settings.trading.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {settings.trading.symbol} not found")
                return False

            self.logger.info(f"Connected to MT5: {settings.mt5.server}")
            return True

        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False

    def _initialize_components(self) -> bool:
        """Initialize trading components."""
        try:
            # Initialize position manager
            self.position_manager.initialize()

            # Test AI filter
            test_features = [100, 50, 1, 2000]
            ai_filter_instance = AISignalFilter()
            ai_result = ai_filter_instance.filter_signal(test_features)
            self.logger.info(f"AI filter test: {ai_result}")

            return True

        except Exception as e:
            self.logger.error(f"Component initialization error: {e}")
            return False

    def _get_market_data(self) -> pd.DataFrame | None:
        """Get market data from MT5."""
        try:
            timeframe = settings.get_timeframe_mt5()
            symbol = settings.trading.symbol

            # Get recent data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 200)
            if rates is None or len(rates) < 50:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            return df

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None

    def _get_current_price(self, signal_type: str) -> float | None:
        """Get current market price."""
        try:
            tick = mt5.symbol_info_tick(settings.trading.symbol)
            if tick is None:
                return None

            if signal_type == 'BUY':
                return tick.ask
            else:
                return tick.bid

        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None

    def _prepare_trade_signal(
        self, signal_result, current_price: float, ai_decision: bool, ai_confidence: float
    ) -> TradeSignal | None:
        """Prepare trade signal with all parameters."""
        try:
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return None

            balance = account_info.balance
            open_positions = self.position_manager.get_open_positions(settings.trading.symbol)

            # Calculate stop loss and take profit
            atr = signal_result.features.get('atr', 0.001)
            stop_loss = risk_manager.calculate_stop_loss(current_price, signal_result.signal, atr)
            take_profit = risk_manager.calculate_take_profit(
                current_price, signal_result.signal, stop_loss
            )

            # Calculate lot size
            stop_loss_pips = abs(current_price - stop_loss) / 0.0001  # Approximate
            pip_value = 1.0  # Simplified

            lot_size = risk_manager.calc_lot_size(
                balance,
                stop_loss_pips,
                pip_value,
                len(open_positions),
                settings.trading.start_balance,
                settings.trading.symbol,
            )

            if lot_size <= 0:
                self.logger.warning("Lot size calculation returned 0 or negative")
                return None

            # Validate trade parameters
            is_valid, error_msg = risk_manager.validate_trade_parameters(
                settings.trading.symbol, lot_size, stop_loss_pips, balance
            )

            if not is_valid:
                self.logger.warning(f"Trade validation failed: {error_msg}")
                return None

            return TradeSignal(
                symbol=settings.trading.symbol,
                action=signal_result.signal,
                confidence=signal_result.confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=lot_size,
                timestamp=datetime.now(),
                strategy=self.strategy.name,
                ai_decision=ai_decision,
                ai_confidence=ai_confidence,
                features=signal_result.features,
                metadata=signal_result.metadata,
            )

        except Exception as e:
            self.logger.error(f"Error preparing trade signal: {e}")
            return None

    def _execute_trade(self, trade_signal: TradeSignal) -> bool:
        """Execute the trade."""
        try:
            # Execute trade
            result = self.executor.execute_trade(trade_signal)

            if result.success:
                # Save to database
                trade_data = {
                    'timestamp': trade_signal.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': trade_signal.symbol,
                    'action': trade_signal.action,
                    'entry_price': trade_signal.entry_price,
                    'sl': trade_signal.stop_loss,
                    'tp': trade_signal.take_profit,
                    'lot_size': trade_signal.lot_size,
                    'balance': mt5.account_info().balance,
                    'ai_decision': int(trade_signal.ai_decision),
                    'ai_confidence': trade_signal.ai_confidence,
                    'result_code': result.retcode,
                    'comment': result.comment,
                    'status': 'open',
                }

                trade_id = db_manager.save_trade(trade_data)
                self.logger.info(f"Trade saved to database with ID: {trade_id}")

                # Update position manager
                self.position_manager.add_position(trade_signal)

                return True
            else:
                self.logger.error(f"Trade execution failed: {result.comment}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False

    def _is_signal_cooldown_active(self) -> bool:
        """Check if signal cooldown is active."""
        if self.last_signal_time is None:
            return False

        time_since_last = datetime.now() - self.last_signal_time
        return time_since_last.total_seconds() < self.signal_cooldown

    def get_performance_summary(self) -> dict[str, Any]:
        """Get trading performance summary."""
        try:
            # Get trades from database
            trades_df = db_manager.get_trades()

            if trades_df.empty:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0,
                    'avg_profit': 0.0,
                }

            # Calculate metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            losing_trades = len(trades_df[trades_df['profit'] < 0])
            total_profit = trades_df['profit'].sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_profit = total_profit / total_trades if total_trades > 0 else 0

            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'current_balance': mt5.account_info().balance if mt5.account_info() else 0,
            }

        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}

    def get_status(self) -> dict[str, Any]:
        """Get current trading engine status."""
        return {
            'is_running': self.is_running,
            'symbol': settings.trading.symbol,
            'timeframe': settings.trading.timeframe,
            'last_signal_time': (
                self.last_signal_time.isoformat() if self.last_signal_time else None
            ),
            'open_positions': len(
                self.position_manager.get_open_positions(settings.trading.symbol)
            ),
            'ai_model_loaded': AISignalFilter().model_loaded,
            'performance': self.get_performance_summary(),
        }


# Global trading engine instance
trading_engine = TradingEngine()
