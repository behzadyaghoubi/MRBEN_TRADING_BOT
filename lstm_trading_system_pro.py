#!/usr/bin/env python3
"""
MRBEN LSTM Trading System Pro - Complete Professional Trading System
===================================================================

Complete LSTM-based trading system with:
- Professional data preparation and labeling
- Optimized LSTM model training
- Balanced signal generation with intelligent thresholds
- Advanced backtesting with risk management
- Comprehensive analysis and reporting

Author: MRBEN Trading System
Version: 4.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import talib
import logging
from datetime import datetime
import warnings
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

warnings.filterwarnings('ignore')

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Complete trading system configuration"""
    # Data parameters
    lookback_period: int = 60
    prediction_horizon: int = 5
    
    # LSTM parameters
    lstm_units: int = 100
    lstm_layers: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    # Signal generation parameters
    buy_threshold: float = 0.05  # Reduced for more BUY signals
    sell_threshold: float = 0.05  # Reduced for more SELL signals
    hold_threshold: float = 0.95  # Increased to reduce HOLD signals
    signal_amplification: float = 2.0  # Increased for stronger signals
    
    # Trading parameters
    stop_loss_pips: int = 30
    take_profit_pips: int = 60
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_open_trades: int = 3
    
    # Backtesting parameters
    initial_balance: float = 10000
    commission: float = 0.0001  # 1 pip commission
    
    def save_config(self, filename: str = "trading_config.json"):
        """Save configuration to JSON file"""
        config_dict = {
            'lookback_period': self.lookback_period,
            'prediction_horizon': self.prediction_horizon,
            'lstm_units': self.lstm_units,
            'lstm_layers': self.lstm_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'hold_threshold': self.hold_threshold,
            'signal_amplification': self.signal_amplification,
            'stop_loss_pips': self.stop_loss_pips,
            'take_profit_pips': self.take_profit_pips,
            'risk_per_trade': self.risk_per_trade,
            'max_open_trades': self.max_open_trades,
            'initial_balance': self.initial_balance,
            'commission': self.commission
        }
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filename}")
    
    @classmethod
    def load_config(cls, filename: str = "trading_config.json"):
        """Load configuration from JSON file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        else:
            logger.warning(f"Config file {filename} not found, using defaults")
            return cls()

class DataPreprocessor:
    """Professional data preprocessing and feature engineering"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features for LSTM model"""
        logger.info("Starting professional data preparation...")
        
        # Create copy to avoid modifying original data
        data = df.copy()
        
        # Calculate technical indicators
        data = self._calculate_indicators(data)
        
        # Create price-based features
        data = self._create_price_features(data)
        
        # Create volatility features
        data = self._create_volatility_features(data)
        
        # Create momentum features
        data = self._create_momentum_features(data)
        
        # Remove NaN values
        data = data.dropna().reset_index(drop=True)
        
        logger.info(f"Data preparation completed. Final shape: {data.shape}")
        return data
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['EMA_12'] = talib.EMA(df['close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # RSI
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_hist'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
        df['BB_upper'] = bb_upper
        df['BB_middle'] = bb_middle
        df['BB_lower'] = bb_lower
        df['BB_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'])
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
        
        # ATR
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
        # Price position relative to moving averages
        df['price_vs_sma20'] = (df['close'] - df['SMA_20']) / df['SMA_20']
        df['price_vs_sma50'] = (df['close'] - df['SMA_50']) / df['SMA_50']
        df['price_vs_ema12'] = (df['close'] - df['EMA_12']) / df['EMA_12']
        
        # Price position in Bollinger Bands
        df['bb_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        # Rolling volatility
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_10'] = df['price_change'].rolling(10).std()
        df['volatility_20'] = df['price_change'].rolling(20).std()
        
        # ATR-based features
        df['atr_ratio'] = df['ATR'] / df['close']
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features"""
        # RSI-based features
        df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
        df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        
        # MACD-based features
        df['macd_cross'] = ((df['MACD'] > df['MACD_signal']) & 
                           (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['MACD'] < df['MACD_signal']) & 
                                (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))).astype(int)
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create LSTM sequences and labels"""
        logger.info("Creating LSTM sequences...")
        
        # Select features for LSTM
        feature_columns = [
            'open', 'high', 'low', 'close', 'tick_volume',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_width', 'Stoch_K', 'Stoch_D', 'ATR',
            'price_change', 'price_vs_sma20', 'price_vs_sma50',
            'bb_position', 'volatility_10', 'atr_ratio'
        ]
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        data = df[available_columns].values
        
        # Scale the data
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.config.lookback_period, len(data_scaled) - self.config.prediction_horizon):
            X.append(data_scaled[i-self.config.lookback_period:i])
            
            # Create labels based on future price movement
            future_return = (df['close'].iloc[i + self.config.prediction_horizon] - df['close'].iloc[i]) / df['close'].iloc[i]
            
            if future_return > 0.005:  # 0.5% gain
                y.append(2)  # BUY (class 2)
            elif future_return < -0.005:  # 0.5% loss
                y.append(0)  # SELL (class 0)
            else:
                y.append(1)  # HOLD (class 1)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
        return X, y

class LSTMModel:
    """Professional LSTM model for trading signals"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture"""
        logger.info("Building LSTM model...")
        
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(keras.layers.LSTM(
            units=self.config.lstm_units,
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(keras.layers.Dropout(self.config.dropout_rate))
        
        # Additional LSTM layers
        for i in range(self.config.lstm_layers - 1):
            model.add(keras.layers.LSTM(
                units=self.config.lstm_units,
                return_sequences=(i < self.config.lstm_layers - 2)
            ))
            model.add(keras.layers.Dropout(self.config.dropout_rate))
        
        # Dense layers
        model.add(keras.layers.Dense(50, activation='relu'))
        model.add(keras.layers.Dropout(self.config.dropout_rate))
        model.add(keras.layers.Dense(25, activation='relu'))
        model.add(keras.layers.Dense(3, activation='softmax'))  # 3 classes: BUY, HOLD, SELL
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("LSTM model built successfully")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> keras.Model:
        """Train the LSTM model"""
        logger.info("Starting LSTM model training...")
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info("LSTM model training completed")
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save_model(self, filename: str = "lstm_trading_model.h5"):
        """Save trained model"""
        if self.model is not None:
            self.model.save(filename)
            logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename: str = "lstm_trading_model.h5"):
        """Load trained model"""
        if os.path.exists(filename):
            self.model = keras.models.load_model(filename)
            logger.info(f"Model loaded from {filename}")
        else:
            logger.warning(f"Model file {filename} not found")

class SignalGenerator:
    """Professional signal generation with balanced distribution"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.signal_map = {2: "BUY", 1: "HOLD", 0: "SELL"}
    
    def generate_signals(self, df: pd.DataFrame, model: keras.Model, 
                        preprocessor: DataPreprocessor) -> pd.DataFrame:
        """Generate balanced trading signals"""
        logger.info("Generating professional trading signals...")
        
        # Prepare data
        data = preprocessor.prepare_data(df)
        
        # Create sequences for prediction
        X, _ = preprocessor.create_sequences(data)
        
        # Generate predictions
        predictions = model.predict(X)
        
        # Create result DataFrame
        result_df = data.iloc[preprocessor.config.lookback_period:].copy()
        result_df = result_df.iloc[:len(predictions)].copy()
        
        # Add prediction probabilities
        result_df['lstm_buy_proba'] = predictions[:, 2]  # BUY probability (class 2)
        result_df['lstm_hold_proba'] = predictions[:, 1]  # HOLD probability (class 1)
        result_df['lstm_sell_proba'] = predictions[:, 0]  # SELL probability (class 0)
        
        # Generate balanced signals
        result_df['signal'] = 0  # Default to HOLD
        result_df['signal_confidence'] = 0.0
        result_df['signal_reason'] = 'HOLD'
        
        for idx, row in result_df.iterrows():
            buy_prob = row['lstm_buy_proba']
            hold_prob = row['lstm_hold_proba']
            sell_prob = row['lstm_sell_proba']
            
            signal, confidence, reason = self._generate_single_signal(buy_prob, hold_prob, sell_prob)
            
            result_df.at[idx, 'signal'] = signal
            result_df.at[idx, 'signal_confidence'] = confidence
            result_df.at[idx, 'signal_reason'] = reason
        
        # Add signal labels
        result_df['signal_label'] = result_df['signal'].map(self.signal_map)
        
        logger.info("Signal generation completed")
        return result_df
    
    def _generate_single_signal(self, buy_prob: float, hold_prob: float, sell_prob: float) -> Tuple[int, float, str]:
        """Generate single signal using professional algorithm"""
        # Normalize probabilities
        total_prob = buy_prob + hold_prob + sell_prob
        if total_prob > 0:
            buy_prob /= total_prob
            hold_prob /= total_prob
            sell_prob /= total_prob
        
        # Amplify BUY/SELL probabilities
        amplified_buy = buy_prob * self.config.signal_amplification
        amplified_sell = sell_prob * self.config.signal_amplification
        
        # ULTRA-AGGRESSIVE SIGNAL GENERATION FOR MORE TRADES
        
        # 1. Ultra-low thresholds for immediate BUY/SELL
        if amplified_buy >= self.config.buy_threshold:
            confidence = min(amplified_buy + 0.1, 1.0)
            return 2, confidence, f"BUY_ULTRA_{buy_prob:.3f}"
        
        if amplified_sell >= self.config.sell_threshold:
            confidence = min(amplified_sell + 0.1, 1.0)
            return 0, confidence, f"SELL_ULTRA_{sell_prob:.3f}"
        
        # 2. Relative strength with very low thresholds
        if buy_prob > sell_prob and buy_prob >= 0.005:  # Reduced from 0.01
            return 2, buy_prob, f"BUY_RELATIVE_{buy_prob:.3f}"
        
        if sell_prob > buy_prob and sell_prob >= 0.005:  # Reduced from 0.01
            return 0, sell_prob, f"SELL_RELATIVE_{sell_prob:.3f}"
        
        # 3. Force BUY/SELL when HOLD is not extremely dominant
        if hold_prob < 0.9:  # Reduced from 0.8
            if buy_prob > sell_prob and buy_prob >= 0.01:  # Reduced from 0.03
                return 2, buy_prob, f"BUY_FORCED_{buy_prob:.3f}"
            elif sell_prob > buy_prob and sell_prob >= 0.01:  # Reduced from 0.03
                return 0, sell_prob, f"SELL_FORCED_{sell_prob:.3f}"
        
        # 4. Last resort with very low thresholds
        if buy_prob > sell_prob and buy_prob >= 0.005:  # Reduced from 0.02
            return 2, buy_prob, f"BUY_LAST_RESORT_{buy_prob:.3f}"
        elif sell_prob > buy_prob and sell_prob >= 0.005:  # Reduced from 0.02
            return 0, sell_prob, f"SELL_LAST_RESORT_{sell_prob:.3f}"
        
        # 5. Only HOLD when absolutely necessary
        if hold_prob >= self.config.hold_threshold:
            return 1, hold_prob, f"HOLD_STRONG_{hold_prob:.3f}"
        
        # 6. Final fallback: choose any non-zero signal
        if buy_prob > 0:
            return 2, buy_prob, f"BUY_FALLBACK_{buy_prob:.3f}"
        elif sell_prob > 0:
            return 0, sell_prob, f"SELL_FALLBACK_{sell_prob:.3f}"
        
        return 1, hold_prob, f"HOLD_DEFAULT_{hold_prob:.3f}"

class Backtester:
    """Professional backtesting system with risk management"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        logger.info("Starting professional backtest...")
        
        # Initialize variables
        balance = self.config.initial_balance
        equity = balance
        open_trades = []
        
        # Process each signal
        for idx, row in df.iterrows():
            current_price = row['close']
            signal = row['signal']
            confidence = row['signal_confidence']
            
            # Update open trades
            open_trades = self._update_open_trades(open_trades, current_price, balance)
            
            # Check if we can open new trades
            if len(open_trades) < self.config.max_open_trades and signal != 0:
                # Calculate position size based on risk
                position_size = self._calculate_position_size(balance, current_price)
                
                if position_size > 0:
                    # Open new trade
                    trade = {
                        'entry_time': row.name,
                        'entry_price': current_price,
                        'signal': signal,
                        'confidence': confidence,
                        'position_size': position_size,
                        'stop_loss': self._calculate_stop_loss(current_price, signal),
                        'take_profit': self._calculate_take_profit(current_price, signal),
                        'status': 'open'
                    }
                    open_trades.append(trade)
            
            # Update equity
            equity = balance + sum(trade.get('unrealized_pnl', 0) for trade in open_trades)
            self.equity_curve.append({
                'time': row.name,
                'equity': equity,
                'balance': balance,
                'open_trades': len(open_trades)
            })
        
        # Close remaining open trades
        for trade in open_trades:
            trade['exit_time'] = df.index[-1]
            trade['exit_price'] = df['close'].iloc[-1]
            trade['status'] = 'closed'
            trade['realized_pnl'] = self._calculate_pnl(trade)
            balance += trade['realized_pnl']
            self.trades.append(trade)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics()
        
        logger.info("Backtest completed")
        return performance
    
    def _update_open_trades(self, open_trades: List[Dict], current_price: float, balance: float) -> List[Dict]:
        """Update open trades and close if needed"""
        updated_trades = []
        
        for trade in open_trades:
            # Check stop loss
            if trade['signal'] == 2 and current_price <= trade['stop_loss']:
                trade['exit_price'] = trade['stop_loss']
                trade['status'] = 'closed'
                trade['realized_pnl'] = self._calculate_pnl(trade)
                balance += trade['realized_pnl']
                self.trades.append(trade)
                continue
            
            if trade['signal'] == 1 and current_price >= trade['stop_loss']:
                trade['exit_price'] = trade['stop_loss']
                trade['status'] = 'closed'
                trade['realized_pnl'] = self._calculate_pnl(trade)
                balance += trade['realized_pnl']
                self.trades.append(trade)
                continue
            
            # Check take profit
            if trade['signal'] == 2 and current_price >= trade['take_profit']:
                trade['exit_price'] = trade['take_profit']
                trade['status'] = 'closed'
                trade['realized_pnl'] = self._calculate_pnl(trade)
                balance += trade['realized_pnl']
                self.trades.append(trade)
                continue
            
            if trade['signal'] == 1 and current_price <= trade['take_profit']:
                trade['exit_price'] = trade['take_profit']
                trade['status'] = 'closed'
                trade['realized_pnl'] = self._calculate_pnl(trade)
                balance += trade['realized_pnl']
                self.trades.append(trade)
                continue
            
            # Update unrealized PnL
            trade['unrealized_pnl'] = self._calculate_unrealized_pnl(trade, current_price)
            updated_trades.append(trade)
        
        return updated_trades
    
    def _calculate_position_size(self, balance: float, current_price: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = balance * self.config.risk_per_trade
        pip_value = 0.1  # For XAUUSD
        stop_loss_pips = self.config.stop_loss_pips
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return min(position_size, balance * 0.1)  # Max 10% of balance
    
    def _calculate_stop_loss(self, entry_price: float, signal: int) -> float:
        """Calculate stop loss price"""
        pip_value = 0.1
        if signal == 2:  # BUY
            return entry_price - (self.config.stop_loss_pips * pip_value)
        else:  # SELL
            return entry_price + (self.config.stop_loss_pips * pip_value)
    
    def _calculate_take_profit(self, entry_price: float, signal: int) -> float:
        """Calculate take profit price"""
        pip_value = 0.1
        if signal == 2:  # BUY
            return entry_price + (self.config.take_profit_pips * pip_value)
        else:  # SELL
            return entry_price - (self.config.take_profit_pips * pip_value)
    
    def _calculate_pnl(self, trade: Dict) -> float:
        """Calculate realized PnL for a trade"""
        if trade['signal'] == 2:  # BUY
            pnl = (trade['exit_price'] - trade['entry_price']) * trade['position_size']
        else:  # SELL
            pnl = (trade['entry_price'] - trade['exit_price']) * trade['position_size']
        
        # Subtract commission
        pnl -= self.config.commission * trade['position_size']
        return pnl
    
    def _calculate_unrealized_pnl(self, trade: Dict, current_price: float) -> float:
        """Calculate unrealized PnL for an open trade"""
        if trade['signal'] == 2:  # BUY
            pnl = (current_price - trade['entry_price']) * trade['position_size']
        else:  # SELL
            pnl = (trade['entry_price'] - current_price) * trade['position_size']
        
        return pnl
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['realized_pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['realized_pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(t['realized_pnl'] for t in self.trades)
        avg_win = np.mean([t['realized_pnl'] for t in self.trades if t['realized_pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['realized_pnl'] for t in self.trades if t['realized_pnl'] < 0]) if losing_trades > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown()
        
        # Return metrics
        initial_balance = self.config.initial_balance
        final_balance = initial_balance + total_pnl
        total_return = (final_balance - initial_balance) / initial_balance
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf'),
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'final_balance': final_balance,
            'equity_curve': self.equity_curve
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.equity_curve:
            return 0
        
        equity_values = [point['equity'] for point in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd

class ReportGenerator:
    """Professional report generation and visualization"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def generate_comprehensive_report(self, signals_df: pd.DataFrame, 
                                    performance: Dict[str, Any],
                                    save_path: str = "trading_report") -> None:
        """Generate comprehensive trading report"""
        logger.info("Generating comprehensive trading report...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Signal Distribution
        ax1 = plt.subplot(3, 3, 1)
        self._plot_signal_distribution(signals_df, ax1)
        
        # 2. Equity Curve
        ax2 = plt.subplot(3, 3, 2)
        self._plot_equity_curve(performance, ax2)
        
        # 3. Signal Confidence Distribution
        ax3 = plt.subplot(3, 3, 3)
        self._plot_confidence_distribution(signals_df, ax3)
        
        # 4. Performance Metrics
        ax4 = plt.subplot(3, 3, 4)
        self._plot_performance_metrics(performance, ax4)
        
        # 5. Signal Over Time
        ax5 = plt.subplot(3, 3, 5)
        self._plot_signals_over_time(signals_df, ax5)
        
        # 6. LSTM Probabilities
        ax6 = plt.subplot(3, 3, 6)
        self._plot_lstm_probabilities(signals_df, ax6)
        
        # 7. Trade Analysis
        ax7 = plt.subplot(3, 3, 7)
        self._plot_trade_analysis(performance, ax7)
        
        # 8. Risk Analysis
        ax8 = plt.subplot(3, 3, 8)
        self._plot_risk_analysis(performance, ax8)
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        self._plot_summary_statistics(signals_df, performance, ax9)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed report
        self._save_detailed_report(signals_df, performance, f"{save_path}.txt")
        
        logger.info(f"Comprehensive report saved to {save_path}")
    
    def _plot_signal_distribution(self, df: pd.DataFrame, ax):
        """Plot signal distribution"""
        # نگاشت سیگنال عددی به نام
        signal_map = {2: "BUY", 1: "HOLD", 0: "SELL"}
        # اگر سیگنال‌ها به صورت عددی هستند، نگاشت کن
        signals = df['signal'].map(signal_map) if np.issubdtype(df['signal'].dtype, np.number) else df['signal']
        # شمارش و اطمینان از وجود هر سه کلاس
        signal_counts = signals.value_counts().reindex(["BUY", "HOLD", "SELL"], fill_value=0)
        labels = signal_counts.index.tolist()
        colors = ["green", "gray", "red"]  # BUY, HOLD, SELL
        ax.pie(signal_counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
        ax.set_title('Signal Distribution')
    
    def _plot_equity_curve(self, performance: Dict[str, Any], ax):
        """Plot equity curve"""
        if 'equity_curve' in performance:
            equity_data = performance['equity_curve']
            times = [point['time'] for point in equity_data]
            equity = [point['equity'] for point in equity_data]
            
            ax.plot(times, equity, linewidth=2)
            ax.set_title('Equity Curve')
            ax.set_ylabel('Equity')
            ax.grid(True, alpha=0.3)
    
    def _plot_confidence_distribution(self, df: pd.DataFrame, ax):
        """Plot signal confidence distribution"""
        ax.hist(df['signal_confidence'], bins=20, alpha=0.7, color='blue')
        ax.set_title('Signal Confidence Distribution')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
    
    def _plot_performance_metrics(self, performance: Dict[str, Any], ax):
        """Plot performance metrics"""
        metrics = ['Win Rate', 'Total Return', 'Max Drawdown']
        values = [
            performance.get('win_rate', 0) * 100,
            performance.get('total_return', 0) * 100,
            performance.get('max_drawdown', 0) * 100
        ]
        
        bars = ax.bar(metrics, values, color=['green', 'blue', 'red'])
        ax.set_title('Performance Metrics')
        ax.set_ylabel('Percentage (%)')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                   f'{value:.1f}%', ha='center', va='bottom')
    
    def _plot_signals_over_time(self, df: pd.DataFrame, ax):
        """Plot signals over time"""
        # Sample every 10th point for clarity
        sample_df = df.iloc[::10]
        
        colors = {1: 'green', 0: 'gray', -1: 'red'}
        for signal in [1, 0, -1]:
            signal_data = sample_df[sample_df['signal'] == signal]
            if not signal_data.empty:
                ax.scatter(signal_data.index, signal_data['close'], 
                          c=colors[signal], alpha=0.6, s=20, label=f'Signal {signal}')
        
        ax.set_title('Signals Over Time')
        ax.set_ylabel('Price')
        ax.legend()
    
    def _plot_lstm_probabilities(self, df: pd.DataFrame, ax):
        """Plot LSTM probabilities"""
        # Sample every 20th point for clarity
        sample_df = df.iloc[::20]
        
        ax.plot(sample_df.index, sample_df['lstm_buy_proba'], label='BUY', color='green')
        ax.plot(sample_df.index, sample_df['lstm_hold_proba'], label='HOLD', color='gray')
        ax.plot(sample_df.index, sample_df['lstm_sell_proba'], label='SELL', color='red')
        
        ax.set_title('LSTM Probabilities')
        ax.set_ylabel('Probability')
        ax.legend()
    
    def _plot_trade_analysis(self, performance: Dict[str, Any], ax):
        """Plot trade analysis"""
        if 'total_trades' in performance and performance['total_trades'] > 0:
            labels = ['Winning', 'Losing']
            sizes = [performance['winning_trades'], performance['losing_trades']]
            colors = ['green', 'red']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title('Trade Analysis')
    
    def _plot_risk_analysis(self, performance: Dict[str, Any], ax):
        """Plot risk analysis"""
        if 'avg_win' in performance and 'avg_loss' in performance:
            metrics = ['Avg Win', 'Avg Loss']
            values = [performance['avg_win'], abs(performance['avg_loss'])]
            colors = ['green', 'red']
            
            bars = ax.bar(metrics, values, color=colors)
            ax.set_title('Risk Analysis')
            ax.set_ylabel('PnL')
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
    
    def _plot_summary_statistics(self, df: pd.DataFrame, performance: Dict[str, Any], ax):
        """Plot summary statistics"""
        ax.axis('off')
        
        # Create summary text
        summary_text = f"""
        TRADING SYSTEM SUMMARY
        
        Data Points: {len(df):,}
        Total Trades: {performance.get('total_trades', 0):,}
        Win Rate: {performance.get('win_rate', 0)*100:.1f}%
        Total Return: {performance.get('total_return', 0)*100:.1f}%
        Max Drawdown: {performance.get('max_drawdown', 0)*100:.1f}%
        Final Balance: ${performance.get('final_balance', 0):,.2f}
        
        Signal Distribution:
        BUY: {(df['signal'] == 2).sum()} ({(df['signal'] == 2).mean()*100:.1f}%)
        HOLD: {(df['signal'] == 1).sum()} ({(df['signal'] == 1).mean()*100:.1f}%)
        SELL: {(df['signal'] == 1).sum()} ({(df['signal'] == 1).mean()*100:.1f}%)
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

class LSTMTradingSystem:
    """Complete LSTM trading system"""
    
    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.preprocessor = DataPreprocessor(self.config)
        self.model = LSTMModel(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.backtester = Backtester(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        # Create output directory
        os.makedirs('outputs', exist_ok=True)
        
        logger.info("=== LSTM Trading System Pro Initialized ===")
    
    def run_complete_system(self, data_file: str = "lstm_signals_pro.csv") -> Dict[str, Any]:
        """Run complete trading system"""
        logger.info("Starting complete LSTM trading system...")
        
        # Load data
        df = pd.read_csv(data_file)
        logger.info(f"Loaded data: {len(df)} rows")
        
        # Prepare data
        prepared_data = self.preprocessor.prepare_data(df)
        
        # Create sequences
        X, y = self.preprocessor.create_sequences(prepared_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.train(X_train, y_train, X_test, y_test)
        
        # Generate signals
        signals_df = self.signal_generator.generate_signals(df, self.model.model, self.preprocessor)
        
        # Run backtest
        performance = self.backtester.run_backtest(signals_df)
        
        # Generate report
        self.report_generator.generate_comprehensive_report(
            signals_df, performance, "outputs/trading_report"
        )
        
        # Save results
        signals_df.to_csv("outputs/signals_with_predictions.csv", index=False)
        self.model.save_model("outputs/lstm_trading_model.h5")
        self.config.save_config("outputs/trading_config.json")
        
        # Save performance summary
        with open("outputs/performance_summary.json", 'w') as f:
            json.dump(performance, f, indent=2, default=str)
        
        logger.info("Complete LSTM trading system finished successfully!")
        
        return {
            'signals': signals_df,
            'performance': performance,
            'model': self.model
        }

def main():
    """Main execution function"""
    logger.info("=== MRBEN LSTM Trading System Pro ===")
    
    # Load or create configuration
    config = TradingConfig.load_config()
    
    # Create and run trading system
    trading_system = LSTMTradingSystem(config)
    results = trading_system.run_complete_system()
    
    # Print summary
    performance = results['performance']
    print("\n" + "="*80)
    print("TRADING SYSTEM RESULTS SUMMARY")
    print("="*80)
    print(f"Total Trades: {performance.get('total_trades', 0):,}")
    print(f"Win Rate: {performance.get('win_rate', 0)*100:.1f}%")
    print(f"Total Return: {performance.get('total_return', 0)*100:.1f}%")
    print(f"Max Drawdown: {performance.get('max_drawdown', 0)*100:.1f}%")
    print(f"Final Balance: ${performance.get('final_balance', 0):,.2f}")
    print(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
    print("="*80)
    print("All outputs saved to 'outputs/' directory")
    print("Check 'outputs/trading_report.png' for comprehensive analysis")

if __name__ == "__main__":
    main() 