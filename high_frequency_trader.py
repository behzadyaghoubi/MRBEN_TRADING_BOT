#!/usr/bin/env python3
"""
MRBEN High-Frequency LSTM Trading System
========================================

Ultra-aggressive trading system designed to generate maximum number of trades
with balanced BUY/SELL signals instead of mostly HOLD signals.

This system uses:
- Ultra-low thresholds for signal generation
- High-frequency signal processing
- Aggressive position sizing
- Multiple signal confirmation levels
- Dynamic threshold adjustment

Author: MRBEN Trading System
Version: 5.0 - High Frequency Edition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import talib
import logging
from datetime import datetime
import warnings
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('high_frequency_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HighFrequencyConfig:
    """High-frequency trading configuration"""
    # Ultra-aggressive signal parameters
    buy_threshold: float = 0.01      # Very low for maximum BUY signals
    sell_threshold: float = 0.01     # Very low for maximum SELL signals
    hold_threshold: float = 0.98     # Very high to minimize HOLD
    signal_amplification: float = 3.0 # High amplification
    
    # Trading parameters
    stop_loss_pips: int = 20         # Tighter stop loss
    take_profit_pips: int = 40       # Tighter take profit
    risk_per_trade: float = 0.01     # Lower risk per trade
    max_open_trades: int = 5         # More concurrent trades
    
    # Backtesting parameters
    initial_balance: float = 10000
    commission: float = 0.0001
    
    # Signal generation modes
    ultra_aggressive: bool = True
    force_signals: bool = True
    use_momentum: bool = True

class HighFrequencySignalGenerator:
    """Ultra-aggressive signal generator for maximum trades"""
    
    def __init__(self, config: HighFrequencyConfig):
        self.config = config
        self.signal_map = {2: "BUY", 1: "HOLD", 0: "SELL"}
    
    def generate_ultra_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ultra-aggressive signals for maximum trades"""
        logger.info("Generating ultra-aggressive signals...")
        
        # Create result DataFrame
        result_df = df.copy()
        
        # Add LSTM probabilities (assuming they exist)
        if 'lstm_buy_proba' not in result_df.columns:
            # Generate mock probabilities for demonstration
            result_df['lstm_buy_proba'] = np.random.uniform(0, 0.4, len(result_df))
            result_df['lstm_hold_proba'] = np.random.uniform(0.3, 0.7, len(result_df))
            result_df['lstm_sell_proba'] = np.random.uniform(0, 0.4, len(result_df))
        
        # Generate signals
        result_df['signal'] = 0
        result_df['signal_confidence'] = 0.0
        result_df['signal_reason'] = 'HOLD'
        
        for idx, row in result_df.iterrows():
            buy_prob = row['lstm_buy_proba']
            hold_prob = row['lstm_hold_proba']
            sell_prob = row['lstm_sell_proba']
            
            signal, confidence, reason = self._generate_ultra_signal(buy_prob, hold_prob, sell_prob, row)
            
            result_df.at[idx, 'signal'] = signal
            result_df.at[idx, 'signal_confidence'] = confidence
            result_df.at[idx, 'signal_reason'] = reason
        
        # Add signal labels
        result_df['signal_label'] = result_df['signal'].map(self.signal_map)
        
        # Analyze signal distribution
        signal_counts = result_df['signal'].value_counts()
        logger.info(f"Signal distribution: {signal_counts.to_dict()}")
        
        return result_df
    
    def _generate_ultra_signal(self, buy_prob: float, hold_prob: float, sell_prob: float, row: pd.Series) -> Tuple[int, float, str]:
        """Generate ultra-aggressive signal"""
        # Normalize probabilities
        total_prob = buy_prob + hold_prob + sell_prob
        if total_prob > 0:
            buy_prob /= total_prob
            hold_prob /= total_prob
            sell_prob /= total_prob
        
        # ULTRA-AGGRESSIVE SIGNAL GENERATION
        
        # 1. Ultra-low thresholds
        if buy_prob >= self.config.buy_threshold:
            return 2, buy_prob, f"BUY_ULTRA_{buy_prob:.3f}"
        
        if sell_prob >= self.config.sell_threshold:
            return 0, sell_prob, f"SELL_ULTRA_{sell_prob:.3f}"
        
        # 2. Relative strength with minimal thresholds
        if buy_prob > sell_prob and buy_prob >= 0.001:
            return 2, buy_prob, f"BUY_RELATIVE_{buy_prob:.3f}"
        
        if sell_prob > buy_prob and sell_prob >= 0.001:
            return 0, sell_prob, f"SELL_RELATIVE_{sell_prob:.3f}"
        
        # 3. Force signals when HOLD is not extremely dominant
        if hold_prob < 0.95:
            if buy_prob > sell_prob:
                return 2, buy_prob, f"BUY_FORCED_{buy_prob:.3f}"
            else:
                return 0, sell_prob, f"SELL_FORCED_{sell_prob:.3f}"
        
        # 4. Momentum-based signals
        if self.config.use_momentum:
            if 'RSI' in row and row['RSI'] < 30:
                return 1, 0.5, "BUY_RSI_OVERSOLD"
            elif 'RSI' in row and row['RSI'] > 70:
                return -1, 0.5, "SELL_RSI_OVERBOUGHT"
        
        # 5. Price action signals
        if 'close' in row and 'SMA_20' in row:
            if row['close'] > row['SMA_20'] * 1.001:  # 0.1% above SMA
                return 1, 0.4, "BUY_PRICE_ABOVE_SMA"
            elif row['close'] < row['SMA_20'] * 0.999:  # 0.1% below SMA
                return -1, 0.4, "SELL_PRICE_BELOW_SMA"
        
        # 6. MACD signals
        if 'MACD' in row and 'MACD_signal' in row:
            if row['MACD'] > row['MACD_signal']:
                return 1, 0.3, "BUY_MACD_CROSS"
            else:
                return -1, 0.3, "SELL_MACD_CROSS"
        
        # 7. Bollinger Bands signals
        if 'BB_upper' in row and 'BB_lower' in row and 'close' in row:
            if row['close'] <= row['BB_lower']:
                return 1, 0.4, "BUY_BB_OVERSOLD"
            elif row['close'] >= row['BB_upper']:
                return -1, 0.4, "SELL_BB_OVERBOUGHT"
        
        # 8. Force signals if configured
        if self.config.force_signals:
            if buy_prob > 0:
                return 2, buy_prob, f"BUY_FORCED_FINAL_{buy_prob:.3f}"
            elif sell_prob > 0:
                return 0, sell_prob, f"SELL_FORCED_FINAL_{sell_prob:.3f}"
        
        # 9. Only HOLD when absolutely necessary
        if hold_prob >= self.config.hold_threshold:
            return 1, hold_prob, f"HOLD_STRONG_{hold_prob:.3f}"
        
        # 10. Final fallback: alternate BUY/SELL
        return 1, 0.1, "BUY_FALLBACK"  # Default to BUY

class HighFrequencyBacktester:
    """High-frequency backtesting system"""
    
    def __init__(self, config: HighFrequencyConfig):
        self.config = config
        self.trades = []
        self.equity_curve = []
    
    def run_high_frequency_backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run high-frequency backtest"""
        logger.info("Starting high-frequency backtest...")
        
        # Initialize variables
        balance = self.config.initial_balance
        equity = balance
        open_trades = []
        trade_count = 0
        
        # Process each signal
        for idx, row in df.iterrows():
            current_price = row['close']
            signal = row['signal']
            confidence = row['signal_confidence']
            
            # Update open trades
            open_trades = self._update_open_trades(open_trades, current_price, balance)
            
            # Check if we can open new trades
            if len(open_trades) < self.config.max_open_trades and signal != 0:
                # Calculate position size
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
                        'status': 'open',
                        'trade_id': trade_count + 1
                    }
                    open_trades.append(trade)
                    trade_count += 1
                    
                    logger.info(f"Opened trade #{trade_count}: {signal} @ {current_price}")
            
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
        
        logger.info(f"High-frequency backtest completed. Total trades: {len(self.trades)}")
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
                logger.info(f"Trade #{trade['trade_id']} closed by stop loss")
                continue
            
            if trade['signal'] == 1 and current_price >= trade['stop_loss']:
                trade['exit_price'] = trade['stop_loss']
                trade['status'] = 'closed'
                trade['realized_pnl'] = self._calculate_pnl(trade)
                balance += trade['realized_pnl']
                self.trades.append(trade)
                logger.info(f"Trade #{trade['trade_id']} closed by stop loss")
                continue
            
            # Check take profit
            if trade['signal'] == 2 and current_price >= trade['take_profit']:
                trade['exit_price'] = trade['take_profit']
                trade['status'] = 'closed'
                trade['realized_pnl'] = self._calculate_pnl(trade)
                balance += trade['realized_pnl']
                self.trades.append(trade)
                logger.info(f"Trade #{trade['trade_id']} closed by take profit")
                continue
            
            if trade['signal'] == 1 and current_price <= trade['take_profit']:
                trade['exit_price'] = trade['take_profit']
                trade['status'] = 'closed'
                trade['realized_pnl'] = self._calculate_pnl(trade)
                balance += trade['realized_pnl']
                self.trades.append(trade)
                logger.info(f"Trade #{trade['trade_id']} closed by take profit")
                continue
            
            # Update unrealized PnL
            trade['unrealized_pnl'] = self._calculate_unrealized_pnl(trade, current_price)
            updated_trades.append(trade)
        
        return updated_trades
    
    def _calculate_position_size(self, balance: float, current_price: float) -> float:
        """Calculate position size"""
        risk_amount = balance * self.config.risk_per_trade
        pip_value = 0.1  # For XAUUSD
        stop_loss_pips = self.config.stop_loss_pips
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return min(position_size, balance * 0.05)  # Max 5% of balance
    
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
        """Calculate realized PnL"""
        if trade['signal'] == 2:  # BUY
            pnl = (trade['exit_price'] - trade['entry_price']) * trade['position_size']
        else:  # SELL
            pnl = (trade['entry_price'] - trade['exit_price']) * trade['position_size']
        
        # Subtract commission
        pnl -= self.config.commission * trade['position_size']
        return pnl
    
    def _calculate_unrealized_pnl(self, trade: Dict, current_price: float) -> float:
        """Calculate unrealized PnL"""
        if trade['signal'] == 2:  # BUY
            pnl = (current_price - trade['entry_price']) * trade['position_size']
        else:  # SELL
            pnl = (trade['entry_price'] - current_price) * trade['position_size']
        
        return pnl
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
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
            'equity_curve': self.equity_curve,
            'trades_per_day': total_trades / max(1, len(self.equity_curve) / 96)  # Assuming 96 15-min bars per day
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

def main():
    """Main function for high-frequency trading system"""
    logger.info("=== MRBEN High-Frequency LSTM Trading System ===")
    
    # Load data
    data_file = "lstm_signals_pro.csv"
    if not os.path.exists(data_file):
        logger.error(f"Data file {data_file} not found!")
        return
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded data: {len(df)} rows")
    
    # Create configuration
    config = HighFrequencyConfig()
    
    # Generate ultra-aggressive signals
    signal_generator = HighFrequencySignalGenerator(config)
    signals_df = signal_generator.generate_ultra_signals(df)
    
    # Run high-frequency backtest
    backtester = HighFrequencyBacktester(config)
    performance = backtester.run_high_frequency_backtest(signals_df)
    
    # Print results
    print("\n" + "="*80)
    print("HIGH-FREQUENCY TRADING RESULTS")
    print("="*80)
    print(f"Total Trades: {performance.get('total_trades', 0):,}")
    print(f"Winning Trades: {performance.get('winning_trades', 0):,}")
    print(f"Losing Trades: {performance.get('losing_trades', 0):,}")
    print(f"Win Rate: {performance.get('win_rate', 0)*100:.1f}%")
    print(f"Total Return: {performance.get('total_return', 0)*100:.1f}%")
    print(f"Max Drawdown: {performance.get('max_drawdown', 0)*100:.1f}%")
    print(f"Final Balance: ${performance.get('final_balance', 0):,.2f}")
    print(f"Trades per Day: {performance.get('trades_per_day', 0):.1f}")
    print("="*80)
    
    # Save results
    signals_df.to_csv("outputs/high_frequency_signals.csv", index=False)
    
    # Save performance
    with open("outputs/high_frequency_performance.json", 'w') as f:
        json.dump(performance, f, indent=2, default=str)
    
    logger.info("High-frequency trading system completed!")

if __name__ == "__main__":
    main() 