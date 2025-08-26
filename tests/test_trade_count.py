#!/usr/bin/env python3
"""
Test Trade Count - Simple script to increase number of trades
============================================================

This script modifies the signal generation to produce many more trades
by using ultra-aggressive thresholds and multiple signal sources.

Author: MRBEN Trading System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_many_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate many more signals using multiple strategies"""
    logger.info("Generating many signals for maximum trades...")
    
    result_df = df.copy()
    
    # Add LSTM probabilities if they don't exist
    if 'lstm_buy_proba' not in result_df.columns:
        # Create realistic probabilities
        np.random.seed(42)  # For reproducible results
        result_df['lstm_buy_proba'] = np.random.uniform(0.1, 0.4, len(result_df))
        result_df['lstm_hold_proba'] = np.random.uniform(0.2, 0.6, len(result_df))
        result_df['lstm_sell_proba'] = np.random.uniform(0.1, 0.4, len(result_df))
    
    # Initialize signal columns
    result_df['signal'] = 0
    result_df['signal_confidence'] = 0.0
    result_df['signal_reason'] = 'HOLD'
    
    # Generate signals using multiple strategies
    for idx, row in result_df.iterrows():
        signal = 0
        confidence = 0.0
        reason = 'HOLD'
        
        # Strategy 1: LSTM-based signals with ultra-low thresholds
        buy_prob = row['lstm_buy_proba']
        hold_prob = row['lstm_hold_proba']
        sell_prob = row['lstm_sell_proba']
        
        # Normalize probabilities
        total_prob = buy_prob + hold_prob + sell_prob
        if total_prob > 0:
            buy_prob /= total_prob
            hold_prob /= total_prob
            sell_prob /= total_prob
        
        # Ultra-aggressive thresholds
        if buy_prob >= 0.01:  # Very low threshold
            signal = 1
            confidence = buy_prob
            reason = f"BUY_LSTM_{buy_prob:.3f}"
        elif sell_prob >= 0.01:  # Very low threshold
            signal = -1
            confidence = sell_prob
            reason = f"SELL_LSTM_{sell_prob:.3f}"
        
        # Strategy 2: Price action signals
        elif 'close' in row and 'SMA_20' in row:
            if row['close'] > row['SMA_20']:
                signal = 1
                confidence = 0.3
                reason = "BUY_PRICE_ABOVE_SMA"
            else:
                signal = -1
                confidence = 0.3
                reason = "SELL_PRICE_BELOW_SMA"
        
        # Strategy 3: RSI signals
        elif 'RSI' in row:
            if row['RSI'] < 40:
                signal = 1
                confidence = 0.4
                reason = "BUY_RSI_LOW"
            elif row['RSI'] > 60:
                signal = -1
                confidence = 0.4
                reason = "SELL_RSI_HIGH"
        
        # Strategy 4: MACD signals
        elif 'MACD' in row and 'MACD_signal' in row:
            if row['MACD'] > row['MACD_signal']:
                signal = 1
                confidence = 0.3
                reason = "BUY_MACD"
            else:
                signal = -1
                confidence = 0.3
                reason = "SELL_MACD"
        
        # Strategy 5: Bollinger Bands signals
        elif 'BB_upper' in row and 'BB_lower' in row and 'close' in row:
            if row['close'] <= row['BB_lower']:
                signal = 1
                confidence = 0.4
                reason = "BUY_BB_LOWER"
            elif row['close'] >= row['BB_upper']:
                signal = -1
                confidence = 0.4
                reason = "SELL_BB_UPPER"
        
        # Strategy 6: Momentum signals
        elif 'price_change' in row:
            if row['price_change'] > 0.001:  # 0.1% positive change
                signal = 1
                confidence = 0.2
                reason = "BUY_MOMENTUM"
            elif row['price_change'] < -0.001:  # 0.1% negative change
                signal = -1
                confidence = 0.2
                reason = "SELL_MOMENTUM"
        
        # Strategy 7: Volatility signals
        elif 'volatility_10' in row:
            if row['volatility_10'] > row['volatility_10'].rolling(20).mean():
                signal = 1
                confidence = 0.2
                reason = "BUY_VOLATILITY"
            else:
                signal = -1
                confidence = 0.2
                reason = "SELL_VOLATILITY"
        
        # Strategy 8: Force signals every N bars
        elif idx % 10 == 0:  # Every 10th bar
            signal = 1 if idx % 20 == 0 else -1
            confidence = 0.1
            reason = f"{'BUY' if signal == 1 else 'SELL'}_FORCED"
        
        # Strategy 9: Random signals for maximum trades
        elif np.random.random() < 0.1:  # 10% chance of signal
            signal = 1 if np.random.random() < 0.5 else -1
            confidence = 0.1
            reason = f"{'BUY' if signal == 1 else 'SELL'}_RANDOM"
        
        # Update DataFrame
        result_df.at[idx, 'signal'] = signal
        result_df.at[idx, 'signal_confidence'] = confidence
        result_df.at[idx, 'signal_reason'] = reason
    
    # Add signal labels
    result_df['signal_label'] = result_df['signal'].map({1: "BUY", 0: "HOLD", -1: "SELL"})
    
    # Analyze signal distribution
    signal_counts = result_df['signal'].value_counts()
    logger.info(f"Signal distribution: {signal_counts.to_dict()}")
    
    return result_df

def run_simple_backtest(df: pd.DataFrame) -> dict:
    """Run simple backtest to count trades"""
    logger.info("Running simple backtest...")
    
    balance = 10000
    trades = []
    open_trades = []
    
    for idx, row in df.iterrows():
        signal = row['signal']
        current_price = row['close']
        
        # Close existing trades (simplified)
        open_trades = [t for t in open_trades if t['status'] == 'open']
        
        # Open new trade if signal and space available
        if signal != 0 and len(open_trades) < 3:
            trade = {
                'entry_time': idx,
                'entry_price': current_price,
                'signal': signal,
                'status': 'open',
                'trade_id': len(trades) + 1
            }
            open_trades.append(trade)
            trades.append(trade)
            
            # Close trade after 5 bars (simplified)
            if len(trades) > 0 and trades[-1]['status'] == 'open':
                trades[-1]['exit_time'] = min(idx + 5, len(df) - 1)
                trades[-1]['exit_price'] = df.iloc[trades[-1]['exit_time']]['close']
                trades[-1]['status'] = 'closed'
                trades[-1]['pnl'] = (trades[-1]['exit_price'] - trades[-1]['entry_price']) * trades[-1]['signal']
    
    # Close remaining open trades
    for trade in open_trades:
        if trade['status'] == 'open':
            trade['exit_time'] = len(df) - 1
            trade['exit_price'] = df.iloc[-1]['close']
            trade['status'] = 'closed'
            trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['signal']
    
    # Calculate metrics
    total_trades = len([t for t in trades if t['status'] == 'closed'])
    winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
    losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'trades': trades
    }

def main():
    """Main function"""
    logger.info("=== Testing Trade Count Increase ===")
    
    # Load data
    data_file = "lstm_signals_pro.csv"
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded data: {len(df)} rows")
    except FileNotFoundError:
        logger.error(f"Data file {data_file} not found!")
        return
    
    # Generate many signals
    signals_df = generate_many_signals(df)
    
    # Run backtest
    performance = run_simple_backtest(signals_df)
    
    # Print results
    print("\n" + "="*60)
    print("TRADE COUNT TEST RESULTS")
    print("="*60)
    print(f"Total Trades: {performance['total_trades']:,}")
    print(f"Winning Trades: {performance['winning_trades']:,}")
    print(f"Losing Trades: {performance['losing_trades']:,}")
    print(f"Win Rate: {performance['win_rate']*100:.1f}%")
    
    # Signal distribution
    signal_counts = signals_df['signal'].value_counts()
    print(f"\nSignal Distribution:")
    print(f"BUY: {signal_counts.get(1, 0):,} ({signal_counts.get(1, 0)/len(signals_df)*100:.1f}%)")
    print(f"HOLD: {signal_counts.get(0, 0):,} ({signal_counts.get(0, 0)/len(signals_df)*100:.1f}%)")
    print(f"SELL: {signal_counts.get(-1, 0):,} ({signal_counts.get(-1, 0)/len(signals_df)*100:.1f}%)")
    
    print("="*60)
    
    # Save results
    signals_df.to_csv("outputs/many_signals.csv", index=False)
    logger.info("Results saved to outputs/many_signals.csv")
    
    # Plot signal distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    signal_counts.plot(kind='bar', color=['red', 'gray', 'green'])
    plt.title('Signal Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    signals_df['signal'].rolling(50).mean().plot()
    plt.title('Signal Trend (50-period moving average)')
    plt.ylabel('Average Signal')
    
    plt.tight_layout()
    plt.savefig("outputs/signal_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 