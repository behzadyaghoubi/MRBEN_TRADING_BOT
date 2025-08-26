import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADAUSDBacktester:
    """
    ADAUSD backtester with crypto-appropriate parameters
    """
    
    def __init__(self, initial_capital=1000, position_size=0.1, stop_loss=0.05, take_profit=0.1):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = []
        self.equity_curve = []
        
    def backtest(self, df):
        """
        Run backtest on ADAUSD signals
        """
        logger.info("Starting ADAUSD backtest...")
        
        # Make a copy
        df = df.copy()
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        entry_time = None
        
        # Track equity
        equity_curve = []
        
        # Process each row
        for i, row in df.iterrows():
            current_price = row['close']
            signal = row['signal']
            
            # Update equity if we have a position
            if position != 0:
                if position == 1:  # Long position
                    pnl = (current_price - entry_price) / entry_price
                else:  # Short position
                    pnl = (entry_price - current_price) / entry_price
                
                current_equity = capital * (1 + pnl * self.position_size)
                
                # Check stop loss and take profit
                if position == 1:  # Long position
                    if pnl <= -self.stop_loss:  # Stop loss hit
                        capital_change = capital * pnl * self.position_size
                        capital += capital_change
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': row.get('time', str(i)),
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': 'LONG',
                            'pnl': pnl,
                            'capital_change': capital_change,
                            'exit_reason': 'STOP_LOSS'
                        })
                        
                        logger.info(f"LONG Stop Loss: Entry={entry_price:.4f}, Exit={current_price:.4f}, PnL={pnl*100:.2f}%")
                        position = 0
                        entry_price = 0
                        entry_time = None
                        
                    elif pnl >= self.take_profit:  # Take profit hit
                        capital_change = capital * pnl * self.position_size
                        capital += capital_change
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': row.get('time', str(i)),
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': 'LONG',
                            'pnl': pnl,
                            'capital_change': capital_change,
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        
                        logger.info(f"LONG Take Profit: Entry={entry_price:.4f}, Exit={current_price:.4f}, PnL={pnl*100:.2f}%")
                        position = 0
                        entry_price = 0
                        entry_time = None
                        
                else:  # Short position
                    if pnl <= -self.stop_loss:  # Stop loss hit
                        capital_change = capital * pnl * self.position_size
                        capital += capital_change
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': row.get('time', str(i)),
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': 'SHORT',
                            'pnl': pnl,
                            'capital_change': capital_change,
                            'exit_reason': 'STOP_LOSS'
                        })
                        
                        logger.info(f"SHORT Stop Loss: Entry={entry_price:.4f}, Exit={current_price:.4f}, PnL={pnl*100:.2f}%")
                        position = 0
                        entry_price = 0
                        entry_time = None
                        
                    elif pnl >= self.take_profit:  # Take profit hit
                        capital_change = capital * pnl * self.position_size
                        capital += capital_change
                        
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': row.get('time', str(i)),
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': 'SHORT',
                            'pnl': pnl,
                            'capital_change': capital_change,
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        
                        logger.info(f"SHORT Take Profit: Entry={entry_price:.4f}, Exit={current_price:.4f}, PnL={pnl*100:.2f}%")
                        position = 0
                        entry_price = 0
                        entry_time = None
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
            
            # Handle signals only if no position
            if position == 0:
                if signal == 2:  # BUY signal
                    position = 1
                    entry_price = current_price
                    entry_time = row.get('time', str(i))
                    logger.info(f"BUY ADAUSD at {current_price:.4f} - {entry_time}")
                    
                elif signal == 0:  # SELL signal
                    position = -1
                    entry_price = current_price
                    entry_time = row.get('time', str(i))
                    logger.info(f"SELL ADAUSD at {current_price:.4f} - {entry_time}")
        
        # Close any remaining position at the end
        if position != 0:
            current_price = df.iloc[-1]['close']
            if position == 1:
                pnl = (current_price - entry_price) / entry_price
                capital_change = capital * pnl * self.position_size
                capital += capital_change
                
                self.trades.append({
                    'entry_time': entry_time,
                    'exit_time': df.iloc[-1].get('time', 'end'),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': 'LONG',
                    'pnl': pnl,
                    'capital_change': capital_change,
                    'exit_reason': 'END_OF_DATA'
                })
            else:
                pnl = (entry_price - current_price) / entry_price
                capital_change = capital * pnl * self.position_size
                capital += capital_change
                
                self.trades.append({
                    'entry_time': entry_time,
                    'exit_time': df.iloc[-1].get('time', 'end'),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': 'SHORT',
                    'pnl': pnl,
                    'capital_change': capital_change,
                    'exit_reason': 'END_OF_DATA'
                })
        
        self.equity_curve = equity_curve
        self.final_capital = capital
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """
        Calculate performance metrics
        """
        if not self.trades:
            return {
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate max drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (simplified)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        total_return = (self.final_capital - self.initial_capital) / self.initial_capital
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': self.final_capital
        }
    
    def plot_results(self, df):
        """
        Plot backtest results
        """
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Price and signals
        plt.subplot(4, 1, 1)
        plt.plot(df['close'], label='ADAUSD Price', alpha=0.7, color='purple')
        
        # Plot buy signals
        buy_signals = df[df['signal'] == 2]
        plt.scatter(buy_signals.index, buy_signals['close'], 
                   color='green', marker='^', s=50, label='BUY', alpha=0.8)
        
        # Plot sell signals
        sell_signals = df[df['signal'] == 0]
        plt.scatter(sell_signals.index, sell_signals['close'], 
                   color='red', marker='v', s=50, label='SELL', alpha=0.8)
        
        plt.title('ADAUSD Price Action Signals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Equity curve
        plt.subplot(4, 1, 2)
        plt.plot(self.equity_curve, label='Equity', color='blue')
        plt.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        plt.title('Equity Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: RSI
        plt.subplot(4, 1, 3)
        if 'RSI' in df.columns:
            plt.plot(df['RSI'], label='RSI', color='orange')
            plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            plt.title('RSI Indicator')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Signal distribution
        plt.subplot(4, 1, 4)
        signal_counts = df['signal'].value_counts().sort_index()
        colors = ['red', 'gray', 'green']
        labels = ['SELL', 'HOLD', 'BUY']
        plt.bar(labels, signal_counts.values, color=colors, alpha=0.7)
        plt.title('Signal Distribution')
        plt.ylabel('Count')
        
        # Add value labels on bars
        for i, v in enumerate(signal_counts.values):
            plt.text(i, v + max(signal_counts.values) * 0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('adausd_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function to run ADAUSD backtest
    """
    logger.info("Starting ADAUSD Backtest...")
    
    # Load signals
    try:
        df = pd.read_csv('adausd_signals.csv')
        logger.info(f"Loaded adausd_signals.csv with {len(df)} rows")
    except FileNotFoundError:
        logger.error("adausd_signals.csv not found. Please run adausd_signal_generator.py first.")
        return
    
    # Initialize backtester with crypto-appropriate parameters
    backtester = ADAUSDBacktester(
        initial_capital=1000,  # $1000 initial capital
        position_size=0.1,     # 10% of capital per trade
        stop_loss=0.05,        # 5% stop loss
        take_profit=0.1        # 10% take profit
    )
    
    # Run backtest
    metrics = backtester.backtest(df)
    
    # Print results
    print("\n" + "="*60)
    print("ADAUSD BACKTEST RESULTS")
    print("="*60)
    print(f"Initial Capital: ${metrics['final_capital']:,.2f}")
    print(f"Final Capital:   ${metrics['final_capital']:,.2f}")
    print(f"Total Return:    {metrics['total_return']*100:.2f}%")
    print(f"Total Trades:    {metrics['total_trades']}")
    print(f"Winning Trades:  {metrics['winning_trades']}")
    print(f"Losing Trades:   {metrics['losing_trades']}")
    print(f"Win Rate:        {metrics['win_rate']*100:.1f}%")
    print(f"Avg Win:         {metrics['avg_win']*100:.2f}%")
    print(f"Avg Loss:        {metrics['avg_loss']*100:.2f}%")
    print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:    {metrics['max_drawdown']*100:.2f}%")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print("="*60)
    
    # Show recent trades
    if backtester.trades:
        print("\nRecent Trades:")
        trades_df = pd.DataFrame(backtester.trades)
        print(trades_df.tail(10).to_string(index=False))
    
    # Plot results
    backtester.plot_results(df)
    
    logger.info("ADAUSD backtest completed!")

if __name__ == "__main__":
    main() 