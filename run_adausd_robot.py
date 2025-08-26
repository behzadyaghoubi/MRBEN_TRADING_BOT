import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADAUSDRobot:
    """
    Simple ADAUSD Trading Robot
    """
    
    def __init__(self, initial_capital=1000, position_size=0.1):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.capital = initial_capital
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0
        self.entry_time = None
        self.trades = []
        self.current_signal = 1  # Default to HOLD
        
    def load_signals(self, signal_file='adausd_signals.csv'):
        """
        Load pre-generated signals
        """
        try:
            df = pd.read_csv(signal_file)
            logger.info(f"Loaded {len(df)} signals from {signal_file}")
            return df
        except FileNotFoundError:
            logger.error(f"Signal file {signal_file} not found!")
            return None
    
    def get_current_price(self, df, index):
        """
        Get current price from dataframe
        """
        if index < len(df):
            return df.iloc[index]['close']
        return None
    
    def get_current_signal(self, df, index):
        """
        Get current signal from dataframe
        """
        if index < len(df):
            return df.iloc[index]['signal']
        return 1  # Default to HOLD
    
    def execute_trade(self, action, price, time_str, reason=""):
        """
        Execute a trade
        """
        if action == "BUY" and self.position == 0:
            self.position = 1
            self.entry_price = price
            self.entry_time = time_str
            logger.info(f"🟢 BUY ADAUSD at ${price:.4f} - {time_str} {reason}")
            
        elif action == "SELL" and self.position == 0:
            self.position = -1
            self.entry_price = price
            self.entry_time = time_str
            logger.info(f"🔴 SELL ADAUSD at ${price:.4f} - {time_str} {reason}")
            
        elif action == "CLOSE_LONG" and self.position == 1:
            pnl = (price - self.entry_price) / self.entry_price
            capital_change = self.capital * pnl * self.position_size
            self.capital += capital_change
            
            self.trades.append({
                'entry_time': self.entry_time,
                'exit_time': time_str,
                'entry_price': self.entry_price,
                'exit_price': price,
                'position': 'LONG',
                'pnl': pnl,
                'capital_change': capital_change,
                'reason': reason
            })
            
            logger.info(f"📈 Close LONG: Entry=${self.entry_price:.4f}, Exit=${price:.4f}, PnL={pnl*100:.2f}%")
            self.position = 0
            self.entry_price = 0
            self.entry_time = None
            
        elif action == "CLOSE_SHORT" and self.position == -1:
            pnl = (self.entry_price - price) / self.entry_price
            capital_change = self.capital * pnl * self.position_size
            self.capital += capital_change
            
            self.trades.append({
                'entry_time': self.entry_time,
                'exit_time': time_str,
                'entry_price': self.entry_price,
                'exit_price': price,
                'position': 'SHORT',
                'pnl': pnl,
                'capital_change': capital_change,
                'reason': reason
            })
            
            logger.info(f"📉 Close SHORT: Entry=${self.entry_price:.4f}, Exit=${price:.4f}, PnL={pnl*100:.2f}%")
            self.position = 0
            self.entry_price = 0
            self.entry_time = None
    
    def check_stop_loss_take_profit(self, current_price, stop_loss=0.05, take_profit=0.1):
        """
        Check if stop loss or take profit should be triggered
        """
        if self.position == 1:  # Long position
            pnl = (current_price - self.entry_price) / self.entry_price
            if pnl <= -stop_loss:
                return "CLOSE_LONG", "STOP_LOSS"
            elif pnl >= take_profit:
                return "CLOSE_LONG", "TAKE_PROFIT"
                
        elif self.position == -1:  # Short position
            pnl = (self.entry_price - current_price) / self.entry_price
            if pnl <= -stop_loss:
                return "CLOSE_SHORT", "STOP_LOSS"
            elif pnl >= take_profit:
                return "CLOSE_SHORT", "TAKE_PROFIT"
        
        return None, None
    
    def run_simulation(self, df, delay=0.1):
        """
        Run trading simulation
        """
        logger.info("🤖 Starting ADAUSD Trading Robot Simulation...")
        logger.info(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"📊 Total Data Points: {len(df)}")
        logger.info("="*60)
        
        for i in range(len(df)):
            current_price = self.get_current_price(df, i)
            current_signal = self.get_current_signal(df, i)
            current_time = df.iloc[i]['time']
            
            if current_price is None:
                continue
            
            # Check stop loss and take profit first
            action, reason = self.check_stop_loss_take_profit(current_price)
            if action:
                self.execute_trade(action, current_price, current_time, reason)
            
            # Handle new signals
            if self.position == 0:  # No position
                if current_signal == 2:  # BUY signal
                    self.execute_trade("BUY", current_price, current_time, "SIGNAL")
                elif current_signal == 0:  # SELL signal
                    self.execute_trade("SELL", current_price, current_time, "SIGNAL")
            
            # Print status every 100 iterations
            if i % 100 == 0:
                logger.info(f"⏰ {current_time} - Price: ${current_price:.4f} - Signal: {current_signal} - Capital: ${self.capital:,.2f}")
            
            # Simulate real-time delay
            time.sleep(delay)
        
        # Close any remaining position
        if self.position != 0:
            final_price = df.iloc[-1]['close']
            final_time = df.iloc[-1]['time']
            if self.position == 1:
                self.execute_trade("CLOSE_LONG", final_price, final_time, "END_OF_DATA")
            else:
                self.execute_trade("CLOSE_SHORT", final_price, final_time, "END_OF_DATA")
        
        self.print_results()
    
    def print_results(self):
        """
        Print trading results
        """
        print("\n" + "="*60)
        print("🤖 ADAUSD TRADING ROBOT RESULTS")
        print("="*60)
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"💰 Final Capital:   ${self.capital:,.2f}")
        print(f"📈 Total Return:    {((self.capital - self.initial_capital) / self.initial_capital) * 100:.2f}%")
        print(f"🔄 Total Trades:    {len(self.trades)}")
        
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            
            print(f"✅ Winning Trades:  {winning_trades}")
            print(f"❌ Losing Trades:   {losing_trades}")
            print(f"📊 Win Rate:        {(winning_trades / len(trades_df)) * 100:.1f}%")
            
            if winning_trades > 0:
                avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() * 100
                print(f"📈 Avg Win:         {avg_win:.2f}%")
            
            if losing_trades > 0:
                avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() * 100
                print(f"📉 Avg Loss:        {avg_loss:.2f}%")
        
        print("="*60)
        
        # Show recent trades
        if self.trades:
            print("\n📋 Recent Trades:")
            trades_df = pd.DataFrame(self.trades)
            print(trades_df.tail(5).to_string(index=False))

def main():
    """
    Main function to run ADAUSD robot
    """
    logger.info("🚀 Starting ADAUSD Trading Robot...")
    
    # Initialize robot
    robot = ADAUSDRobot(
        initial_capital=1000,  # $1000 starting capital
        position_size=0.1      # 10% of capital per trade
    )
    
    # Load signals
    df = robot.load_signals('adausd_signals.csv')
    if df is None:
        logger.error("Failed to load signals. Please run adausd_signal_generator.py first.")
        return
    
    # Run simulation
    robot.run_simulation(df, delay=0.01)  # Fast simulation
    
    logger.info("✅ ADAUSD Trading Robot completed!")

if __name__ == "__main__":
    main() 