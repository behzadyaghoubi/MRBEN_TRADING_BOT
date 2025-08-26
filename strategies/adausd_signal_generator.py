import pandas as pd
import numpy as np
import talib
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADAUSDSignalGenerator:
    """
    ADAUSD signal generator using simple price action strategy
    """
    
    def __init__(self, lookback_period=5, price_threshold=0.005):
        self.lookback_period = lookback_period
        self.price_threshold = price_threshold
        logger.info(f"Initialized ADAUSD Signal Generator with lookback={lookback_period}, threshold={price_threshold}")
    
    def add_features(self, df):
        """
        Add technical features for ADAUSD
        """
        logger.info("Adding technical features for ADAUSD...")
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(2)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Price position relative to moving averages
        df['price_vs_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
        df['price_vs_sma10'] = (df['close'] - df['sma_10']) / df['sma_10']
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Volatility
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_10'] = df['price_change'].rolling(window=10).std()
        
        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_5'] = df['hl_range'].rolling(window=5).mean()
        
        # RSI
        try:
            df['RSI'] = talib.RSI(df['close'].values, timeperiod=14)
        except:
            # Simple RSI calculation if talib fails
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        try:
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
            df['MACD'] = macd
            df['MACD_signal'] = macd_signal
            df['MACD_hist'] = macd_hist
        except:
            # Simple MACD calculation
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        try:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_pos'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        except:
            # Simple Bollinger Bands
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma20 + (std20 * 2)
            df['bb_middle'] = sma20
            df['bb_lower'] = sma20 - (std20 * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        if 'tick_volume' in df.columns:
            df['volume_change'] = df['tick_volume'].pct_change()
            df['volume_sma_5'] = df['tick_volume'].rolling(window=5).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma_5']
        else:
            df['volume_change'] = 0
            df['volume_sma_5'] = 1
            df['volume_ratio'] = 1
        
        # Fill NaN values
        df = df.ffill().bfill().fillna(0)
        
        return df
    
    def generate_signals(self, df):
        """
        Generate signals for ADAUSD using simple price action
        BUY = 2, HOLD = 1, SELL = 0
        """
        logger.info("Generating ADAUSD signals...")
        
        # Make a copy
        df = df.copy()
        
        # Initialize signal column
        df['signal'] = 1  # Default to HOLD
        
        # Generate signals based on simple price action rules
        for i in range(self.lookback_period, len(df)):
            current_price = df.iloc[i]['close']
            prev_price = df.iloc[i-1]['close']
            price_change = (current_price - prev_price) / prev_price
            
            # Get recent price history
            recent_prices = df.iloc[i-self.lookback_period:i]['close'].values
            recent_changes = df.iloc[i-self.lookback_period:i]['price_change'].values
            
            # Simple BUY conditions (more aggressive for crypto)
            buy_conditions = [
                price_change > self.price_threshold,  # Price up
                current_price > recent_prices.mean(),  # Above recent average
                len([x for x in recent_changes if x > 0]) >= 3,  # Mostly positive recent changes
                df.iloc[i]['momentum_5'] > 0,  # Positive 5-period momentum
                df.iloc[i]['RSI'] < 70,  # Not overbought
                df.iloc[i]['MACD'] > df.iloc[i]['MACD_signal'],  # MACD bullish
            ]
            
            # Simple SELL conditions (more aggressive for crypto)
            sell_conditions = [
                price_change < -self.price_threshold,  # Price down
                current_price < recent_prices.mean(),  # Below recent average
                len([x for x in recent_changes if x < 0]) >= 3,  # Mostly negative recent changes
                df.iloc[i]['momentum_5'] < 0,  # Negative 5-period momentum
                df.iloc[i]['RSI'] > 30,  # Not oversold
                df.iloc[i]['MACD'] < df.iloc[i]['MACD_signal'],  # MACD bearish
            ]
            
            # Assign signals (more aggressive thresholds for crypto)
            if sum(buy_conditions) >= 3:  # At least 3 buy conditions
                df.iloc[i, df.columns.get_loc('signal')] = 2  # BUY
            elif sum(sell_conditions) >= 3:  # At least 3 sell conditions
                df.iloc[i, df.columns.get_loc('signal')] = 0  # SELL
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1  # HOLD
        
        # Add some random signals to increase variety (for training purposes)
        random_signals = np.random.choice([0, 1, 2], size=len(df), p=[0.25, 0.5, 0.25])
        df['random_signal'] = random_signals
        
        # Combine price action with some random signals for more variety
        mask = np.random.random(len(df)) < 0.15  # 15% random signals
        df.loc[mask, 'signal'] = df.loc[mask, 'random_signal']
        
        # Ensure first few rows are HOLD
        df.loc[:self.lookback_period, 'signal'] = 1
        
        # Calculate signal distribution
        signal_counts = df['signal'].value_counts().sort_index()
        logger.info(f"Signal distribution: {signal_counts.to_dict()}")
        
        return df

def main():
    """
    Main function to generate ADAUSD signals
    """
    logger.info("Starting ADAUSD Signal Generation...")
    
    # Load ADAUSD data
    try:
        df = pd.read_csv('adausd_data.csv')
        logger.info(f"Loaded adausd_data.csv with {len(df)} rows")
    except FileNotFoundError:
        logger.error("adausd_data.csv not found. Please run download_adausd_data.py first.")
        return
    
    # Initialize signal generator
    signal_generator = ADAUSDSignalGenerator(
        lookback_period=5,
        price_threshold=0.003  # Lower threshold for crypto volatility
    )
    
    # Add features
    df = signal_generator.add_features(df)
    
    # Generate signals
    df = signal_generator.generate_signals(df)
    
    # Save results
    output_file = 'adausd_signals.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Saved ADAUSD signals to {output_file}")
    
    # Print signal distribution
    signal_counts = df['signal'].value_counts().sort_index()
    print("\n" + "="*50)
    print("ADAUSD SIGNAL DISTRIBUTION")
    print("="*50)
    buy_count = signal_counts.get(2, 0)
    hold_count = signal_counts.get(1, 0)
    sell_count = signal_counts.get(0, 0)
    total_count = len(df)
    
    print(f"BUY (2):  {buy_count:,} signals ({buy_count/total_count*100:.1f}%)")
    print(f"HOLD (1): {hold_count:,} signals ({hold_count/total_count*100:.1f}%)")
    print(f"SELL (0): {sell_count:,} signals ({sell_count/total_count*100:.1f}%)")
    print(f"Total:    {total_count:,} signals")
    print("="*50)
    
    # Show sample of signals
    print("\nSample of generated signals:")
    sample_cols = ['time', 'close', 'signal', 'RSI', 'MACD']
    available_cols = [col for col in sample_cols if col in df.columns]
    if available_cols:
        sample_df = df[available_cols].tail(20)
        print(sample_df.to_string(index=False))
    else:
        sample_df = df[['close', 'signal']].tail(20)
        print(sample_df.to_string(index=False))
    
    logger.info("ADAUSD signal generation completed!")

if __name__ == "__main__":
    main() 