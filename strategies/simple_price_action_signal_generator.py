import logging

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimplePriceActionSignalGenerator:
    """
    Simple price action based signal generator for LSTM training
    Produces more signals based on basic price movements
    """

    def __init__(self, lookback_period=5, price_threshold=0.001):
        self.lookback_period = lookback_period
        self.price_threshold = price_threshold
        logger.info(
            f"Initialized SimplePriceActionSignalGenerator with lookback={lookback_period}, threshold={price_threshold}"
        )

    def generate_signals(self, df):
        """
        Generate simple price action signals
        BUY = 2, HOLD = 1, SELL = 0
        """
        logger.info("Generating simple price action signals...")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Calculate basic price movements
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()

        # Calculate rolling price changes
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)

        # Calculate volatility
        df['volatility'] = df['price_change'].rolling(window=20).std()

        # Simple momentum indicators
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)

        # Initialize signal column
        df['signal'] = 1  # Default to HOLD

        # Generate signals based on simple price action rules
        for i in range(self.lookback_period, len(df)):
            current_price = df.iloc[i]['close']
            prev_price = df.iloc[i - 1]['close']
            price_change = (current_price - prev_price) / prev_price

            # Get recent price history
            recent_prices = df.iloc[i - self.lookback_period : i]['close'].values
            recent_changes = df.iloc[i - self.lookback_period : i]['price_change'].values

            # Simple BUY conditions (more aggressive)
            buy_conditions = [
                price_change > self.price_threshold,  # Price up
                current_price > recent_prices.mean(),  # Above recent average
                len([x for x in recent_changes if x > 0]) >= 3,  # Mostly positive recent changes
                df.iloc[i]['momentum_5'] > 0,  # Positive 5-period momentum
            ]

            # Simple SELL conditions (more aggressive)
            sell_conditions = [
                price_change < -self.price_threshold,  # Price down
                current_price < recent_prices.mean(),  # Below recent average
                len([x for x in recent_changes if x < 0]) >= 3,  # Mostly negative recent changes
                df.iloc[i]['momentum_5'] < 0,  # Negative 5-period momentum
            ]

            # Assign signals (more aggressive thresholds)
            if sum(buy_conditions) >= 2:  # At least 2 buy conditions
                df.iloc[i, df.columns.get_loc('signal')] = 2  # BUY
            elif sum(sell_conditions) >= 2:  # At least 2 sell conditions
                df.iloc[i, df.columns.get_loc('signal')] = 0  # SELL
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1  # HOLD

        # Add some random signals to increase variety (for training purposes)
        random_signals = np.random.choice([0, 1, 2], size=len(df), p=[0.3, 0.4, 0.3])
        df['random_signal'] = random_signals

        # Combine price action with some random signals for more variety
        mask = np.random.random(len(df)) < 0.2  # 20% random signals
        df.loc[mask, 'signal'] = df.loc[mask, 'random_signal']

        # Ensure first few rows are HOLD
        df.loc[: self.lookback_period, 'signal'] = 1

        # Calculate signal distribution
        signal_counts = df['signal'].value_counts().sort_index()
        logger.info(f"Signal distribution: {signal_counts.to_dict()}")

        return df

    def add_basic_features(self, df):
        """
        Add basic features for LSTM training
        """
        logger.info("Adding basic features for LSTM training...")

        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(2)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)

        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()

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

        # Volume features (if available)
        if 'tick_volume' in df.columns:
            df['volume_change'] = df['tick_volume'].pct_change()
            df['volume_sma_5'] = df['tick_volume'].rolling(window=5).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma_5']
        else:
            # Create dummy volume features
            df['volume_change'] = 0
            df['volume_sma_5'] = 1
            df['volume_ratio'] = 1

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)

        return df


def main():
    """
    Main function to generate simple price action signals
    """
    logger.info("Starting Simple Price Action Signal Generation...")

    # Load data
    try:
        # Try to load the features file first
        df = pd.read_csv('lstm_signals_features.csv')
        logger.info(f"Loaded lstm_signals_features.csv with {len(df)} rows")
    except FileNotFoundError:
        try:
            # Fallback to original signals file
            df = pd.read_csv('lstm_signals.csv')
            logger.info(f"Loaded lstm_signals.csv with {len(df)} rows")
        except FileNotFoundError:
            logger.error(
                "No data file found. Please ensure lstm_signals.csv or lstm_signals_features.csv exists."
            )
            return

    # Initialize signal generator
    signal_generator = SimplePriceActionSignalGenerator(
        lookback_period=5, price_threshold=0.0005  # Very low threshold for more signals
    )

    # Add basic features
    df = signal_generator.add_basic_features(df)

    # Generate signals
    df = signal_generator.generate_signals(df)

    # Save results
    output_file = 'simple_price_action_signals.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Saved signals to {output_file}")

    # Print signal distribution
    signal_counts = df['signal'].value_counts().sort_index()
    print("\n" + "=" * 50)
    print("SIMPLE PRICE ACTION SIGNAL DISTRIBUTION")
    print("=" * 50)
    buy_count = signal_counts.get(2, 0)
    hold_count = signal_counts.get(1, 0)
    sell_count = signal_counts.get(0, 0)
    total_count = len(df)

    print(f"BUY (2):  {buy_count:,} signals ({buy_count/total_count*100:.1f}%)")
    print(f"HOLD (1): {hold_count:,} signals ({hold_count/total_count*100:.1f}%)")
    print(f"SELL (0): {sell_count:,} signals ({sell_count/total_count*100:.1f}%)")
    print(f"Total:    {total_count:,} signals")
    print("=" * 50)

    # Show sample of signals
    print("\nSample of generated signals:")
    # Check what columns are available
    available_cols = [col for col in ['datetime', 'close', 'signal'] if col in df.columns]
    if available_cols:
        sample_df = df[available_cols].tail(20)
        print(sample_df.to_string(index=False))
    else:
        print("Available columns:", df.columns.tolist()[:10])
        sample_df = df[['close', 'signal']].tail(20)
        print(sample_df.to_string(index=False))

    logger.info("Simple price action signal generation completed!")


if __name__ == "__main__":
    main()
