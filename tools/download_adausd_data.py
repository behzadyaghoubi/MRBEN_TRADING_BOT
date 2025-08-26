import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_adausd_data(days=30, interval='15m'):
    """
    Download ADAUSD data from Yahoo Finance
    """
    logger.info(f"Downloading ADAUSD data for last {days} days with {interval} interval...")

    try:
        # Download ADAUSD data
        ticker = yf.Ticker("ADA-USD")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Download data
        data = ticker.history(start=start_date, end=end_date, interval=interval)

        if data.empty:
            logger.error("No data downloaded. Trying alternative method...")
            # Try alternative symbol
            ticker = yf.Ticker("ADAUSD=X")
            data = ticker.history(start=start_date, end=end_date, interval=interval)

        if data.empty:
            logger.error("Failed to download ADAUSD data")
            return None

        # Reset index to get datetime as column
        data = data.reset_index()

        # Rename columns to match our format
        data = data.rename(
            columns={
                'Datetime': 'time',
                'Date': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'tick_volume',
            }
        )

        # Ensure time column is datetime
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])

        # Add missing columns if needed
        if 'tick_volume' not in data.columns:
            data['tick_volume'] = 1000000  # Default tick_volume

        logger.info(f"Downloaded {len(data)} rows of ADAUSD data")
        logger.info(f"Date range: {data['time'].min()} to {data['time'].max()}")

        return data

    except Exception as e:
        logger.error(f"Error downloading ADAUSD data: {e}")
        return None


def create_mock_adausd_data(days=30, interval_minutes=15):
    """
    Create mock ADAUSD data for testing if download fails
    """
    logger.info(f"Creating mock ADAUSD data for {days} days...")

    # Generate time series
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    # Create time intervals
    time_delta = timedelta(minutes=interval_minutes)
    times = []
    current_time = start_time
    while current_time <= end_time:
        times.append(current_time)
        current_time += time_delta

    # Generate mock price data (realistic ADA price movements)
    np.random.seed(42)  # For reproducible results

    # Start with realistic ADA price
    base_price = 0.45  # ADA typical price
    prices = [base_price]

    for i in range(1, len(times)):
        # Generate realistic price movements
        change_pct = np.random.normal(0, 0.02)  # 2% volatility
        new_price = prices[-1] * (1 + change_pct)
        prices.append(max(new_price, 0.01))  # Minimum price

    # Create OHLCV data
    data = []
    for i, (time, close) in enumerate(zip(times, prices, strict=False)):
        # Generate realistic OHLC from close price
        volatility = np.random.uniform(0.005, 0.02)

        high = close * (1 + np.random.uniform(0, volatility))
        low = close * (1 - np.random.uniform(0, volatility))
        open_price = np.random.uniform(low, high)

        volume = np.random.uniform(1000000, 10000000)

        data.append(
            {
                'time': time,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'tick_volume': volume,
            }
        )

    df = pd.DataFrame(data)
    logger.info(f"Created {len(df)} rows of mock ADAUSD data")

    return df


def main():
    """
    Main function to download or create ADAUSD data
    """
    logger.info("Starting ADAUSD data preparation...")

    # Try to download real data first
    data = download_adausd_data(days=30, interval='15m')

    if data is None:
        logger.info("Download failed, creating mock data...")
        data = create_mock_adausd_data(days=30, interval_minutes=15)

    if data is not None:
        # Save data
        output_file = 'adausd_data.csv'
        data.to_csv(output_file, index=False)
        logger.info(f"Saved ADAUSD data to {output_file}")

        # Print sample
        print("\n" + "=" * 50)
        print("ADAUSD DATA SAMPLE")
        print("=" * 50)
        print(f"Total rows: {len(data)}")
        print(f"Date range: {data['time'].min()} to {data['time'].max()}")
        print(f"Price range: ${data['close'].min():.4f} - ${data['close'].max():.4f}")
        print("\nFirst 5 rows:")
        print(data.head().to_string(index=False))
        print("\nLast 5 rows:")
        print(data.tail().to_string(index=False))
        print("=" * 50)

        return data
    else:
        logger.error("Failed to prepare ADAUSD data")
        return None


if __name__ == "__main__":
    main()
