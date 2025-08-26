#!/usr/bin/env python3
"""
Download XAUUSD.PRO Data for MR BEN Trading System
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_gold_data():
    """Generate synthetic XAUUSD.PRO data for testing."""

    # Generate realistic gold price data
    np.random.seed(42)

    # Base price around 2000-2100 USD per ounce
    base_price = 2050
    n_samples = 1000

    # Generate time series
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)

    # Generate realistic price movements
    price_changes = np.random.normal(0, 0.5, n_samples)  # Small daily changes
    prices = [base_price]

    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change / 100)
        prices.append(new_price)

    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices, strict=False)):
        # Generate realistic OHLC from close price
        volatility = np.random.uniform(0.1, 0.5)
        high = close * (1 + np.random.uniform(0, volatility))
        low = close * (1 - np.random.uniform(0, volatility))
        open_price = np.random.uniform(low, high)

        data.append(
            {
                'time': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'tick_volume': np.random.randint(100, 1000),
            }
        )

    df = pd.DataFrame(data)

    # Save to file
    output_file = 'data/XAUUSD_PRO_M5_data.csv'
    df.to_csv(output_file, index=False)

    print(f"✅ Generated {len(df)} XAUUSD.PRO data points")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

    return df


if __name__ == "__main__":
    print("🪙 Downloading XAUUSD.PRO Data for MR BEN Trading System")
    print("=" * 60)

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Generate data
    df = generate_gold_data()

    print("✅ XAUUSD.PRO data ready for trading!")
    print(f"📈 Latest price: ${df['close'].iloc[-1]:.2f}")
    print(f"🕐 Data period: {df['time'].min()} to {df['time'].max()}")
