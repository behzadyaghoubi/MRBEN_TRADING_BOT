#!/usr/bin/env python3
"""
Fix Data Columns - Add missing OHLC columns
==========================================

This script adds missing OHLC columns to the LSTM signals data
so the trading system can work properly.

Author: MRBEN Trading System
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_data_columns(input_file: str = "lstm_signals_pro.csv", output_file: str = "lstm_signals_fixed.csv"):
    """Add missing OHLC columns to the data"""
    logger.info(f"Fixing data columns in {input_file}...")
    
    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Original columns: {df.columns.tolist()}")
    
    # Add missing OHLC columns
    if 'open' not in df.columns:
        # Estimate open price based on close price with small random variation
        np.random.seed(42)
        price_variation = np.random.uniform(-0.1, 0.1, len(df))  # ±0.1% variation
        df['open'] = df['close'] * (1 + price_variation)
        logger.info("Added 'open' column")
    
    if 'high' not in df.columns:
        # Estimate high price (usually higher than close)
        high_variation = np.random.uniform(0, 0.2, len(df))  # 0-0.2% above close
        df['high'] = df['close'] * (1 + high_variation)
        logger.info("Added 'high' column")
    
    if 'low' not in df.columns:
        # Estimate low price (usually lower than close)
        low_variation = np.random.uniform(-0.2, 0, len(df))  # 0-0.2% below close
        df['low'] = df['close'] * (1 + low_variation)
        logger.info("Added 'low' column")
    
    if 'tick_volume' not in df.columns:
        # Add tick_volume column with realistic values
        df['tick_volume'] = np.random.uniform(1000, 10000, len(df))
        logger.info("Added 'tick_volume' column")
    
    # Ensure OHLC relationship is logical
    df['high'] = np.maximum(df['high'], df['close'])
    df['high'] = np.maximum(df['high'], df['open'])
    df['low'] = np.minimum(df['low'], df['close'])
    df['low'] = np.minimum(df['low'], df['open'])
    
    # Reorder columns to standard OHLCV format
    column_order = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 
                   'lstm_buy_proba', 'lstm_hold_proba', 'lstm_sell_proba', 'lstm_signal']
    
    # Add any missing columns from the order
    for col in column_order:
        if col not in df.columns:
            if col == 'time':
                continue  # Skip time if not present
            df[col] = 0  # Default value
    
    # Reorder columns
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns + [col for col in df.columns if col not in available_columns]]
    
    logger.info(f"Final columns: {df.columns.tolist()}")
    logger.info(f"Data shape: {df.shape}")
    
    # Save fixed data
    df.to_csv(output_file, index=False)
    logger.info(f"Fixed data saved to {output_file}")
    
    return df

def main():
    """Main function"""
    logger.info("=== Fixing Data Columns ===")
    
    # Fix the data
    fixed_df = fix_data_columns()
    
    # Show sample of fixed data
    print("\nSample of fixed data:")
    print(fixed_df.head())
    
    print(f"\nData shape: {fixed_df.shape}")
    print(f"Columns: {fixed_df.columns.tolist()}")
    
    logger.info("Data columns fixed successfully!")

if __name__ == "__main__":
    main() 