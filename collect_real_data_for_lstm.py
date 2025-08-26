#!/usr/bin/env python3
"""
Collect Real Market Data for LSTM Training
Get real XAUUSD.PRO data and create balanced dataset
"""

import json
import os
import time
from multiprocessing import Process, Queue

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import talib


def load_config():
    """Load MT5 configuration from settings.json."""
    try:
        with open('config/settings.json', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"⚠️ Could not load config: {e}")
        return None


def initialize_mt5_with_retry(max_attempts=5, timeout=10000):
    """Initialize MT5 connection with retry logic."""
    config = load_config()

    for attempt in range(max_attempts):
        print(f"🔄 MT5 connection attempt {attempt + 1}/{max_attempts}")

        try:
            # Initialize MT5 with timeout
            if not mt5.initialize(timeout=timeout):
                print(f"❌ MT5 initialization failed (attempt {attempt + 1})")
                if attempt < max_attempts - 1:
                    time.sleep(2)
                    continue
                return False

            # Login with config or defaults
            if config:
                login = config.get('mt5_login', 1104123)
                password = config.get('mt5_password', '-4YcBgRd')
                server = config.get('mt5_server', 'OxSecurities-Demo')
            else:
                login = 1104123
                password = '-4YcBgRd'
                server = 'OxSecurities-Demo'

            if not mt5.login(login=login, password=password, server=server):
                print(f"❌ MT5 login failed (attempt {attempt + 1})")
                if attempt < max_attempts - 1:
                    time.sleep(2)
                    continue
                return False

            print("✅ MT5 connected successfully")
            return True

        except Exception as e:
            print(f"❌ MT5 connection error (attempt {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(2)
                continue

    print("❌ Failed to connect to MT5 after all attempts")
    return False


def get_market_data(symbol="XAUUSD.PRO", timeframe=mt5.TIMEFRAME_M5, bars=10000):
    """Get real market data from MT5."""
    print(f"📊 Getting {bars} bars of {symbol} data...")

    try:
        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to select symbol {symbol}")
            return None

        # Get historical data
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

        if rates is None or len(rates) == 0:
            print("❌ Failed to get market data")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        print(f"✅ Got {len(df)} bars of data")
        print(f"   Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    except Exception as e:
        print(f"❌ Error getting market data: {e}")
        return None


def add_technical_indicators(df):
    """Add technical indicators to the data."""
    print("🔧 Adding technical indicators...")

    try:
        # RSI
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
        df['macd'] = macd

        # ATR
        df['atr'] = talib.ATR(
            df['high'].values, df['low'].values, df['close'].values, timeperiod=14
        )

        # Remove NaN values
        df = df.dropna()

        print(f"✅ Added indicators, {len(df)} rows remaining")
        return df

    except Exception as e:
        print(f"❌ Error adding indicators: {e}")
        return None


def create_sequences_with_labels(df, timesteps=50):
    """Create sequences and labels based on price movement."""
    print(f"🔧 Creating sequences with {timesteps} timesteps...")

    try:
        sequences = []
        labels = []

        # Features to use (7 features as required)
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd']

        for i in range(timesteps, len(df) - 1):
            # Get sequence
            sequence = df[features].iloc[i - timesteps : i].values
            sequences.append(sequence)

            # Calculate future price movement
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + 1]
            price_change = (future_price - current_price) / current_price * 100

            # Label based on price movement
            if price_change > 0.1:  # More than 0.1% increase
                label = 2  # BUY
            elif price_change < -0.1:  # More than 0.1% decrease
                label = 0  # SELL
            else:
                label = 1  # HOLD

            labels.append(label)

        sequences = np.array(sequences)
        labels = np.array(labels)

        print(f"✅ Created {len(sequences)} sequences")
        print(f"   BUY signals: {np.sum(labels == 2)}")
        print(f"   SELL signals: {np.sum(labels == 0)}")
        print(f"   HOLD signals: {np.sum(labels == 1)}")

        return sequences, labels

    except Exception as e:
        print(f"❌ Error creating sequences: {e}")
        return None, None


def balance_dataset(sequences, labels, samples_per_class=1000):
    """Balance the dataset by undersampling majority classes."""
    print(f"⚖️ Balancing dataset to {samples_per_class} samples per class...")

    try:
        balanced_sequences = []
        balanced_labels = []

        for class_id in [0, 1, 2]:  # SELL, HOLD, BUY
            class_indices = np.where(labels == class_id)[0]

            if len(class_indices) >= samples_per_class:
                # Randomly sample
                selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
            else:
                # Use all available samples
                selected_indices = class_indices
                print(f"⚠️ Class {class_id} has only {len(class_indices)} samples")

            balanced_sequences.extend(sequences[selected_indices])
            balanced_labels.extend([class_id] * len(selected_indices))

        balanced_sequences = np.array(balanced_sequences)
        balanced_labels = np.array(balanced_labels)

        # Shuffle
        indices = np.random.permutation(len(balanced_sequences))
        balanced_sequences = balanced_sequences[indices]
        balanced_labels = balanced_labels[indices]

        print("✅ Balanced dataset created")
        print(f"   Total samples: {len(balanced_sequences)}")
        print(f"   BUY signals: {np.sum(balanced_labels == 2)}")
        print(f"   SELL signals: {np.sum(balanced_labels == 0)}")
        print(f"   HOLD signals: {np.sum(balanced_labels == 1)}")

        return balanced_sequences, balanced_labels

    except Exception as e:
        print(f"❌ Error balancing dataset: {e}")
        return None, None


def save_dataset(sequences, labels, filename_prefix="real_market"):
    """Save the dataset."""
    print("💾 Saving dataset...")

    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Save sequences and labels
        np.save(f"data/{filename_prefix}_sequences.npy", sequences)
        np.save(f"data/{filename_prefix}_labels.npy", labels)

        print(f"✅ Dataset saved to data/{filename_prefix}_*.npy")

        return f"data/{filename_prefix}_sequences.npy", f"data/{filename_prefix}_labels.npy"

    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
        return None, None


def collect_data_worker(queue):
    """Worker function to collect data in separate process."""
    try:
        print("🎯 Starting data collection worker...")

        # Initialize MT5 in worker process
        if not initialize_mt5_with_retry():
            queue.put(("error", "Failed to initialize MT5"))
            return

        # Get market data
        df = get_market_data()
        if df is None:
            queue.put(("error", "Failed to get market data"))
            return

        # Add technical indicators
        df = add_technical_indicators(df)
        if df is None:
            queue.put(("error", "Failed to add indicators"))
            return

        # Create sequences with labels
        sequences, labels = create_sequences_with_labels(df)
        if sequences is None:
            queue.put(("error", "Failed to create sequences"))
            return

        # Balance dataset
        balanced_sequences, balanced_labels = balance_dataset(sequences, labels)
        if balanced_sequences is None:
            queue.put(("error", "Failed to balance dataset"))
            return

        # Save dataset
        sequences_path, labels_path = save_dataset(balanced_sequences, balanced_labels)
        if sequences_path is None:
            queue.put(("error", "Failed to save dataset"))
            return

        # Send success result
        result = {
            "sequences_path": sequences_path,
            "labels_path": labels_path,
            "shape": balanced_sequences.shape,
            "buy_signals": int(np.sum(balanced_labels == 2)),
            "sell_signals": int(np.sum(balanced_labels == 0)),
            "hold_signals": int(np.sum(balanced_labels == 1)),
        }

        queue.put(("success", result))

    except Exception as e:
        queue.put(("error", f"Worker error: {e}"))
    finally:
        # Don't shutdown MT5 here to avoid affecting other processes
        pass


def main():
    """Main function."""
    print("🎯 Collecting Real Market Data for LSTM Training")
    print("=" * 60)

    # Use multiprocessing to avoid MT5 connection issues
    queue = Queue()
    process = Process(target=collect_data_worker, args=(queue,))

    try:
        print("🚀 Starting data collection process...")
        process.start()

        # Wait for result with timeout
        try:
            result_type, result_data = queue.get(timeout=300)  # 5 minutes timeout

            if result_type == "success":
                print("\n✅ Real market data collection completed!")
                print("\n📝 Summary:")
                print(f"   Sequences: {result_data['sequences_path']}")
                print(f"   Labels: {result_data['labels_path']}")
                print(f"   Shape: {result_data['shape']}")
                print(f"   BUY signals: {result_data['buy_signals']}")
                print(f"   SELL signals: {result_data['sell_signals']}")
                print(f"   HOLD signals: {result_data['hold_signals']}")
                print("   Classes: SELL(0), HOLD(1), BUY(2)")

                print("\n🎯 Next Steps:")
                print("   1. Use this dataset to retrain LSTM model")
                print("   2. Test with real market patterns")
                print("   3. Verify balanced signal generation")

            else:
                print(f"❌ Data collection failed: {result_data}")

        except Exception as e:
            print(f"❌ Timeout or error waiting for result: {e}")
            process.terminate()

    except Exception as e:
        print(f"❌ Error starting process: {e}")

    finally:
        # Clean up
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

        print("🏁 Data collection process finished")


if __name__ == "__main__":
    main()
