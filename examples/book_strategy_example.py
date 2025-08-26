#!/usr/bin/env python3
"""
Example usage of BookStrategy class.
Demonstrates how to use the strategy in a real trading scenario.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.strategies.book_strategy import BookStrategy


def create_realistic_market_data(symbol: str = "XAUUSD", days: int = 30) -> pd.DataFrame:
    """Create realistic market data that mimics real trading conditions."""

    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='15min')

    # Generate realistic price movement
    np.random.seed(123)  # For reproducible results

    # Start with realistic price
    base_price = 2000.0 if symbol == "XAUUSD" else 1.1000

    # Create price movement with trends and reversals
    prices = [base_price]
    trends = []

    for i in range(1, len(date_range)):
        # Add some trending behavior
        if i % 100 == 0:  # Change trend every ~25 hours
            trend = np.random.choice([-0.1, 0.1])  # Small trend
        else:
            trend = trends[-1] if trends else 0

        # Add noise and trend
        noise = np.random.normal(0, 0.05)  # Small random movement
        price_change = trend + noise

        new_price = prices[-1] * (1 + price_change / 100)
        prices.append(new_price)
        trends.append(trend)

    # Create OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(date_range, prices, strict=False)):
        # Create realistic OHLC from price
        volatility = abs(np.random.normal(0, 0.002))  # Price volatility
        high = price * (1 + volatility)
        low = price * (1 - volatility)

        # Randomize open and close
        open_price = price * (1 + np.random.normal(0, 0.0005))
        close_price = price * (1 + np.random.normal(0, 0.0005))

        # Ensure OHLC relationships
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)

        # Realistic volume (higher during active hours)
        hour = timestamp.hour
        if 8 <= hour <= 16:  # Active trading hours
            volume = np.random.randint(8000, 20000)
        else:
            volume = np.random.randint(2000, 8000)

        data.append(
            {
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'tick_volume': volume,
            }
        )

    return pd.DataFrame(data)


def demonstrate_strategy_usage():
    """Demonstrate how to use the BookStrategy in a trading scenario."""

    print("📈 BookStrategy Usage Example")
    print("=" * 50)

    # 1. Create realistic market data
    print("1. Creating realistic market data...")
    market_data = create_realistic_market_data("XAUUSD", days=7)
    print(f"   Created {len(market_data)} data points")
    print(f"   Date range: {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")
    print(f"   Price range: {market_data['close'].min():.2f} - {market_data['close'].max():.2f}")
    print()

    # 2. Initialize strategy
    print("2. Initializing BookStrategy...")
    strategy = BookStrategy()
    print(f"   Strategy: {strategy.name}")
    print(f"   Symbol: {strategy.symbol}")
    print(f"   Timeframe: {strategy.timeframe}")
    print(f"   Risk per trade: {strategy.base_risk:.1%}")
    print()

    # 3. Calculate indicators
    print("3. Calculating technical indicators...")
    df_with_indicators = strategy.calculate_indicators(market_data)

    # Show some indicator values
    latest = df_with_indicators.iloc[-1]
    print(f"   Latest RSI: {latest['rsi']:.2f}")
    print(f"   Latest MACD: {latest['macd']:.5f}")
    print(f"   Bollinger Band Width: {latest['bb_width']:.4f}")
    print(f"   Volume Ratio: {latest['volume_ratio']:.2f}")
    print()

    # 4. Generate trading signal
    print("4. Generating trading signal...")
    signal = strategy.generate_signal(df_with_indicators)

    print("   Signal Analysis:")
    print(f"   └─ Signal: {signal['signal']}")
    print(f"   └─ Confidence: {signal['confidence']:.2f}")
    print(f"   └─ Entry Price: {signal['entry_price']:.5f}")
    print(f"   └─ Stop Loss: {signal['stop_loss']:.5f}")
    print(f"   └─ Take Profit: {signal['take_profit']:.5f}")
    print(f"   └─ Risk/Reward: {signal['risk_reward_ratio']:.2f}")
    print(f"   └─ Reasons: {', '.join(signal['reasons'])}")
    print()

    # 5. Demonstrate parameter customization
    print("5. Demonstrating parameter customization...")
    custom_params = {
        'rsi_period': 21,  # Longer RSI period
        'min_confidence': 0.8,  # Higher confidence threshold
        'risk_reward_ratio': 3.0,  # Higher risk/reward ratio
    }

    custom_strategy = BookStrategy(custom_params)
    custom_signal = custom_strategy.generate_signal(df_with_indicators)

    print(
        f"   Custom Strategy Signal: {custom_signal['signal']} (confidence: {custom_signal['confidence']:.2f})"
    )
    print(f"   Custom RSI Period: {custom_strategy.parameters['rsi_period']}")
    print(f"   Custom Confidence Threshold: {custom_strategy.parameters['min_confidence']}")
    print()

    # 6. Show signal distribution over time
    print("6. Analyzing signal distribution over time...")
    df_with_signals = strategy.generate_signals(market_data)

    signal_counts = df_with_signals['signal'].value_counts()
    total_points = len(df_with_signals)

    print("   Signal Distribution:")
    for signal_type, count in signal_counts.items():
        percentage = (count / total_points) * 100
        print(f"   └─ {signal_type}: {count} ({percentage:.1f}%)")
    print()

    # 7. Demonstrate trading decision logic
    print("7. Trading decision logic...")
    if signal['signal'] == 'BUY' and signal['confidence'] >= 0.6:
        print("   ✅ BUY signal generated with sufficient confidence")
        print("   📊 Position size calculation:")
        print(f"      └─ Account risk: {strategy.base_risk:.1%}")
        print(
            f"      └─ Stop loss distance: {abs(signal['entry_price'] - signal['stop_loss']):.5f}"
        )
        print("      └─ Risk per pip: Calculate based on lot size")
    elif signal['signal'] == 'SELL' and signal['confidence'] >= 0.6:
        print("   ✅ SELL signal generated with sufficient confidence")
        print("   📊 Position size calculation:")
        print(f"      └─ Account risk: {strategy.base_risk:.1%}")
        print(
            f"      └─ Stop loss distance: {abs(signal['entry_price'] - signal['stop_loss']):.5f}"
        )
    else:
        print("   ⏸️  No trade signal (HOLD) or insufficient confidence")
        print(f"   📊 Confidence threshold: {strategy.parameters['min_confidence']}")

    print()
    print("✅ BookStrategy example completed successfully!")


def demonstrate_backtesting_scenario():
    """Demonstrate a backtesting scenario with the strategy."""

    print("\n🔄 Backtesting Scenario Example")
    print("=" * 50)

    # Create historical data
    historical_data = create_realistic_market_data("XAUUSD", days=14)

    # Initialize strategy
    strategy = BookStrategy()

    # Simulate backtesting
    signals = []
    trades = []

    print("Simulating backtesting over historical data...")

    for i in range(50, len(historical_data)):  # Start from 50 to have enough data for indicators
        window = historical_data.iloc[: i + 1]
        signal = strategy.generate_signal(strategy.calculate_indicators(window))

        if signal['signal'] in ['BUY', 'SELL'] and signal['confidence'] >= 0.6:
            signals.append(signal)

            # Simulate trade
            trade = {
                'timestamp': signal['timestamp'],
                'signal': signal['signal'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'confidence': signal['confidence'],
                'reasons': signal['reasons'],
            }
            trades.append(trade)

    print(f"   Generated {len(signals)} signals")
    print(f"   Executed {len(trades)} trades")

    if trades:
        print("\n   Recent Trades:")
        for i, trade in enumerate(trades[-3:], 1):  # Show last 3 trades
            print(
                f"   {i}. {trade['signal']} at {trade['entry_price']:.5f} "
                f"(confidence: {trade['confidence']:.2f})"
            )

    print("\n✅ Backtesting example completed!")


if __name__ == "__main__":
    try:
        demonstrate_strategy_usage()
        demonstrate_backtesting_scenario()
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
