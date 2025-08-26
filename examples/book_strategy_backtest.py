#!/usr/bin/env python3
"""
Backtest BookStrategy on 1 month of XAUUSD M15 data.
Generates signals, simulates trades, and reports performance metrics.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.strategies.book_strategy import BookStrategy
from src.config.settings import settings

# 1. Generate 1 month of realistic XAUUSD M15 OHLCV data
def generate_xauusd_m15_data(days=31):
    np.random.seed(42)
    periods = days * 24 * 4  # 4 bars per hour
    base_price = 2000.0
    timestamps = [datetime.now() - timedelta(minutes=15*(periods-i)) for i in range(periods)]
    prices = [base_price]
    for _ in range(1, periods):
        change = np.random.normal(0, 0.25)  # ~0.25% stddev per bar
        prices.append(prices[-1] * (1 + change/100))
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.0015)))
        low = price * (1 - abs(np.random.normal(0, 0.0015)))
        open_ = price * (1 + np.random.normal(0, 0.0005))
        close = price * (1 + np.random.normal(0, 0.0005))
        volume = np.random.randint(5000, 20000)
        high = max(high, open_, close)
        low = min(low, open_, close)
        data.append({
            'timestamp': timestamps[i],
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'tick_volume': volume
        })
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# 2. Run BookStrategy to generate signals
def run_backtest(df, strategy):
    trades = []
    equity_curve = []
    balance = 100_000.0
    open_trade = None
    max_equity = balance
    drawdowns = []
    for i in range(50, len(df)):
        window = df.iloc[:i+1]
        signal = strategy.generate_signal(window)
        bar = df.iloc[i]
        # Close trade if open
        if open_trade:
            # Check SL/TP
            if open_trade['side'] == 'BUY':
                if bar['low'] <= open_trade['stop_loss']:
                    exit_price = open_trade['stop_loss']
                    result = 'SL'
                elif bar['high'] >= open_trade['take_profit']:
                    exit_price = open_trade['take_profit']
                    result = 'TP'
                else:
                    equity_curve.append(balance)
                    continue
            else:  # SELL
                if bar['high'] >= open_trade['stop_loss']:
                    exit_price = open_trade['stop_loss']
                    result = 'SL'
                elif bar['low'] <= open_trade['take_profit']:
                    exit_price = open_trade['take_profit']
                    result = 'TP'
                else:
                    equity_curve.append(balance)
                    continue
            # Close trade
            pl = (exit_price - open_trade['entry']) if open_trade['side']=='BUY' else (open_trade['entry'] - exit_price)
            pl = pl * 10  # 1 lot, $10 per XAUUSD pip (0.1)
            balance += pl
            trades.append({
                'entry_time': open_trade['entry_time'],
                'exit_time': bar['timestamp'],
                'side': open_trade['side'],
                'entry': open_trade['entry'],
                'exit': exit_price,
                'stop_loss': open_trade['stop_loss'],
                'take_profit': open_trade['take_profit'],
                'result': result,
                'pl': pl,
                'risk_reward': open_trade['risk_reward'],
                'confidence': open_trade['confidence'],
                'reasons': open_trade['reasons']
            })
            open_trade = None
            equity_curve.append(balance)
            max_equity = max(max_equity, balance)
            drawdowns.append(max_equity - balance)
        # Open new trade if no trade is open
        if not open_trade and signal['signal'] in ['BUY', 'SELL'] and signal['confidence'] >= 0.6:
            open_trade = {
                'side': signal['signal'],
                'entry': bar['close'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'entry_time': bar['timestamp'],
                'risk_reward': signal['risk_reward_ratio'],
                'confidence': signal['confidence'],
                'reasons': signal['reasons']
            }
        equity_curve.append(balance)
        max_equity = max(max_equity, balance)
        drawdowns.append(max_equity - balance)
    return trades, equity_curve, drawdowns

def summarize_results(trades, equity_curve, drawdowns):
    total_trades = len(trades)
    wins = [t for t in trades if t['result']=='TP']
    losses = [t for t in trades if t['result']=='SL']
    win_rate = 100 * len(wins) / total_trades if total_trades else 0
    avg_rr = np.mean([t['risk_reward'] for t in trades]) if trades else 0
    total_pl = sum(t['pl'] for t in trades)
    max_dd = max(drawdowns) if drawdowns else 0
    final_balance = equity_curve[-1] if equity_curve else 0
    return {
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Average R/R': avg_rr,
        'Total Profit/Loss': total_pl,
        'Final Balance': final_balance,
        'Max Drawdown': max_dd
    }, wins, losses

def plot_equity_curve(equity_curve):
    plt.figure(figsize=(10,5))
    plt.plot(equity_curve, label='Equity Curve')
    plt.title('BookStrategy Backtest Equity Curve')
    plt.xlabel('Trade/Bar')
    plt.ylabel('Balance ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_summary_table(summary, wins, losses):
    print("\nBacktest Summary:")
    print("================")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:,.2f}")
        else:
            print(f"{k:20}: {v}")
    print(f"Profitable Trades : {len(wins)}")
    print(f"Losing Trades     : {len(losses)}")
    print()
    if wins:
        print("Sample Winning Trade:")
        print(wins[0])
    if losses:
        print("Sample Losing Trade:")
        print(losses[0])

if __name__ == "__main__":
    print("Generating XAUUSD M15 data...")
    df = generate_xauusd_m15_data(days=31)
    print(f"Data points: {len(df)}")
    print("Initializing BookStrategy...")
    strategy = BookStrategy()
    print("Running backtest...")
    trades, equity_curve, drawdowns = run_backtest(df, strategy)
    summary, wins, losses = summarize_results(trades, equity_curve, drawdowns)
    print_summary_table(summary, wins, losses)
    plot_equity_curve(equity_curve) 