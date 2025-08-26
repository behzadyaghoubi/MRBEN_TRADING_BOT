import pandas as pd
import numpy as np
import talib
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def detect_price_action(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects Pin Bar and Engulfing candlestick patterns.
    Adds 'pinbar' and 'engulfing' columns to the DataFrame.
    """
    body = abs(df['open'] - df['close'])
    candle_range = df['high'] - df['low']
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    pinbar = (
        (body < candle_range * 0.3) &
        (upper_wick > candle_range * 0.4) &
        (lower_wick > candle_range * 0.4)
    )
    df['pinbar'] = pinbar.astype(int)
    engulfing = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    df['engulfing'] = np.where(engulfing > 0, 1, np.where(engulfing < 0, -1, 0))
    return df

def generate_book_signals(
    df: pd.DataFrame,
    sma_fast: int = 20,
    sma_slow: int = 50,
    rsi_period: int = 14,
    rsi_buy: float = 45,
    rsi_sell: float = 55,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    min_strength: int = 2,
    custom_thresholds: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Generate trading signals for XAUUSD M15 using SMA, RSI, MACD, and price action.
    Returns DataFrame with 'signal' and 'signal_strength' columns.
    - min_strength: Minimum score to accept a signal (default: 2)
    - custom_thresholds: dict for custom thresholds (e.g. {'macd_hist': 0.15, 'rsi_dist': 12})
    """
    df = df.copy()
    df['SMA_FAST'] = talib.SMA(df['close'], timeperiod=sma_fast)
    df['SMA_SLOW'] = talib.SMA(df['close'], timeperiod=sma_slow)
    df['RSI'] = talib.RSI(df['close'], timeperiod=rsi_period)
    macd, macd_sig, macd_hist = talib.MACD(df['close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
    df['MACD'] = macd
    df['MACD_signal'] = macd_sig
    df['MACD_hist'] = macd_hist
    df = detect_price_action(df)

    macd_hist_thr = 0.15 if not custom_thresholds else custom_thresholds.get('macd_hist', 0.15)
    rsi_dist_thr = 12 if not custom_thresholds else custom_thresholds.get('rsi_dist', 12)

    prev_sma_fast = df['SMA_FAST'].shift(1)
    prev_sma_slow = df['SMA_SLOW'].shift(1)
    prev_rsi = df['RSI'].shift(1)

    buy_cond = (
        (df['SMA_FAST'] > df['SMA_SLOW']) &
        (prev_sma_fast <= prev_sma_slow) &
        (df['RSI'] < rsi_buy) &
        (df['RSI'] > prev_rsi) &
        ((df['pinbar'] == 1) | (df['engulfing'] == 1))
    )
    sell_cond = (
        (df['SMA_FAST'] < df['SMA_SLOW']) &
        (prev_sma_fast >= prev_sma_slow) &
        (df['RSI'] > rsi_sell) &
        (df['RSI'] < prev_rsi) &
        ((df['pinbar'] == 1) | (df['engulfing'] == -1))
    )

    strength = (
        (abs(df['SMA_FAST'] - df['SMA_SLOW']) > 0.1 * df['close']).astype(int) +
        (abs(df['RSI'] - 50) > rsi_dist_thr).astype(int) +
        (abs(df['MACD_hist']) > macd_hist_thr).astype(int) +
        ((df['pinbar'] == 1) | (abs(df['engulfing']) == 1)).astype(int)
    )

    df['signal'] = np.where(buy_cond, 'BUY', np.where(sell_cond, 'SELL', 'HOLD'))
    df['signal_strength'] = np.where((buy_cond | sell_cond), strength, 0)

    df.loc[df['signal_strength'] < min_strength, 'signal'] = 'HOLD'
    df.loc[df['signal_strength'] < min_strength, 'signal_strength'] = 0

    return df

def get_latest_signal(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Get the latest signal for live trading (returns dict).
    """
    df = generate_book_signals(df, **kwargs)
    if len(df) == 0:
        return {"signal": "HOLD", "strength": 0}
    last = df.iloc[-1]
    return {
        "signal": last["signal"],
        "strength": last["signal_strength"],
        "row": last.to_dict()
    }

def plot_signals(df: pd.DataFrame, filename: str = 'signals_plot.png'):
    """
    Plot OHLC and signals for visual inspection.
    Requires matplotlib.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['time'], df['close'], label='Close', color='black', linewidth=1)
    ax.plot(df['time'], df['SMA_FAST'], label='SMA_FAST', color='blue', alpha=0.7)
    ax.plot(df['time'], df['SMA_SLOW'], label='SMA_SLOW', color='red', alpha=0.7)

    buy_signals = df[df['signal'] == 'BUY']
    sell_signals = df[df['signal'] == 'SELL']
    ax.scatter(buy_signals['time'], buy_signals['close'], marker='^', color='green', label='BUY', s=60, zorder=5)
    ax.scatter(sell_signals['time'], sell_signals['close'], marker='v', color='red', label='SELL', s=60, zorder=5)

    ax.set_title('Book Strategy Signals')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

if __name__ == "__main__":
    try:
        df = pd.read_csv("ohlc_data.csv")
        df = generate_book_signals(df)
        print(df.tail(5)[["time", "open", "close", "SMA_FAST", "SMA_SLOW", "RSI", "pinbar", "engulfing", "signal", "signal_strength"]])
        df.to_csv("signals_book_strategy.csv", index=False)
        print("✅ Signals generated and saved successfully.")
        print("Latest signal:", get_latest_signal(df))
        plot_signals(df)
    except Exception as e:
        logging.error(f"⛔️ Error: {e}")