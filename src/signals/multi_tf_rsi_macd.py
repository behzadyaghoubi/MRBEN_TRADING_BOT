from __future__ import annotations

from typing import Literal

import pandas as pd

from src.indicators.rsi_macd import compute_macd, compute_rsi

Side = Literal["buy", "sell", "neutral"]


def analyze_multi_tf_rsi_macd(
    dfs: dict[str, pd.DataFrame],
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_overbought: int = 70,
    rsi_oversold: int = 30,
) -> dict[str, Side]:
    """
    Analyze RSI/MACD across multiple timeframes.
    dfs: mapping timeframe->DataFrame with 'close','high','low'
    Returns dict timeframe->signal: "buy","sell","neutral".
    """
    results: dict[str, Side] = {}

    for tf, df in dfs.items():
        rsi = compute_rsi(df, rsi_period)
        macd_line, signal_line, hist = compute_macd(df, macd_fast, macd_slow, macd_signal)

        last_rsi = rsi.iloc[-1]
        last_macd = macd_line.iloc[-1]
        last_signal = signal_line.iloc[-1]

        decision: Side = "neutral"
        if last_rsi < rsi_oversold and last_macd > last_signal:
            decision = "buy"
        elif last_rsi > rsi_overbought and last_macd < last_signal:
            decision = "sell"

        results[tf] = decision

    return results
