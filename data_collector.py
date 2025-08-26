# fetch_mt5_history.py - FINAL PROFESSIONAL VERSION

import argparse
import json
import os
import time

import MetaTrader5 as mt5
import pandas as pd


def load_settings(config_path="settings.json"):
    """Load MT5 credentials/settings from a JSON config file."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"{config_path} not found.")


def fetch_mt5_data(login, password, server, symbol, timeframe, n_candles, out_csv=None, retries=3):
    """
    Connects to MT5, fetches n_candles for given symbol/timeframe.
    Returns DataFrame. Optionally saves to CSV.
    """
    for attempt in range(retries):
        if not mt5.initialize(login=login, password=password, server=server):
            print(f"[ERROR] Try {attempt+1}: Could not connect: {mt5.last_error()}")
            time.sleep(2)
            continue
        else:
            break
    else:
        print("[ERROR] Could not connect to MetaTrader 5 after retries.")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        print("[WARN] Not enough data received or symbol not found.")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[OK] Data saved to {out_csv}. Candles: {len(df)}")
    return df


# Timeframe mapping for flexibility
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV data from MetaTrader 5")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol (e.g. XAUUSD)")
    parser.add_argument("--timeframe", type=str, default=None, help="Timeframe (e.g. M15, H1, D1)")
    parser.add_argument("--n", type=int, default=5000, help="Number of candles")
    parser.add_argument("--out", type=str, default=None, help="Output CSV file name")
    parser.add_argument("--config", type=str, default="settings.json", help="Settings file")
    args = parser.parse_args()

    config = load_settings(args.config)
    login = config.get("login")
    password = config.get("password")
    server = config.get("server")
    if not all([login, password, server]):
        print("[ERROR] Please set login, password, and server in your config file.")
        exit(1)

    symbol = args.symbol or config.get("symbol", "XAUUSD")
    timeframe = args.timeframe or config.get("timeframe", "M15")
    tf_val = TF_MAP.get(timeframe.upper(), mt5.TIMEFRAME_M15)
    out_file = args.out or f"{symbol}_{timeframe}_history.csv"

    fetch_mt5_data(login, password, server, symbol, tf_val, args.n, out_csv=out_file)
