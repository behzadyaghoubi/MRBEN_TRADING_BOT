def prepare_lstm_input(df, lookback=60):
    if len(df) < lookback:
        return None

    features = [
        'open',
        'high',
        'low',
        'close',
        'tick_volume',
        'sma_5',
        'sma_10',
        'sma_20',
        'sma_50',
        'rsi_14',
        'macd',
        'macd_signal',
        'upper_band',
        'lower_band',
        'atr',
        'willr',
        'cci',
        'adx',
        'ema_5',
        'ema_10',
        'ema_20',
        'momentum',
        'obv',
    ]

    df = df.copy()
    df = df[features].dropna()

    if len(df) < lookback:
        return None

    X = df[-lookback:].values.reshape(1, lookback, len(features))
    return X
