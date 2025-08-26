def mean_reversion_strategy(df, lookback=20, entry_threshold=2.0):
    """
    Mean Reversion Strategy inspired by Ernest Chan
    Enter long if price is below mean - threshold * std deviation
    Enter short if price is above mean + threshold * std deviation
    """
    df['mean'] = df['close'].rolling(lookback).mean()
    df['std'] = df['close'].rolling(lookback).std()

    current_price = df['close'].iloc[-1]
    mean = df['mean'].iloc[-1]
    std = df['std'].iloc[-1]

    if current_price < mean - entry_threshold * std:
        signal = 1  # Buy
    elif current_price > mean + entry_threshold * std:
        signal = -1  # Sell
    else:
        signal = 0  # Hold

    return signal
