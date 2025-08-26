def breakout_signal(df, lookback=20):
    if len(df) < lookback+1:
        return "NO SIGNAL"
    if df['close'].iloc[-1] > df['high'].iloc[-lookback:-1].max():
        return "BUY"
    elif df['close'].iloc[-1] < df['low'].iloc[-lookback:-1].min():
        return "SELL"
    return "NO SIGNAL"