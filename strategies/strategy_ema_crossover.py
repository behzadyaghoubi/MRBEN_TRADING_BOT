import talib

def ema_crossover_signal(df, fast=20, slow=50):
    df = df.copy()
    df['EMA_FAST'] = talib.EMA(df['close'], timeperiod=fast)
    df['EMA_SLOW'] = talib.EMA(df['close'], timeperiod=slow)
    if len(df) < 2:
        return "NO SIGNAL"
    if df['EMA_FAST'].iloc[-1] > df['EMA_SLOW'].iloc[-1] and df['EMA_FAST'].iloc[-2] <= df['EMA_SLOW'].iloc[-2]:
        return "BUY"
    elif df['EMA_FAST'].iloc[-1] < df['EMA_SLOW'].iloc[-1] and df['EMA_FAST'].iloc[-2] >= df['EMA_SLOW'].iloc[-2]:
        return "SELL"
    return "NO SIGNAL"