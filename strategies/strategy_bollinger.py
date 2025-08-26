import talib

def bollinger_signal(df, period=20):
    if len(df) < period:
        return "NO SIGNAL"
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=period)
    close = df['close'].iloc[-1]
    if close < lower.iloc[-1]:
        return "BUY"
    elif close > upper.iloc[-1]:
        return "SELL"
    return "NO SIGNAL"