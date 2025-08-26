def calculate_atr(df, period=14):
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = abs(df["high"] - df["close"].shift(1))
    df["L-PC"] = abs(df["low"] - df["close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=period).mean()
    return df["ATR"]


def get_sl_tp(signal, entry_price, atr, multiplier=2.0):
    sl, tp = None, None
    if signal == "BUY":
        sl = entry_price - (atr * multiplier)
        tp = entry_price + (atr * multiplier)
    elif signal == "SELL":
        sl = entry_price + (atr * multiplier)
        tp = entry_price - (atr * multiplier)
    return round(sl, 3), round(tp, 3)
