def detect_pin_bar(df, n=5):
    if len(df) < n:
        return "NO PA"
    row = df.iloc[-1]
    body = abs(row['close'] - row['open'])
    total = row['high'] - row['low']
    up_shadow = row['high'] - max(row['close'], row['open'])
    down_shadow = min(row['close'], row['open']) - row['low']

    if body < total * 0.3:
        if up_shadow > total * 0.6:
            return "BEARISH_PIN"
        elif down_shadow > total * 0.6:
            return "BULLISH_PIN"
    return "NO PA"


def detect_engulfing(df, n=5):
    if len(df) < n + 1:
        return "NO PA"
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    if prev['close'] < prev['open'] and curr['close'] > curr['open']:
        if curr['close'] > prev['open'] and curr['open'] < prev['close']:
            return "BULLISH_ENGULF"
    if prev['close'] > prev['open'] and curr['close'] < curr['open']:
        if curr['open'] > prev['close'] and curr['close'] < prev['open']:
            return "BEARISH_ENGULF"
    return "NO PA"


def price_action_signal(df):
    pin = detect_pin_bar(df)
    eng = detect_engulfing(df)
    if pin == "BULLISH_PIN" or eng == "BULLISH_ENGULF":
        return "BUY"
    elif pin == "BEARISH_PIN" or eng == "BEARISH_ENGULF":
        return "SELL"
    else:
        return "NO PA"
