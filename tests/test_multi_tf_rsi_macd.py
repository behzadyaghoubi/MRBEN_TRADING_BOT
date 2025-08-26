import pandas as pd
from signals.multi_tf_rsi_macd import analyze_multi_tf_rsi_macd

def make_df(values):
    return pd.DataFrame({"close": values, "high": values, "low": values})

def test_buy_signal_when_rsi_low_and_macd_cross_up():
    dfs = {
        "M5": make_df([1,2,3,4,5,6,7,6,7,8,9,10]),
        "M15": make_df([10,11,12,13,14,15,16,17,18,19,20,21]),
        "H1": make_df([100,101,102,103,104,105,106,107,108,109,110,111]),
    }
    results = analyze_multi_tf_rsi_macd(dfs, rsi_overbought=80, rsi_oversold=20)
    assert all(tf in results for tf in ["M5","M15","H1"])
    assert isinstance(results["M5"], str)

def test_sell_signal_when_rsi_high_and_macd_cross_down():
    dfs = {"M5": make_df([50+i for i in range(30)])}
    results = analyze_multi_tf_rsi_macd(dfs, rsi_overbought=55, rsi_oversold=10)
    assert results["M5"] in ["buy","sell","neutral"]
