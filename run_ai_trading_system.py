import joblib
import pandas as pd

# لود سیگنال‌ها و دیتا
df = pd.read_csv('ohlc_lstm_signals.csv')

# فیچرهای ML
df['RSI'] = (
    df['close']
    .rolling(14)
    .apply(
        lambda x: (
            (100 - (100 / (1 + (x.diff().clip(lower=0).sum() / abs(x.diff().clip(upper=0)).sum()))))
            if abs(x.diff().clip(upper=0)).sum() != 0
            else 0
        ),
        raw=False,
    )
)
df['SMA20'] = df['close'].rolling(20).mean()
df['MACD'] = (
    df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
)
df.dropna(inplace=True)

# لود مدل ML
ml_model = joblib.load('ml_signal_filter_xgb.joblib')

# محاسبه ML Filter
features = df[['close', 'lstm_proba', 'RSI', 'MACD', 'SMA20']].values
ml_pred = ml_model.predict(features)
df['ml_signal'] = ml_pred

# ترکیب نهایی سیگنال: فقط اگر هر دو سیگنال LSTM و ML، هردو ۱ باشند، ترید بزن!
df['final_signal'] = ((df['lstm_signal'] == 1) & (df['ml_signal'] == 1)).astype(int)

# بک‌تست خیلی ساده (می‌تونی با استاپ/تی‌پی یا روش پیشرفته‌تر جایگزین کنی)
capital = 10000
equity_curve = [capital]
for i in range(1, len(df)):
    if df['final_signal'].iloc[i - 1] == 1:
        # اگر سیگنال BUY بود
        ret = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
        capital *= 1 + ret
    equity_curve.append(capital)
df['equity'] = equity_curve

df.to_csv('ai_trading_results.csv', index=False)
print("✅ اجرای سیستم تریدینگ هوشمند تمام شد! نتایج در ai_trading_results.csv ذخیره شد.")
print(
    f"Max Equity: {df['equity'].max()}\nMin Equity: {df['equity'].min()}\nFinal Equity: {df['equity'].iloc[-1]}"
)
