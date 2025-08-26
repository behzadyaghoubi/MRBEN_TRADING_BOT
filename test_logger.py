from logger.trade_logger import TradeLogger

logger = TradeLogger()

features = {
    "RSI": 55.2,
    "MACD": 0.015,
    "ATR": 2.9,
    "Volume": 132000
}

logger.log_trade(
    symbol="XAUUSD",
    signal="BUY",
    predicted_confidence=0.89,
    actual_result="WIN",
    entry_price=2410.5,
    exit_price=2422.1,
    profit=11.6,
    feature_dict=features
)

print("✅ Trade logged successfully.")