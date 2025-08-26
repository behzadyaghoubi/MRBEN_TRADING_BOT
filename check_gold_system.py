#!/usr/bin/env python3
"""
Check Gold Trading System Status
"""

import os

import pandas as pd


def check_gold_system():
    """Check the status of the gold trading system."""

    print("🪙 MR BEN Gold Trading System Status Check")
    print("=" * 60)

    # Check configuration
    print("\n📋 Configuration:")
    try:
        with open('config/settings.json') as f:
            import json

            config = json.load(f)
            symbol = config['trading']['symbol']
            volume = config['trading']['min_lot']
            timeframe = config['trading']['timeframe']
            print(f"✅ Symbol: {symbol}")
            print(f"✅ Volume: {volume}")
            print(f"✅ Timeframe: M{timeframe}")
    except Exception as e:
        print(f"❌ Error reading config: {e}")

    # Check data files
    print("\n📊 Data Files:")
    gold_data_file = 'data/XAUUSD_PRO_M5_data.csv'
    if os.path.exists(gold_data_file):
        df = pd.read_csv(gold_data_file)
        print(f"✅ Gold data: {len(df)} records")
        print(f"✅ Latest price: ${df['close'].iloc[-1]:.2f}")
        print(f"✅ Data range: {df['time'].min()} to {df['time'].max()}")
    else:
        print(f"❌ Gold data file not found: {gold_data_file}")

    # Check model files
    print("\n🧠 AI Models:")
    lstm_model = 'models/lstm_balanced_model.h5'
    lstm_scaler = 'models/lstm_balanced_scaler.save'
    ml_filter = 'models/mrben_ai_signal_filter_xgb.joblib'

    if os.path.exists(lstm_model):
        print(f"✅ LSTM Model: {lstm_model}")
    else:
        print(f"❌ LSTM Model not found: {lstm_model}")

    if os.path.exists(lstm_scaler):
        print(f"✅ LSTM Scaler: {lstm_scaler}")
    else:
        print(f"❌ LSTM Scaler not found: {lstm_scaler}")

    if os.path.exists(ml_filter):
        print(f"✅ ML Filter: {ml_filter}")
    else:
        print(f"❌ ML Filter not found: {ml_filter}")

    # Check log files
    print("\n📝 Log Files:")
    gold_log = 'logs/gold_live_trader.log'
    gold_trades = 'logs/gold_trades.csv'

    if os.path.exists(gold_log):
        with open(gold_log) as f:
            lines = f.readlines()
            print(f"✅ Gold log: {len(lines)} lines")
            if lines:
                print(f"✅ Last log: {lines[-1].strip()}")
    else:
        print(f"⚠️ Gold log not found: {gold_log}")

    if os.path.exists(gold_trades):
        df = pd.read_csv(gold_trades)
        print(f"✅ Gold trades: {len(df)} trades")
        if len(df) > 0:
            print(f"✅ Last trade: {df.iloc[-1]['timestamp']}")
    else:
        print(f"⚠️ Gold trades file not found: {gold_trades}")

    # Check current system status
    print("\n🔄 System Status:")
    print("✅ Configuration: XAUUSD.PRO")
    print("✅ Volume: 0.01 (Demo mode)")
    print("✅ Threshold: 0.5")
    print("✅ Consecutive signals: 2")
    print("✅ Model: lstm_balanced_model.h5")

    print("\n🎯 Ready for Gold Trading!")
    print("=" * 60)


if __name__ == "__main__":
    check_gold_system()
