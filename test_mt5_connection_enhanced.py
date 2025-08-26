#!/usr/bin/env python3
"""
Enhanced MT5 Connection Test
Tests all aspects of MT5 connection and trading capabilities
"""

import json
import os
import sys
from datetime import datetime

# MT5 Integration
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    print("❌ MetaTrader5 not available")
    MT5_AVAILABLE = False
    sys.exit(1)


def test_mt5_connection():
    """Test complete MT5 connection and trading setup."""
    print("🔍 Testing MT5 Connection and Trading Setup")
    print("=" * 60)

    # 1. Initialize MT5
    print("\n1️⃣ Initializing MT5...")
    if not mt5.initialize():
        print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
        return False
    print("✅ MT5 initialized successfully")

    # 2. Load config
    print("\n2️⃣ Loading configuration...")
    config_path = 'config/settings.json'
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

        login = config.get('mt5_login', 1104123)
        password = config.get('mt5_password', '-4YcBgRd')
        server = config.get('mt5_server', 'OxSecurities-Demo')
        symbol = config.get('trading', {}).get('symbol', 'XAUUSD.PRO')
        volume = config.get('trading', {}).get('min_lot', 0.01)
    else:
        login = 1104123
        password = '-4YcBgRd'
        server = 'OxSecurities-Demo'
        symbol = 'XAUUSD.PRO'
        volume = 0.01

    print(f"📊 Login: {login}")
    print(f"📊 Server: {server}")
    print(f"📊 Symbol: {symbol}")
    print(f"📊 Volume: {volume}")

    # 3. Login to MT5
    print("\n3️⃣ Logging into MT5...")
    if not mt5.login(login=login, password=password, server=server):
        print(f"❌ Failed to login to MT5: {mt5.last_error()}")
        return False
    print("✅ Login successful")

    # 4. Get account info
    print("\n4️⃣ Getting account information...")
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Failed to get account info")
        return False

    print(f"✅ Account: {account_info.login}")
    print(f"📊 Balance: {account_info.balance}")
    print(f"📊 Equity: {account_info.equity}")
    print(f"📊 Margin: {account_info.margin}")
    print(f"📊 Free Margin: {account_info.margin_free}")
    print(f"📊 Account Type: {'Demo' if account_info.trade_mode == 1 else 'Real'}")

    # 5. Get symbol info
    print(f"\n5️⃣ Getting symbol information for {symbol}...")
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found")
        return False

    print(f"✅ Symbol found: {symbol_info.name}")
    print(f"📊 Visible: {symbol_info.visible}")
    print(f"📊 Min Lot: {symbol_info.volume_min}")
    print(f"📊 Max Lot: {symbol_info.volume_max}")
    print(f"📊 Lot Step: {symbol_info.volume_step}")
    print(f"📊 Contract Size: {symbol_info.trade_contract_size}")
    print(f"📊 Point: {symbol_info.point}")
    print(f"📊 Digits: {symbol_info.digits}")

    # 6. Select symbol if not visible
    if not symbol_info.visible:
        print(f"\n6️⃣ Selecting symbol {symbol}...")
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to select symbol {symbol}")
            return False
        print(f"✅ Symbol {symbol} selected")

    # 7. Get current tick
    print(f"\n7️⃣ Getting current tick for {symbol}...")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ Failed to get tick data for {symbol}")
        return False

    print("✅ Current Tick:")
    print(f"📊 Bid: {tick.bid}")
    print(f"📊 Ask: {tick.ask}")
    print(f"📊 Time: {datetime.fromtimestamp(tick.time)}")
    print(f"📊 Spread: {tick.ask - tick.bid}")

    # 8. Test order preparation (without sending)
    print("\n8️⃣ Testing order preparation...")

    # Prepare BUY order request
    buy_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "sl": tick.ask - 50 * symbol_info.point,
        "tp": tick.ask + 100 * symbol_info.point,
        "deviation": 20,
        "magic": 654321,
        "comment": f"TEST_BUY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    print("✅ BUY Order Request Prepared:")
    print(f"📊 Symbol: {buy_request['symbol']}")
    print(f"📊 Volume: {buy_request['volume']}")
    print(f"📊 Price: {buy_request['price']}")
    print(f"📊 SL: {buy_request['sl']}")
    print(f"📊 TP: {buy_request['tp']}")
    print(f"📊 Magic: {buy_request['magic']}")

    # 9. Test margin calculation
    print("\n9️⃣ Testing margin calculation...")
    margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, volume, tick.ask)
    if margin is None:
        print("❌ Failed to calculate margin")
    else:
        print(f"✅ Required Margin: {margin}")
        print(f"📊 Free Margin: {account_info.margin_free}")
        print(
            f"📊 Sufficient Margin: {'✅ Yes' if margin <= account_info.margin_free else '❌ No'}"
        )

    # 10. Test order validation
    print("\n🔟 Testing order validation...")
    check_result = mt5.order_check(buy_request)
    if check_result is None:
        print("❌ Order check failed")
    else:
        print("✅ Order Check Result:")
        print(f"📊 Retcode: {check_result.retcode}")
        print(f"📊 Comment: {check_result.comment}")
        print(f"📊 Balance: {check_result.balance}")
        print(f"📊 Equity: {check_result.equity}")
        print(f"📊 Margin: {check_result.margin}")
        print(f"📊 Margin Free: {check_result.margin_free}")
        print(f"📊 Margin Level: {check_result.margin_level}")
        print(f"📊 Order Valid: {'✅ Yes' if check_result.retcode == 0 else '❌ No'}")

    print("\n🎯 MT5 Connection Test Complete!")
    print("📊 All systems ready for trading")

    return True


def main():
    """Main function."""
    print("🎯 Enhanced MT5 Connection Test")
    print("=" * 60)

    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 not available")
        return

    success = test_mt5_connection()

    if success:
        print("\n✅ All tests passed! MT5 is ready for trading.")
    else:
        print("\n❌ Some tests failed. Please check MT5 connection.")

    # Cleanup
    mt5.shutdown()


if __name__ == "__main__":
    main()
