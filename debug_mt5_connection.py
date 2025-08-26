#!/usr/bin/env python3
"""
Debug MT5 Connection
Debug MT5 connection and order sending issues
"""

import json
import os

import MetaTrader5 as mt5


def debug_mt5():
    """Debug MT5 connection and order sending."""

    print("🔍 Debugging MT5 Connection...")
    print("=" * 40)

    try:
        # Check if MT5 is available
        print(f"📊 MT5 Available: {mt5 is not None}")

        # Initialize MT5
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return

        print("✅ MT5 initialized")

        # Load config
        config_path = 'config/settings.json'
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)

            login = config.get('mt5_login', 1104123)
            password = config.get('mt5_password', '-4YcBgRd')
            server = config.get('mt5_server', 'OxSecurities-Demo')
        else:
            login = 1104123
            password = '-4YcBgRd'
            server = 'OxSecurities-Demo'

        print(f"📊 Login: {login}")
        print(f"📊 Server: {server}")

        # Login to MT5
        if not mt5.login(login=login, password=password, server=server):
            print(f"❌ MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return

        print("✅ MT5 login successful")

        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print("✅ Account Info:")
            print(f"📊 Balance: {account_info.balance}")
            print(f"📊 Equity: {account_info.equity}")
            print(f"📊 Currency: {account_info.currency}")

        # Get symbol info
        symbol = "XAUUSD.PRO"
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            print(f"✅ Symbol Info for {symbol}:")
            print(f"📊 Point: {symbol_info.point}")
            print(f"📊 Digits: {symbol_info.digits}")
            print(f"📊 Spread: {symbol_info.spread}")
            print(f"📊 Trade Mode: {symbol_info.trade_mode}")
            print(f"📊 Trade Stops Level: {symbol_info.trade_stops_level}")
            print(f"📊 Trade Freeze Level: {symbol_info.trade_freeze_level}")
        else:
            print(f"❌ Could not get symbol info for {symbol}")

        # Get current tick
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print("✅ Current Tick:")
            print(f"📊 Bid: {tick.bid}")
            print(f"📊 Ask: {tick.ask}")
            print(f"📊 Time: {tick.time}")
        else:
            print(f"❌ Could not get tick data for {symbol}")

        # Test order request
        print("\n🧪 Testing order request...")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask if tick else 3314.0,
            "deviation": 20,
            "magic": 654321,
            "comment": "DEBUG_TEST",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"📊 Request: {request}")

        # Send order
        result = mt5.order_send(request)

        print(f"📊 Result: {result}")

        if result is None:
            print("❌ Order send returned None")
        else:
            print(f"📊 Retcode: {result.retcode}")
            print(f"📊 Comment: {result.comment}")
            print(f"📊 Order: {result.order}")
            print(f"📊 Volume: {result.volume}")
            print(f"📊 Price: {result.price}")

        # Shutdown MT5
        mt5.shutdown()
        print("✅ MT5 shutdown complete")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_mt5()
