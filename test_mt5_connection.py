#!/usr/bin/env python3
"""
Test MT5 Connection
Verify connection to MetaTrader 5 and access to XAUUSD.PRO data
"""

import json
import sys

import MetaTrader5 as mt5


def test_mt5_connection():
    """Test MT5 connection with current configuration"""
    print("🔍 Testing MT5 Connection...")

    # Load configuration
    try:
        with open('enhanced_config.json') as f:
            config = json.load(f)
        mt5_config = config['mt5']
        print(f"✅ Configuration loaded: {mt5_config['server']}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

    # Test MT5 initialization
    print("📡 Initializing MT5...")
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    print("✅ MT5 initialized successfully")

    # Test login
    print(f"🔐 Attempting login to {mt5_config['server']}...")
    if not mt5.login(
        login=mt5_config['login'], password=mt5_config['password'], server=mt5_config['server']
    ):
        print(f"❌ Login failed: {mt5.last_error()}")
        return False
    print("✅ Login successful")

    # Test symbol selection
    symbol = config['trading']['symbol']
    print(f"📊 Testing symbol selection: {symbol}")
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Symbol selection failed: {mt5.last_error()}")
        return False
    print("✅ Symbol selected successfully")

    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"💰 Account Balance: {account_info.balance}")
        print(f"💰 Account Equity: {account_info.equity}")
        print(f"💰 Account Currency: {account_info.currency}")
    else:
        print("❌ Failed to get account info")
        return False

    # Test data retrieval
    print("📈 Testing data retrieval...")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
    if rates is None:
        print("❌ Failed to get price data")
        return False
    print(f"✅ Retrieved {len(rates)} price bars")

    print("🎉 All MT5 tests passed!")
    return True


if __name__ == "__main__":
    success = test_mt5_connection()
    if not success:
        print("\n🔧 Troubleshooting Tips:")
        print("1. Make sure MetaTrader 5 is installed and running")
        print("2. Check your login credentials in enhanced_config.json")
        print("3. Verify the server name is correct")
        print("4. Ensure you have internet connection")
        print("5. Try logging into MT5 manually first")
    sys.exit(0 if success else 1)
