#!/usr/bin/env python3
"""
Simple MT5 Connection Test
Test basic MT5 connection and account info
"""

import MetaTrader5 as mt5
import json

def test_mt5_connection():
    """Test basic MT5 connection."""
    
    print("🧪 Testing MT5 Connection...")
    print("=" * 40)
    
    try:
        # Load config
        with open('config/settings.json', 'r') as f:
            config = json.load(f)
        
        mt5_login = config['mt5_login']
        mt5_password = config['mt5_password']
        mt5_server = config['mt5_server']
        
        print(f"📊 Login: {mt5_login}")
        print(f"📊 Server: {mt5_server}")
        
        # Initialize MT5
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return
        
        print("✅ MT5 initialized")
        
        # Login to MT5
        if not mt5.login(login=mt5_login, password=mt5_password, server=mt5_server):
            print("❌ MT5 login failed")
            mt5.shutdown()
            return
        
        print("✅ MT5 login successful")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Account Info:")
            print(f"📊 Balance: {account_info.balance}")
            print(f"📊 Equity: {account_info.equity}")
            print(f"📊 Margin: {account_info.margin}")
            print(f"📊 Free Margin: {account_info.margin_free}")
            print(f"📊 Currency: {account_info.currency}")
        else:
            print("❌ Could not get account info")
        
        # Get symbol info
        symbol = "XAUUSD.PRO"
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            print(f"✅ Symbol Info for {symbol}:")
            print(f"📊 Point: {symbol_info.point}")
            print(f"📊 Digits: {symbol_info.digits}")
            print(f"📊 Spread: {symbol_info.spread}")
            print(f"📊 Trade Mode: {symbol_info.trade_mode}")
        else:
            print(f"❌ Could not get symbol info for {symbol}")
        
        # Test basic order request (without sending)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": 3314.0,
            "deviation": 20,
            "magic": 654321,
            "comment": "TEST_ORDER",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        print(f"✅ Order request prepared:")
        print(f"📊 Symbol: {request['symbol']}")
        print(f"📊 Volume: {request['volume']}")
        print(f"📊 Type: {request['type']}")
        print(f"📊 Price: {request['price']}")
        
        # Shutdown MT5
        mt5.shutdown()
        print("✅ MT5 shutdown complete")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mt5_connection() 