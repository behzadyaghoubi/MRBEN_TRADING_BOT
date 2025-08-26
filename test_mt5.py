"""
Test MT5 Connection
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

print("🤖 Testing MT5 Connection")
print("=" * 30)

try:
    # Initialize MT5
    print("📡 Initializing MT5...")
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        print("💡 Make sure MetaTrader 5 is running")
        exit(1)
    
    print("✅ MT5 initialized successfully")
    
    # Try to login
    print("🔐 Attempting to login...")
    if not mt5.login():
        print("❌ MT5 login failed")
        print("💡 Please check your MT5 credentials")
        print("💡 Make sure MT5 is running and you're logged in")
        exit(1)
    
    print("✅ MT5 login successful")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Cannot get account info")
        exit(1)
    
    print(f"💰 Account: {account_info.login}")
    print(f"💵 Balance: {account_info.balance}")
    print(f"📊 Equity: {account_info.equity}")
    print(f"🏦 Broker: {account_info.company}")
    
    # Get symbol info
    symbol = "XAUUSD.PRO"
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found")
        print("💡 Available symbols:")
        symbols = mt5.symbols_get()
        for s in symbols[:10]:  # Show first 10 symbols
            print(f"   - {s.name}")
        exit(1)
    
    print(f"✅ Symbol {symbol} found")
    print(f"📈 Bid: {symbol_info.bid}")
    print(f"📉 Ask: {symbol_info.ask}")
    
    # Get market data
    print("📊 Getting market data...")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
    if rates is None:
        print("❌ Cannot get market data")
        exit(1)
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print("✅ Market data received:")
    print(df.tail())
    
    # Test order (simulation)
    print("\n🧪 Testing order simulation...")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("❌ Cannot get current tick")
        exit(1)
    
    print(f"📈 Current Bid: {tick.bid}")
    print(f"📉 Current Ask: {tick.ask}")
    
    # Try to send a test order (with very small volume)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,  # Very small volume
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "deviation": 20,
        "magic": 234000,
        "comment": "Test_Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print("📤 Sending test order...")
    result = mt5.order_send(request)
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("✅ Test order successful!")
        print(f"🎫 Ticket: {result.order}")
    else:
        print(f"❌ Test order failed: {result.comment}")
        print(f"📋 Retcode: {result.retcode}")
    
    print("\n✅ MT5 connection test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Shutdown MT5
    mt5.shutdown()
    print("�� MT5 disconnected") 