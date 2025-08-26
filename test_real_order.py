#!/usr/bin/env python3
"""
Test Real Order Sending
Tests actual order sending to MT5
"""

import os
import sys
import json
import time
from datetime import datetime

# MT5 Integration
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("❌ MetaTrader5 not available")
    MT5_AVAILABLE = False
    sys.exit(1)

def test_real_order():
    """Test sending a real order to MT5."""
    print("🔍 Testing Real Order Sending")
    print("=" * 60)
    
    # 1. Initialize MT5
    print("\n1️⃣ Initializing MT5...")
    if not mt5.initialize():
        print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
        return False
    print("✅ MT5 initialized successfully")
    
    # 2. Load config
    config_path = 'config/settings.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
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
    
    # 3. Login to MT5
    print("\n2️⃣ Logging into MT5...")
    if not mt5.login(login=login, password=password, server=server):
        print(f"❌ Failed to login to MT5: {mt5.last_error()}")
        return False
    print("✅ Login successful")
    
    # 4. Get symbol info
    print(f"\n3️⃣ Getting symbol information for {symbol}...")
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found")
        return False
    
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to select symbol {symbol}")
            return False
    
    # 5. Get current tick
    print(f"\n4️⃣ Getting current tick...")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ Failed to get tick data")
        return False
    
    print(f"✅ Current Tick:")
    print(f"📊 Bid: {tick.bid}")
    print(f"📊 Ask: {tick.ask}")
    print(f"📊 Time: {datetime.fromtimestamp(tick.time)}")
    
    # 6. Prepare order request
    print(f"\n5️⃣ Preparing order request...")
    
    # Use current ask price for BUY
    entry_price = tick.ask
    sl_price = entry_price - 50 * symbol_info.point  # 50 points SL
    tp_price = entry_price + 100 * symbol_info.point  # 100 points TP
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20,
        "magic": 654321,
        "comment": f"TEST_REAL_ORDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print(f"✅ Order Request Details:")
    print(f"📊 Symbol: {request['symbol']}")
    print(f"📊 Volume: {request['volume']}")
    print(f"📊 Price: {request['price']}")
    print(f"📊 SL: {request['sl']}")
    print(f"📊 TP: {request['tp']}")
    print(f"📊 Magic: {request['magic']}")
    print(f"📊 Comment: {request['comment']}")
    
    # 7. Check order before sending
    print(f"\n6️⃣ Checking order validity...")
    check_result = mt5.order_check(request)
    if check_result is None:
        print(f"❌ Order check failed")
        return False
    
    print(f"✅ Order Check Result:")
    print(f"📊 Retcode: {check_result.retcode}")
    print(f"📊 Comment: {check_result.comment}")
    print(f"📊 Order Valid: {'✅ Yes' if check_result.retcode == 0 else '❌ No'}")
    
    if check_result.retcode != 0:
        print(f"❌ Order validation failed: {check_result.comment}")
        return False
    
    # 8. Send order
    print(f"\n7️⃣ Sending order to MT5...")
    print(f"📤 Sending order with request: {request}")
    
    result = mt5.order_send(request)
    
    print(f"📥 Order result received: {result}")
    
    if result is None:
        print(f"❌ Order result is None!")
        print(f"📊 Last MT5 error: {mt5.last_error()}")
        return False
    
    print(f"✅ Order Result Details:")
    print(f"📊 Retcode: {result.retcode}")
    print(f"📊 Order: {result.order}")
    print(f"📊 Volume: {result.volume}")
    print(f"📊 Price: {result.price}")
    print(f"📊 Bid: {result.bid}")
    print(f"📊 Ask: {result.ask}")
    print(f"📊 Comment: {result.comment}")
    print(f"📊 Request: {result.request}")
    print(f"📊 Retcode Description: {result.retcode_description}")
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ ORDER EXECUTED SUCCESSFULLY!")
        print(f"📊 Order ID: {result.order}")
        print(f"📊 Executed Price: {result.price}")
        return True
    else:
        print(f"❌ ORDER FAILED!")
        print(f"📊 Retcode: {result.retcode}")
        print(f"📊 Comment: {result.comment}")
        return False

def main():
    """Main function."""
    print("🎯 Test Real Order Sending")
    print("=" * 60)
    
    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 not available")
        return
    
    success = test_real_order()
    
    if success:
        print("\n✅ Order sent successfully!")
    else:
        print("\n❌ Order sending failed.")
    
    # Cleanup
    mt5.shutdown()

if __name__ == "__main__":
    main() 