#!/usr/bin/env python3
"""
Test Real MT5 Order
Send a real order to MT5 and verify it appears
"""

import MetaTrader5 as mt5
import json
import time

def test_real_mt5_order():
    """Test sending a real order to MT5."""
    
    print("🧪 Testing Real MT5 Order...")
    print("=" * 40)
    
    try:
        # Load config
        with open('config/settings.json', 'r') as f:
            config = json.load(f)
        
        mt5_login = config['mt5_login']
        mt5_password = config['mt5_password']
        mt5_server = config['mt5_server']
        
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
        
        # Get current tick for accurate pricing
        symbol = "XAUUSD.PRO"
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            current_price = tick.ask
            print(f"📊 Current {symbol} Ask: {current_price}")
        else:
            current_price = 3314.0
            print(f"⚠️ Using fallback price: {current_price}")
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": current_price,
            "sl": current_price - 50,  # 50 points stop loss
            "tp": current_price + 100,  # 100 points take profit
            "deviation": 20,
            "magic": 654321,
            "comment": "MR_BEN_TEST_ORDER",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        print(f"🚀 Sending order to MT5...")
        print(f"📊 Symbol: {request['symbol']}")
        print(f"📊 Volume: {request['volume']}")
        print(f"📊 Price: {request['price']}")
        print(f"📊 SL: {request['sl']}")
        print(f"📊 TP: {request['tp']}")
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            print("❌ Order send failed - result is None")
        elif result.retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ ORDER EXECUTED SUCCESSFULLY!")
            print(f"📊 Order ID: {result.order}")
            print(f"📊 Volume: {result.volume}")
            print(f"📊 Price: {result.price}")
            print(f"📊 Retcode: {result.retcode}")
            print(f"📊 Comment: {result.comment}")
            print("\n🎉 CHECK YOUR MT5 TERMINAL - ORDER SHOULD BE VISIBLE!")
        else:
            print("❌ Order failed")
            print(f"📊 Retcode: {result.retcode}")
            print(f"📊 Comment: {result.comment}")
        
        # Wait a moment
        print("\n⏳ Waiting 3 seconds...")
        time.sleep(3)
        
        # Check open positions
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            print(f"📊 Open positions for {symbol}: {len(positions)}")
            for pos in positions:
                print(f"📊 Position ID: {pos.ticket}")
                print(f"📊 Type: {'BUY' if pos.type == 0 else 'SELL'}")
                print(f"📊 Volume: {pos.volume}")
                print(f"📊 Price: {pos.price_open}")
                print(f"📊 SL: {pos.sl}")
                print(f"📊 TP: {pos.tp}")
        else:
            print(f"📊 No open positions for {symbol}")
        
        # Shutdown MT5
        mt5.shutdown()
        print("✅ MT5 shutdown complete")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_mt5_order() 