#!/usr/bin/env python3
"""
Check Open Trades
Check all open trades and their details
"""

import os
import sys
import json
from datetime import datetime

# MT5 Integration
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("❌ MetaTrader5 not available")
    MT5_AVAILABLE = False
    sys.exit(1)

def check_open_trades():
    """Check all open trades."""
    print("🔍 Checking Open Trades")
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
    else:
        login = 1104123
        password = '-4YcBgRd'
        server = 'OxSecurities-Demo'
        symbol = 'XAUUSD.PRO'
    
    # 3. Login to MT5
    print("\n2️⃣ Logging into MT5...")
    if not mt5.login(login=login, password=password, server=server):
        print(f"❌ Failed to login to MT5: {mt5.last_error()}")
        return False
    print("✅ Login successful")
    
    # 4. Get all open positions
    print(f"\n3️⃣ Getting all open positions...")
    positions = mt5.positions_get()
    if positions is None:
        print("❌ Failed to get positions")
        return False
    
    print(f"📊 Total open positions: {len(positions)}")
    
    # 5. Filter positions by symbol
    symbol_positions = [pos for pos in positions if pos.symbol == symbol]
    print(f"📊 {symbol} positions: {len(symbol_positions)}")
    
    # 6. Show position details
    if symbol_positions:
        print(f"\n4️⃣ {symbol} Position Details:")
        print("-" * 80)
        print(f"{'Ticket':<10} {'Type':<6} {'Volume':<8} {'Price Open':<12} {'Price Current':<15} {'SL':<10} {'TP':<10} {'Profit':<12} {'Time':<20}")
        print("-" * 80)
        
        total_profit = 0
        for pos in symbol_positions:
            pos_type = "BUY" if pos.type == 0 else "SELL"
            open_time = datetime.fromtimestamp(pos.time)
            
            print(f"{pos.ticket:<10} {pos_type:<6} {pos.volume:<8.2f} {pos.price_open:<12.2f} {pos.price_current:<15.2f} {pos.sl:<10.2f} {pos.tp:<10.2f} {pos.profit:<12.2f} {open_time.strftime('%H:%M:%S'):<20}")
            total_profit += pos.profit
        
        print("-" * 80)
        print(f"Total Profit: {total_profit:.2f}")
    
    # 7. Get account info
    print(f"\n5️⃣ Account Information:")
    account_info = mt5.account_info()
    if account_info:
        print(f"📊 Balance: {account_info.balance}")
        print(f"📊 Equity: {account_info.equity}")
        print(f"📊 Margin: {account_info.margin}")
        print(f"📊 Free Margin: {account_info.margin_free}")
        print(f"📊 Margin Level: {account_info.margin_level}")
    
    return True

def main():
    """Main function."""
    print("🎯 Check Open Trades")
    print("=" * 60)
    
    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 not available")
        return
    
    success = check_open_trades()
    
    if success:
        print("\n✅ Open trades check completed!")
    else:
        print("\n❌ Open trades check failed.")
    
    # Cleanup
    mt5.shutdown()

if __name__ == "__main__":
    main() 