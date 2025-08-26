#!/usr/bin/env python3
"""
Close All Positions Script
Closes all open positions in MT5 account
"""

import json
import os
import sys
import time

# MT5 Integration
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    print("❌ MetaTrader5 not available")
    MT5_AVAILABLE = False
    sys.exit(1)


def close_all_positions():
    """Close all open positions."""
    print("🔒 Closing All Open Positions")
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
        with open(config_path) as f:
            config = json.load(f)

        login = config.get('mt5_login', 1104123)
        password = config.get('mt5_password', '-4YcBgRd')
        server = config.get('mt5_server', 'OxSecurities-Demo')
    else:
        login = 1104123
        password = '-4YcBgRd'
        server = 'OxSecurities-Demo'

    # 3. Login to MT5
    print("\n2️⃣ Logging into MT5...")
    if not mt5.login(login=login, password=password, server=server):
        print(f"❌ Failed to login to MT5: {mt5.last_error()}")
        return False
    print("✅ Login successful")

    # 4. Get account info
    account_info = mt5.account_info()
    if account_info:
        print(
            f"📊 Account: {account_info.login} | Balance: {account_info.balance} | Equity: {account_info.equity}"
        )

    # 5. Get all open positions
    print("\n3️⃣ Getting all open positions...")
    positions = mt5.positions_get()
    if positions is None:
        print("❌ Failed to get positions")
        return False

    total_positions = len(positions)
    print(f"📊 Total open positions: {total_positions}")

    if total_positions == 0:
        print("✅ No open positions to close!")
        return True

    # 6. Show position details before closing
    print("\n4️⃣ Position Details Before Closing:")
    print("-" * 80)
    print(
        f"{'Ticket':<10} {'Symbol':<12} {'Type':<6} {'Volume':<8} {'Price Open':<12} {'Price Current':<15} {'Profit':<12}"
    )
    print("-" * 80)

    total_profit = 0
    for pos in positions:
        pos_type = "BUY" if pos.type == 0 else "SELL"
        print(
            f"{pos.ticket:<10} {pos.symbol:<12} {pos_type:<6} {pos.volume:<8.2f} {pos.price_open:<12.2f} {pos.price_current:<15.2f} {pos.profit:<12.2f}"
        )
        total_profit += pos.profit

    print("-" * 80)
    print(f"Total Profit: {total_profit:.2f}")

    # 7. Close all positions
    print("\n5️⃣ Closing all positions...")
    closed_count = 0
    failed_count = 0

    for pos in positions:
        try:
            # Get current tick for accurate pricing
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                print(f"❌ Cannot get tick for {pos.symbol}")
                failed_count += 1
                continue

            # Determine close action (opposite of position type)
            if pos.type == 0:  # BUY position -> SELL to close
                action = "SELL"
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid  # Use bid price for selling
            else:  # SELL position -> BUY to close
                action = "BUY"
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask  # Use ask price for buying

            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,  # Close specific position
                "price": price,
                "deviation": 20,
                "magic": 654321,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send close order
            print(f"📤 Closing {pos.symbol} {action} position {pos.ticket} at {price:.2f}...")
            result = mt5.order_send(request)

            if result is None:
                print(f"❌ Failed to close position {pos.ticket}: MT5 order_send returned None")
                print(f"   Last MT5 error: {mt5.last_error()}")
                failed_count += 1
            elif result.retcode == 10009:  # TRADE_RETCODE_DONE
                print(f"✅ Successfully closed position {pos.ticket} | Order ID: {result.order}")
                closed_count += 1
            else:
                print(
                    f"❌ Failed to close position {pos.ticket}: retcode={result.retcode}, comment={result.comment}"
                )
                failed_count += 1

            # Small delay between orders
            time.sleep(0.1)

        except Exception as e:
            print(f"❌ Error closing position {pos.ticket}: {e}")
            failed_count += 1

    # 8. Final summary
    print("\n6️⃣ Closing Summary:")
    print(f"📊 Total positions: {total_positions}")
    print(f"✅ Successfully closed: {closed_count}")
    print(f"❌ Failed to close: {failed_count}")

    if failed_count > 0:
        print("\n⚠️ Some positions failed to close. You may need to close them manually.")

    # 9. Verify all positions are closed
    print("\n7️⃣ Verifying all positions are closed...")
    remaining_positions = mt5.positions_get()
    if remaining_positions is None:
        print("❌ Failed to get remaining positions")
    else:
        remaining_count = len(remaining_positions)
        if remaining_count == 0:
            print("✅ All positions successfully closed!")
        else:
            print(f"⚠️ {remaining_count} positions still open:")
            for pos in remaining_positions:
                pos_type = "BUY" if pos.type == 0 else "SELL"
                print(
                    f"   {pos.ticket} | {pos.symbol} | {pos_type} | {pos.volume} | Profit: {pos.profit:.2f}"
                )

    return True


def main():
    """Main function."""
    print("🎯 Close All Positions Script")
    print("=" * 60)

    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 not available")
        return

    # Confirm before closing
    print("\n⚠️ WARNING: This will close ALL open positions!")
    confirm = input("Are you sure you want to continue? (yes/no): ")

    if confirm.lower() not in ['yes', 'y']:
        print("❌ Operation cancelled by user")
        return

    success = close_all_positions()

    if success:
        print("\n✅ Close all positions operation completed!")
    else:
        print("\n❌ Close all positions operation failed.")

    # Cleanup
    mt5.shutdown()


if __name__ == "__main__":
    main()
