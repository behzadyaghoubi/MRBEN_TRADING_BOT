#!/usr/bin/env python3
"""
System Monitor
Monitor both live_trader_clean.py and dynamic_position_manager.py
"""

import time
from datetime import datetime

from dynamic_position_manager import DynamicPositionManager


class SystemMonitor:
    """Monitor both trading systems."""

    def __init__(self):
        """Initialize the monitor."""
        self.manager = None
        self.monitoring = False

    def start_monitoring(self):
        """Start monitoring both systems."""
        print("🔍 Starting System Monitor")
        print("=" * 60)

        try:
            # Initialize position manager
            self.manager = DynamicPositionManager()

            if not self.manager.mt5_connected:
                print("❌ Failed to connect to MT5")
                return False

            self.monitoring = True
            print("✅ Monitoring started successfully")
            print("📊 Monitoring interval: 30 seconds")
            print("=" * 60)

            while self.monitoring:
                try:
                    # Update position tracking
                    self.manager.update_position_tracking()

                    # Get current status
                    self._print_system_status()

                    # Wait for next check
                    time.sleep(30)

                except KeyboardInterrupt:
                    print("\n🛑 Monitoring stopped by user")
                    break
                except Exception as e:
                    print(f"❌ Error in monitoring: {e}")
                    time.sleep(30)

            return True

        except Exception as e:
            print(f"❌ Error starting monitor: {e}")
            return False

    def _print_system_status(self):
        """Print current system status."""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"\n📊 {timestamp} - System Status Report")
            print("-" * 50)

            # Get position summary
            summary = self.manager.get_position_summary()

            print(f"📈 Active Positions: {summary['total_positions']}")
            print(f"💰 Total P&L: {summary['total_profit']:.2f}")

            if summary['total_positions'] > 0:
                print("\n📋 Position Details:")
                for pos in summary['positions']:
                    print(
                        f"   🎯 {pos['ticket']} | {pos['type']} | "
                        f"Volume: {pos['volume']} | "
                        f"Entry: {pos['entry_price']:.2f} | "
                        f"Current: {pos['current_price']:.2f} | "
                        f"P&L: {pos['profit']:.2f}"
                    )

                    # Calculate ATR-based SL/TP
                    df = self.manager._get_market_data(100)
                    if df is not None:
                        atr = self.manager.calculate_atr(df)
                        new_sl = self.manager.calculate_trailing_stop(pos, atr)
                        new_tp = self.manager.calculate_trailing_take_profit(pos)

                        print(
                            f"      📊 ATR: {atr:.2f} | "
                            f"Current SL: {pos['sl']:.2f} | "
                            f"Current TP: {pos['tp']:.2f}"
                        )

                        if new_sl is not None:
                            print(f"      🔄 New SL: {new_sl:.2f}")
                        if new_tp is not None:
                            print(f"      🔄 New TP: {new_tp:.2f}")

            # Check if new positions can be opened
            can_open = self.manager.can_open_new_position()
            print(f"\n🔒 Can Open New Position: {can_open}")

            # Get account info
            if self.manager.mt5_connected:
                try:
                    import MetaTrader5 as mt5

                    account_info = mt5.account_info()
                    if account_info:
                        print(
                            f"💳 Account: {account_info.login} | "
                            f"Balance: {account_info.balance:.2f} | "
                            f"Equity: {account_info.equity:.2f}"
                        )
                except:
                    pass

            print("-" * 50)

        except Exception as e:
            print(f"❌ Error printing status: {e}")

    def stop_monitoring(self):
        """Stop monitoring."""
        print("🛑 Stopping System Monitor...")
        self.monitoring = False

        if self.manager and self.manager.mt5_connected:
            try:
                import MetaTrader5 as mt5

                mt5.shutdown()
                print("✅ MT5 connection closed")
            except:
                pass

        print("✅ System Monitor stopped")


def main():
    """Main function."""
    print("🎯 System Monitor")
    print("=" * 60)

    # Create monitor
    monitor = SystemMonitor()

    try:
        # Start monitoring
        if monitor.start_monitoring():
            print("✅ Monitor started successfully")
        else:
            print("❌ Failed to start monitor")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()
