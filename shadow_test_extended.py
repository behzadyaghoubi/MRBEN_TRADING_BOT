#!/usr/bin/env python3
"""
Extended Shadow Mode Test - 30-60 minutes with comprehensive logging
"""
import logging
import time
from datetime import datetime, timedelta

from live_trader_ai_enhanced import EnhancedAILiveTrader


def setup_file_logging():
    """Setup file logging for detailed analysis"""
    log_filename = f"logs/shadow_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create logs directory if it doesn't exist
    import os

    os.makedirs("logs", exist_ok=True)

    # Setup file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    print(f"📝 Detailed logs will be saved to: {log_filename}")
    return log_filename


def run_extended_shadow_test(duration_minutes=10):
    """Run extended shadow mode test"""

    print("🤖 EXTENDED SHADOW MODE TEST")
    print("=" * 60)
    print(f"Duration: {duration_minutes} minutes")
    print(f"Start Time: {datetime.now()}")

    # Setup logging
    log_file = setup_file_logging()

    # Create trader
    print("🔧 Initializing Enhanced AI Trader...")
    trader = EnhancedAILiveTrader()

    # Verify Shadow mode
    print(f"📊 Execution Mode: {trader.config.AI_CONTROL_MODE}")
    if trader.config.AI_CONTROL_MODE != "shadow":
        print("⚠️ WARNING: Not in Shadow mode!")
        return

    # Start the system
    print("🚀 Starting trading system...")
    trader.start()

    print("\n📋 WATCHING FOR THESE KEY LOGS:")
    print("  1. ✅ 'Conformal gate loaded'")
    print("  2. 🔍 'Conformal: accept=False' (rejection)")
    print("  3. 🛡️ 'Risk Governor: [reason]' (risk rejection)")
    print("  4. 👁️ '[SHADOW] Would execute...' (would execute)")
    print("  5. 🤖 'AI DECISION' entries")
    print("=" * 60)

    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)

    # Statistics tracking
    iteration_count = 0
    last_stats_time = start_time

    try:
        while datetime.now() < end_time:
            # Print progress every 2 minutes
            now = datetime.now()
            if (now - last_stats_time).total_seconds() >= 120:  # 2 minutes
                elapsed = now - start_time
                remaining = end_time - now
                print(
                    f"\n⏱️ Progress: {elapsed.total_seconds()/60:.1f}min elapsed, {remaining.total_seconds()/60:.1f}min remaining"
                )

                # Print current stats
                if hasattr(trader, 'session_stats'):
                    stats = trader.session_stats
                    print(
                        f"📊 Stats: Signals={stats.get('signals_generated', 0)}, "
                        f"Conformal Accept={stats.get('conformal_accepted', 0)}, "
                        f"Conformal Reject={stats.get('conformal_rejected', 0)}, "
                        f"Risk Reject={stats.get('risk_rejected', 0)}"
                    )

                last_stats_time = now

            time.sleep(10)  # Check every 10 seconds
            iteration_count += 1

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")

    finally:
        # Stop the system
        print("\n🛑 Stopping trading system...")
        trader.stop()

        # Final results
        print("\n" + "=" * 60)
        print("📊 EXTENDED SHADOW MODE TEST RESULTS")
        print("=" * 60)
        print(f"Test Duration: {duration_minutes} minutes")
        print(f"Iterations: {iteration_count}")

        if hasattr(trader, 'session_stats'):
            stats = trader.session_stats
            print(f"Signals Generated: {stats.get('signals_generated', 0)}")
            print(f"Conformal Accepted: {stats.get('conformal_accepted', 0)}")
            print(f"Conformal Rejected: {stats.get('conformal_rejected', 0)}")
            print(f"Risk Rejected: {stats.get('risk_rejected', 0)}")
            print(f"Shadow Executions: {stats.get('trades_executed', 0)}")

            # Calculate rates
            total_decisions = stats.get('conformal_accepted', 0) + stats.get(
                'conformal_rejected', 0
            )
            if total_decisions > 0:
                accept_rate = stats.get('conformal_accepted', 0) / total_decisions * 100
                print(f"Conformal Accept Rate: {accept_rate:.1f}%")

        print(f"\n📝 Detailed logs saved to: {log_file}")
        print("🔍 Look for the specific log patterns mentioned above!")
        print("=" * 60)


if __name__ == "__main__":
    # Run for 10 minutes (can be adjusted)
    run_extended_shadow_test(duration_minutes=10)
