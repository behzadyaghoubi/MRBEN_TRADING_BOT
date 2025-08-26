#!/usr/bin/env python3
"""
Standalone Baseline Backtest Runner
"""

import os
import sys

sys.path.insert(0, 'src')

try:
    print("🚀 Starting Baseline Backtest...")

    # Import and run backtest
    from core.backtest import run

    # Create simple config
    class SimpleConfig:
        SYMBOL = "XAUUSD.PRO"

    # Run backtest
    result = run('XAUUSD.PRO', '2025-07-01', '2025-08-15', SimpleConfig())
    print(f"✅ Backtest completed: {result}")

    # Check generated files
    print("\n📁 Checking generated files...")
    if os.path.exists('docs/pro/01_baseline/backtest_report.md'):
        print("✅ Backtest report created")
    else:
        print("❌ Backtest report not found")

    if os.path.exists('docs/pro/01_baseline/metrics.json'):
        print("✅ Metrics file created")
    else:
        print("❌ Metrics file not found")

    if os.path.exists('docs/pro/01_baseline/trade-list.csv'):
        print("✅ Trade list created")
    else:
        print("❌ Trade list not found")

    print("\n🎯 Baseline Backtest Phase 1 Complete!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
