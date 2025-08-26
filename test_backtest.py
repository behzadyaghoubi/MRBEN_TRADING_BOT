#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, 'src')

try:
    print("Testing imports...")
    from core.backtest import run

    print("✅ Import successful")

    print("Running backtest...")

    # Create a simple config object for testing
    class SimpleConfig:
        SYMBOL = "XAUUSD.PRO"

    result = run('XAUUSD.PRO', '2025-07-01', '2025-08-15', SimpleConfig())
    print(f"✅ Backtest result: {result}")

    # Check if files were created
    print("Checking generated files...")
    if os.path.exists('docs/pro/01_baseline/backtest_report.md'):
        print("✅ Backtest report created")
    else:
        print("❌ Backtest report not found")

    if os.path.exists('docs/pro/01_baseline/metrics.json'):
        print("✅ Metrics file created")
    else:
        print("❌ Metrics file not found")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
