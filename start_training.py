#!/usr/bin/env python3
"""
Start Training
Immediate execution
"""

print("🚀 Starting LSTM Training with Real Data...")

# Import and execute training
try:
    from execute_training_direct import main

    main()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
