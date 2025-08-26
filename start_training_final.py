#!/usr/bin/env python3
"""
Start Training Final
Final execution script
"""

print("🚀 Starting LSTM Training...")

# Import and execute training
try:
    from run_training_direct import main
    main()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 