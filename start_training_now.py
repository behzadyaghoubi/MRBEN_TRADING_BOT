#!/usr/bin/env python3
"""
Start Training Now
Simple script to start LSTM training
"""

print("🚀 Starting LSTM Training...")

# Import and run training directly
try:
    from direct_training import main
    print("✅ Imported training module")
    success = main()
    if success:
        print("🎉 Training completed successfully!")
    else:
        print("❌ Training failed!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Press Enter to exit...")
input() 