#!/usr/bin/env python3
"""
Run Training with Keyboard Fix
Final execution script with automatic keyboard management
"""

import os
import subprocess


def run_training():
    """Run the LSTM training with keyboard management"""
    print("🚀 Starting LSTM Training with Keyboard Management")
    print("=" * 70)

    # First, try to import and use keyboard manager
    try:
        from keyboard_manager import run_python_script_with_english_keyboard

        print("✅ Keyboard manager loaded successfully")

        # Run the training script with keyboard management
        result = run_python_script_with_english_keyboard(
            "execute_training_final.py", timeout=1800  # 30 minutes timeout
        )

        if result and result.returncode == 0:
            print("\n🎉 Training completed successfully!")
            print("✅ Model saved: models/mrben_lstm_real_data.h5")
            print("✅ Scaler saved: models/mrben_lstm_real_data_scaler.save")
            return True
        else:
            print("\n❌ Training failed!")
            if result:
                print(f"Return code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
            return False

    except ImportError:
        print("⚠️  Keyboard manager not available, using fallback method")
        return run_training_fallback()


def run_training_fallback():
    """Fallback method without keyboard manager"""
    print("🔄 Using fallback training method...")

    # Check if training script exists
    if not os.path.exists("execute_training_final.py"):
        print("❌ Training script not found: execute_training_final.py")
        return False

    # Try different Python commands
    python_commands = [
        "python execute_training_final.py",
        "py execute_training_final.py",
        "python3 execute_training_final.py",
    ]

    for cmd in python_commands:
        print(f"\n🔄 Trying: {cmd}")

        try:
            # Run the command
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=1800  # 30 minutes
            )

            if result.returncode == 0:
                print(f"✅ Training completed successfully with: {cmd}")
                print("✅ Model saved: models/mrben_lstm_real_data.h5")
                print("✅ Scaler saved: models/mrben_lstm_real_data_scaler.save")
                return True
            else:
                print(f"❌ Failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"❌ Timeout with: {cmd}")
        except Exception as e:
            print(f"❌ Error with {cmd}: {e}")

    print("❌ All training methods failed")
    return False


def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")

    # Check data files
    required_files = ["data/real_market_sequences.npy", "data/real_market_labels.npy"]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_path}")

    # Check models directory
    if not os.path.exists("models"):
        os.makedirs("models")
        print("✅ Created models directory")
    else:
        print("✅ Models directory exists")

    # Check training script
    if not os.path.exists("execute_training_final.py"):
        print("❌ Training script not found: execute_training_final.py")
        return False
    else:
        print("✅ Training script found")

    print("✅ All prerequisites met")
    return True


def main():
    """Main function"""
    print("🎯 LSTM Training with Keyboard Management")
    print("=" * 70)

    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please check the errors above.")
        return

    # Run training
    success = run_training()

    if success:
        print("\n🎉 LSTM Training Completed Successfully!")
        print("=" * 50)
        print("📁 Files created:")
        print("   - models/mrben_lstm_real_data.h5")
        print("   - models/mrben_lstm_real_data_scaler.save")
        print("   - models/mrben_lstm_real_data_best.h5")

        print("\n🎯 Next Steps:")
        print("   1. Test the system: test_complete_system_real_data.py")
        print("   2. Run live trading: live_trader_clean.py")
        print("   3. Monitor performance")

    else:
        print("\n❌ Training failed!")
        print("Please check the errors above and try again.")


if __name__ == "__main__":
    main()
