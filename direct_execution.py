#!/usr/bin/env python3
"""
MR BEN AI System - Direct Execution
Bypasses terminal encoding issues by running everything directly
"""

import logging
import os
from datetime import datetime


def setup_logging():
    """Setup logging"""
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(
                f'logs/direct_execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                encoding='utf-8',
            ),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def run_python_script(script_name, description=""):
    """Run a Python script directly"""
    logger = logging.getLogger(__name__)

    print(f"\n📊 Running: {description}")
    logger.info(f"Executing: {script_name}")

    try:
        # Import and run the script directly
        if script_name == "test_system_update.py":
            # Import the test module
            import test_system_update

            test_system_update.test_system_components()
            return True
        elif script_name == "fixed_comprehensive_update.py":
            # Import the fixed update module
            import fixed_comprehensive_update

            updater = fixed_comprehensive_update.FixedComprehensiveUpdater()
            updater.run_fixed_comprehensive_update()
            return True
        else:
            print(f"❌ Unknown script: {script_name}")
            return False

    except Exception as e:
        logger.error(f"❌ Error executing {script_name}: {e}")
        print(f"❌ Error: {e}")
        return False


def main():
    """Main execution function"""
    logger = setup_logging()

    print("🎯 MR BEN AI System - Direct Execution")
    print("=" * 50)
    logger.info("Starting MR BEN AI System direct execution")

    # Step 1: Test system components
    print("\n📊 Step 1: Testing system components...")
    success = run_python_script("test_system_update.py", "System component test")

    if not success:
        print("❌ System component test failed.")
        return

    # Step 2: Run comprehensive update
    print("\n🚀 Step 2: Running comprehensive system update...")
    success = run_python_script("fixed_comprehensive_update.py", "Comprehensive system update")

    if success:
        print("\n✅ MR BEN AI System update completed successfully!")
        print("📋 Check the logs/ directory for detailed reports")
    else:
        print("\n❌ System update failed. Please check the logs for details.")

    logger.info("MR BEN AI System direct execution completed")


if __name__ == "__main__":
    main()
