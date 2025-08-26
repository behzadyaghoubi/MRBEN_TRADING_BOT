#!/usr/bin/env python3
"""
MR BEN Trading System - Startup Script
======================================
Simple script to start the trading system with proper error handling.
"""

import sys
import os
import time
import logging
from datetime import datetime

def setup_logger():
    """Setup logging for the startup script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("SystemStartup")

def check_system_health():
    """Run a quick health check before starting."""
    logger = logging.getLogger("SystemStartup")
    logger.info("🔍 Running quick system health check...")
    
    # Check if main files exist
    required_files = [
        "src/main_runner.py",
        "config/settings.json",
        "models/mrben_ai_signal_filter_xgb.joblib"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return False
    
    logger.info("✅ Basic health check passed")
    return True

def start_trading_system():
    """Start the trading system."""
    logger = logging.getLogger("SystemStartup")
    
    logger.info("🚀 Starting MR BEN Trading System...")
    logger.info("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # Import and run the main runner
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import the main runner module
        from src.main_runner import main
        
        logger.info("✅ Main runner imported successfully")
        logger.info("🔄 Starting trading loop...")
        
        # Run the main function
        main()
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Please check that all required packages are installed")
        return False
    except Exception as e:
        logger.error(f"❌ System startup error: {e}")
        return False
    
    return True

def main():
    """Main function."""
    logger = setup_logger()
    
    print("="*60)
    print("🤖 MR BEN TRADING SYSTEM - STARTUP")
    print("="*60)
    
    # Check system health
    if not check_system_health():
        logger.error("❌ System health check failed. Please fix issues before starting.")
        sys.exit(1)
    
    # Start the system
    if start_trading_system():
        logger.info("✅ Trading system started successfully")
    else:
        logger.error("❌ Failed to start trading system")
        sys.exit(1)

if __name__ == "__main__":
    main() 