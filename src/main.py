#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Trading System - Modular Main Entry Point
Simplified main function using modular components
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import MT5Config
from src.trading_system import TradingSystem
from src.trading_loop import TradingLoopManager


def setup_logging(config: MT5Config):
    """Setup logging configuration"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format='[%(asctime)s][%(levelname)s] %(message)s'
    )
    
    # Add file handler
    try:
        from logging.handlers import RotatingFileHandler
        os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
        
        fh = RotatingFileHandler(
            config.LOG_FILE, 
            maxBytes=5_000_000, 
            backupCount=5, 
            encoding='utf-8'
        )
        fh.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
        fh.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s'))
        
        # Add to root logger
        logging.getLogger().addHandler(fh)
        
    except Exception as e:
        logging.warning(f"File logging setup failed: {e}")


def main():
    """Main entry point with enhanced error handling and graceful shutdown"""
    print("🎯 MR BEN Live Trading System - Modular Version")
    print("=" * 60)
    
    config = None
    trading_system = None
    loop_manager = None
    
    try:
        # Initialize configuration
        print("[STARTUP] Loading configuration...")
        config = MT5Config()
        print(f"✅ Configuration loaded: {config.get_config_summary()}")
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger("Main")
        logger.info("🚀 Starting MR BEN Live Trading System")
        
        # Initialize trading system
        print("[STARTUP] Initializing trading system...")
        trading_system = TradingSystem(config)
        print("✅ Trading system initialized")
        
        # Initialize trading loop manager
        print("[STARTUP] Initializing trading loop...")
        loop_manager = TradingLoopManager(trading_system)
        print("✅ Trading loop manager initialized")
        
        # Start trading loop
        print("[STARTUP] Starting trading loop...")
        loop_manager.start()
        print("✅ Trading loop started")
        
        print("[SUCCESS] Trading system started successfully")
        print("Press Ctrl+C to stop gracefully")
        print("=" * 60)
        
        # Main monitoring loop
        while True:
            time.sleep(5)  # Check status every 5 seconds
            
            # Get and display status
            status = loop_manager.get_status()
            if status.get("running"):
                print(f"📊 Status: Cycle {status['cycle']}, "
                      f"Trades: {status['total_trades']}, "
                      f"Memory: {status['memory_mb']:.1f} MB")
            
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
        if loop_manager:
            loop_manager.stop()
        if trading_system:
            trading_system.cleanup()
            
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        if loop_manager:
            try:
                loop_manager.stop()
            except Exception as stop_error:
                print(f"[WARNING] Error during loop shutdown: {stop_error}")
        if trading_system:
            try:
                trading_system.cleanup()
            except Exception as cleanup_error:
                print(f"[WARNING] Error during cleanup: {cleanup_error}")
        raise
        
    finally:
        # Ensure cleanup
        if loop_manager and hasattr(loop_manager, 'running') and loop_manager.running:
            try:
                loop_manager.stop()
            except Exception:
                pass
        if trading_system:
            try:
                trading_system.cleanup()
            except Exception:
                pass
        
        print("[EXIT] Goodbye!")


if __name__ == "__main__":
    main()