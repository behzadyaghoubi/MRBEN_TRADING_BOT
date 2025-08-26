#!/usr/bin/env python3
"""
MR BEN AI System - Safe Execution Script
Handles character encoding issues and runs the comprehensive update safely
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

def setup_logging():
    """Setup logging for the execution"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f'logs/execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def clean_command(command):
    """Clean command from unwanted characters"""
    # Remove Persian/Arabic characters that might be added
    unwanted_prefixes = ['رز', 'ز', 'ر', 'زpython', 'رpython']
    
    for prefix in unwanted_prefixes:
        if command.startswith(prefix):
            command = command[len(prefix):]
            print(f"Removed unwanted prefix: {prefix}")
    
    return command.strip()

def run_command_safely(command, description=""):
    """Run command safely with error handling"""
    logger = logging.getLogger(__name__)
    
    # Clean the command
    clean_cmd = clean_command(command)
    
    logger.info(f"Executing: {description}")
    logger.info(f"Command: {clean_cmd}")
    
    try:
        # Run the command
        result = subprocess.run(
            clean_cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        # Log results
        if result.stdout:
            logger.info("STDOUT:")
            logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("STDERR:")
            logger.warning(result.stderr)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            return True
        else:
            logger.error(f"❌ {description} failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error executing {description}: {e}")
        return False

def main():
    """Main execution function"""
    logger = setup_logging()
    
    print("🎯 MR BEN AI System - Safe Execution")
    print("=" * 50)
    logger.info("Starting MR BEN AI System safe execution")
    
    # Test system components first
    print("\n📊 Step 1: Testing system components...")
    success = run_command_safely(
        "python test_system_update.py",
        "System component test"
    )
    
    if not success:
        print("❌ System component test failed. Please check the logs.")
        return
    
    # Run the comprehensive update
    print("\n🚀 Step 2: Running comprehensive system update...")
    success = run_command_safely(
        "python fixed_comprehensive_update.py",
        "Comprehensive system update"
    )
    
    if success:
        print("\n✅ MR BEN AI System update completed successfully!")
        print("📋 Check the logs/ directory for detailed reports")
    else:
        print("\n❌ System update failed. Please check the logs for details.")
    
    logger.info("MR BEN AI System execution completed")

if __name__ == "__main__":
    main() 