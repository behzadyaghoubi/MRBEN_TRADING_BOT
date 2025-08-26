#!/usr/bin/env python3
"""
Run Dynamic Position Manager
Standalone script to run the dynamic position manager independently
"""

import os
import sys
import time
import signal
import json
from datetime import datetime
from dynamic_position_manager import DynamicPositionManager

class DynamicPositionManagerRunner:
    """Runner for the dynamic position manager."""
    
    def __init__(self, config_file: str = 'dynamic_position_config.json'):
        """Initialize the runner."""
        self.config_file = config_file
        self.manager = None
        self.running = False
        self.config = self._load_config()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                print(f"⚠️ Config file {self.config_file} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "position_management": {
                "enabled": True,
                "check_interval_seconds": 30,
                "max_positions": 2
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self):
        """Start the dynamic position manager."""
        try:
            print("🚀 Starting Dynamic Position Manager")
            print("=" * 60)
            
            # Create manager
            self.manager = DynamicPositionManager()
            
            if not self.manager.mt5_connected:
                print("❌ Failed to connect to MT5")
                return False
            
            # Apply configuration
            self._apply_config()
            
            # Start management loop
            self.running = True
            interval = self.config['position_management']['check_interval_seconds']
            
            print(f"📊 Management interval: {interval} seconds")
            print(f"📊 Max positions: {self.manager.max_positions}")
            print(f"📊 ATR multiplier: {self.manager.atr_multiplier}")
            print(f"📊 TP trail percent: {self.manager.tp_trail_percent}%")
            print("=" * 60)
            
            while self.running:
                try:
                    # Manage positions
                    self.manager.manage_positions()
                    
                    # Print status
                    self._print_status()
                    
                    # Wait for next iteration
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"❌ Error in management loop: {e}")
                    time.sleep(interval)
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting manager: {e}")
            return False
    
    def _apply_config(self):
        """Apply configuration to manager."""
        try:
            if 'atr_settings' in self.config['position_management']:
                atr_settings = self.config['position_management']['atr_settings']
                self.manager.atr_period = atr_settings.get('period', 14)
                self.manager.atr_multiplier = atr_settings.get('multiplier', 2.0)
                self.manager.min_atr_distance = atr_settings.get('min_distance_points', 10)
            
            if 'take_profit_settings' in self.config['position_management']:
                tp_settings = self.config['position_management']['take_profit_settings']
                self.manager.tp_trail_percent = tp_settings.get('trail_percent', 0.5)
            
            if 'max_positions' in self.config['position_management']:
                self.manager.max_positions = self.config['position_management']['max_positions']
                
            print("✅ Configuration applied successfully")
            
        except Exception as e:
            print(f"❌ Error applying configuration: {e}")
    
    def _print_status(self):
        """Print current status."""
        try:
            summary = self.manager.get_position_summary()
            
            if summary['total_positions'] > 0:
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"📊 {timestamp} | "
                      f"Positions: {summary['total_positions']} | "
                      f"Total Profit: {summary['total_profit']:.2f}")
                
                # Print individual position details
                for pos in summary['positions']:
                    print(f"   📈 {pos['ticket']} | "
                          f"{pos['type']} | "
                          f"Volume: {pos['volume']} | "
                          f"Entry: {pos['entry_price']:.2f} | "
                          f"Current: {pos['current_price']:.2f} | "
                          f"Profit: {pos['profit']:.2f}")
            else:
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"📊 {timestamp} | No active positions")
                
        except Exception as e:
            print(f"❌ Error printing status: {e}")
    
    def stop(self):
        """Stop the manager."""
        print("🛑 Stopping Dynamic Position Manager...")
        self.running = False
        
        if self.manager and self.manager.mt5_connected:
            try:
                mt5.shutdown()
                print("✅ MT5 connection closed")
            except:
                pass
        
        print("✅ Dynamic Position Manager stopped")

def main():
    """Main function."""
    print("🎯 Dynamic Position Manager Runner")
    print("=" * 60)
    
    # Create runner
    runner = DynamicPositionManagerRunner()
    
    try:
        # Start the manager
        if runner.start():
            print("✅ Manager started successfully")
        else:
            print("❌ Failed to start manager")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        runner.stop()

if __name__ == "__main__":
    main() 