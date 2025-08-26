#!/usr/bin/env python3
"""
MR BEN - Main Entry Point
Professional-Grade Live Trading System

This is the main entry point for the MR BEN trading system.
It initializes the System Integrator and provides command-line interface
for system control and monitoring.
"""

import sys
import os
import argparse
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.system_integrator import SystemIntegrator, SystemStatus
from loguru import logger
import json


class MRBENSystem:
    """Main MR BEN Trading System Controller"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.integrator: Optional[SystemIntegrator] = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("MR BEN Trading System initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start(self, mode: str = "paper", symbol: str = "XAUUSD.PRO", ab: bool = False) -> bool:
        """Start the MR BEN trading system"""
        try:
            logger.info(f"Starting MR BEN Trading System in {mode} mode with symbol {symbol}, A/B testing: {ab}")
            
            # Initialize system integrator
            self.integrator = SystemIntegrator(self.config_path)
            
            # Start the system integrator with specified parameters
            self.integrator.start(mode=mode, symbol=symbol, track="pro", ab=ab)
            
            # Wait for system to be ready
            max_wait_time = 60  # seconds
            wait_time = 0
            while (self.integrator.status not in [SystemStatus.RUNNING, SystemStatus.ERROR] 
                   and wait_time < max_wait_time):
                time.sleep(1)
                wait_time += 1
                logger.debug(f"Waiting for system to be ready... ({wait_time}s)")
            
            if self.integrator.status == SystemStatus.RUNNING:
                self.running = True
                logger.info("MR BEN Trading System started successfully")
                return True
            else:
                logger.error(f"System failed to start, status: {self.integrator.status}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the MR BEN trading system"""
        try:
            if self.integrator and self.running:
                logger.info("Stopping MR BEN Trading System...")
                
                # Stop the system integrator
                self.integrator.stop()
                stop_success = True
                
                if stop_success:
                    self.running = False
                    logger.info("MR BEN Trading System stopped successfully")
                    return True
                else:
                    logger.error("Failed to stop system gracefully")
                    return False
            else:
                logger.info("System is not running")
                return True
                
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            return False
    
    def restart(self) -> bool:
        """Restart the MR BEN trading system"""
        try:
            logger.info("Restarting MR BEN Trading System...")
            
            # Stop current system
            if not self.stop():
                logger.error("Failed to stop system during restart")
                return False
            
            # Wait a moment
            time.sleep(2)
            
            # Start system again
            if not self.start():
                logger.error("Failed to start system during restart")
                return False
            
            logger.info("MR BEN Trading System restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart system: {e}")
            return False
    
    def status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            if self.integrator:
                return {
                    "status": self.integrator.get_status().value,
                    "start_time": self.integrator.start_time.isoformat() if self.integrator.start_time else None,
                    "components": len(self.integrator.components)
                }
            else:
                # Check if system is running by looking for a status file
                import os
                status_file = "mrben_status.json"
                
                if os.path.exists(status_file):
                    try:
                        import json
                        with open(status_file, 'r') as f:
                            status_data = json.load(f)
                        return {
                            "status": "running",
                            "note": "System is running in another process",
                            "last_update": status_data.get("last_update"),
                            "uptime": status_data.get("uptime")
                        }
                    except:
                        pass
                
                return {"error": "System not initialized"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def health(self) -> Dict[str, Any]:
        """Get system health information"""
        try:
            if self.integrator:
                return self.integrator.get_health().__dict__
            else:
                return {"error": "System not initialized"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def pause(self) -> bool:
        """Pause system operations"""
        try:
            if self.integrator and self.running:
                return self.integrator.pause_system()
            else:
                logger.warning("Cannot pause system that is not running")
                return False
                
        except Exception as e:
            logger.error(f"Failed to pause system: {e}")
            return False
    
    def resume(self) -> bool:
        """Resume system operations"""
        try:
            if self.integrator and self.running:
                return self.integrator.resume_system()
            else:
                logger.warning("Cannot resume system that is not running")
                return False
                
        except Exception as e:
            logger.error(f"Failed to resume system: {e}")
            return False
    
    def run_interactive(self):
        """Run the system in interactive mode"""
        try:
            logger.info("Starting MR BEN Trading System in interactive mode...")
            
            if not self.start():
                logger.error("Failed to start system")
                return
            
            logger.info("System is running. Press Ctrl+C to stop.")
            
            # Keep the system running
            try:
                while self.running:
                    time.sleep(1)
                    
                    # Check if system is still healthy
                    if self.integrator and self.integrator.status == SystemStatus.ERROR:
                        logger.error("System entered error state")
                        break
                        
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
            finally:
                self.stop()
                
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            self.stop()
    
    def run_daemon(self):
        """Run the system in daemon mode"""
        try:
            logger.info("Starting MR BEN Trading System in daemon mode...")
            
            if not self.start():
                logger.error("Failed to start system")
                return False
            
            logger.info("System is running in daemon mode")
            
            # Keep the system running
            while self.running:
                time.sleep(30)  # Check every 30 seconds
                
                # Check system health
                if self.integrator:
                    health = self.integrator._check_system_health()
                    if health.overall_status == SystemStatus.ERROR:
                        logger.error("System health check failed")
                        break
            
            return True
            
        except Exception as e:
            logger.error(f"Error in daemon mode: {e}")
            return False
        finally:
            self.stop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MR BEN Professional-Grade Live Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py start                    # Start the system
  python main.py stop                     # Stop the system
  python main.py restart                  # Restart the system
  python main.py status                  # Show system status
  python main.py health                  # Show system health
  python main.py interactive             # Run in interactive mode
  python main.py daemon                  # Run in daemon mode
        """
    )
    
    parser.add_argument(
        'command',
        choices=['start', 'stop', 'restart', 'status', 'health', 'interactive', 'daemon'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Configuration file path (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode: paper or live (default: paper)'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        default='XAUUSD.PRO',
        help='Trading symbol (default: XAUUSD.PRO)'
    )
    
    parser.add_argument(
        '--ab', '-a',
        action='store_true',
        help='Enable A/B testing'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Create system instance
    system = MRBENSystem(args.config)
    
    try:
        if args.command == 'start':
            if system.start(mode=args.mode, symbol=args.symbol, ab=args.ab):
                print("✓ MR BEN Trading System started successfully")
                sys.exit(0)
            else:
                print("✗ Failed to start MR BEN Trading System")
                sys.exit(1)
                
        elif args.command == 'stop':
            if system.stop():
                print("✓ MR BEN Trading System stopped successfully")
                sys.exit(0)
            else:
                print("✗ Failed to stop MR BEN Trading System")
                sys.exit(1)
                
        elif args.command == 'restart':
            if system.restart():
                print("✓ MR BEN Trading System restarted successfully")
                sys.exit(0)
            else:
                print("✗ Failed to restart MR BEN Trading System")
                sys.exit(1)
                
        elif args.command == 'status':
            status = system.status()
            print("MR BEN Trading System Status:")
            print(json.dumps(status, indent=2, default=str))
            sys.exit(0)
            
        elif args.command == 'health':
            health = system.health()
            print("MR BEN Trading System Health:")
            print(json.dumps(health, indent=2, default=str))
            sys.exit(0)
            
        elif args.command == 'interactive':
            system.run_interactive()
            sys.exit(0)
            
        elif args.command == 'daemon':
            if system.run_daemon():
                print("✓ MR BEN Trading System daemon completed successfully")
                sys.exit(0)
            else:
                print("✗ MR BEN Trading System daemon failed")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        system.stop()
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        system.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
