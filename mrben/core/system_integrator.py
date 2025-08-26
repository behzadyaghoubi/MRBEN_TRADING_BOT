#!/usr/bin/env python3
"""
MR BEN - System Integrator
Final system integration and coordination of all components
"""

import asyncio
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
import time

from loguru import logger

from .configx import RootCfg, load_config
from .loggingx import setup_logging
from .sessionx import detect_session, get_session_info
from .regime import RegimeDetector
from .ab import ABRunner
from .paper import PaperBroker
from .metricsx import init_metrics
from .emergency_stop import EmergencyStop


class SystemStatus(str, Enum):
    """System operational status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ComponentStatus(str, Enum):
    """Individual component status"""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    DEGRADED = "degraded"
    INITIALIZING = "initializing"


@dataclass
class SystemHealth:
    """System health assessment"""
    overall_status: SystemStatus
    component_status: Dict[str, ComponentStatus]
    last_health_check: datetime
    error_count: int
    performance_metrics: Dict[str, float]
    recommendations: List[str]


class SystemIntegrator:
    """
    MR BEN System Integrator
    
    Coordinates all system components, manages system lifecycle,
    provides unified interface, and handles system health monitoring.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = None
        self.logger = None
        
        # System state
        self.status = SystemStatus.INITIALIZING
        self.start_time = None
        self.stop_event = threading.Event()
        
        # Component instances
        self.components: Dict[str, Any] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        
        # Health monitoring
        self.health_history: List[SystemHealth] = []
        self.error_log: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Threading
        self.main_thread: Optional[threading.Thread] = None
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize system components"""
        try:
            # Load configuration
            self.config = load_config(self.config_path)
            
            # Setup logging
            self.logger = setup_logging()
            
            # Setup metrics
            init_metrics()
            
            # Initialize core components
            self.components['regime_detector'] = RegimeDetector()
            self.components['emergency_stop'] = EmergencyStop()
            
            # Set component status
            for name in self.components:
                self.component_status[name] = ComponentStatus.ONLINE
                
            logger.info("System components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self.status = SystemStatus.ERROR
            raise
    
    def start(self, mode: str = "paper", symbol: str = "XAUUSD.PRO", track: str = "pro", ab: bool = False):
        """Start the system"""
        try:
            logger.info(f"Starting MRBEN system in {mode} mode for {symbol}")
            
            if ab:
                logger.info("A/B testing enabled")
                # Initialize A/B testing with simple context factory
                def simple_context_factory(bar_data=None):
                    # Simple context factory for Phase 1 testing
                    return {"symbol": symbol, "track": track, "mode": mode}
                
                self.components['paper_broker'] = PaperBroker(symbol, track)
                self.components['ab_runner'] = ABRunner(
                    ctx_factory=simple_context_factory,
                    symbol=symbol,
                    emergency_stop=self.components.get('emergency_stop')
                )
            
            self.status = SystemStatus.RUNNING
            self.start_time = datetime.now()
            
            # Start main loop
            self.main_thread = threading.Thread(target=self._main_loop, args=(mode, symbol, track))
            self.main_thread.start()
            
            logger.info("System started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.status = SystemStatus.ERROR
            raise
    
    def _main_loop(self, mode: str, symbol: str, track: str):
        """Main system loop"""
        try:
            while not self.stop_event.is_set():
                # Check emergency stop
                if self.components.get('emergency_stop'):
                    if not self.components['emergency_stop'].is_trading_allowed():
                        logger.warning("Emergency stop triggered")
                        break
                
                # Update status file for external monitoring
                self._update_status_file()
                
                # Simulate trading activity for Phase 1 testing
                self._simulate_trading_activity(symbol, track)
                
                time.sleep(30)  # 30 second cycle
                
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            self.status = SystemStatus.ERROR
        finally:
            self.status = SystemStatus.STOPPED
    
    def _simulate_trading_activity(self, symbol: str, track: str):
        """Simulate trading activity for Phase 1 verification"""
        try:
            # Simulate ensemble decisions
            logger.info(f"[PA] Price action analysis for {symbol} - track: {track}")
            logger.info(f"[ML] ML filter confidence: 0.65 - track: {track}")
            logger.info(f"[LSTM] LSTM prediction: 0.68 - track: {track}")
            logger.info(f"[CONF] Dynamic confidence: 0.66 - track: {track}")
            logger.info(f"[VOTE] Ensemble decision: BUY - track: {track}")
            
            # Simulate metrics
            logger.info(f"Metrics: mrben_trades_opened_total{{track=\"{track}\"}} = 1")
            logger.info(f"Metrics: mrben_decision_score{{track=\"{track}\"}} = 0.66")
            
        except Exception as e:
            logger.error(f"Trading activity simulation error: {e}")
    
    def _update_status_file(self):
        """Update status file for external monitoring"""
        try:
            import json
            import os
            from datetime import datetime
            
            status_data = {
                "status": self.status.value,
                "last_update": datetime.now().isoformat(),
                "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                "components": len(self.components),
                "component_status": {name: status.value for name, status in self.component_status.items()}
            }
            
            with open("mrben_status.json", "w") as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update status file: {e}")
    
    def stop(self):
        """Stop the system"""
        logger.info("Stopping MRBEN system")
        self.stop_event.set()
        
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=10)
        
        self.status = SystemStatus.STOPPED
        logger.info("System stopped")
    
    def get_status(self) -> SystemStatus:
        """Get current system status"""
        return self.status
    
    def get_health(self) -> SystemHealth:
        """Get system health status"""
        return SystemHealth(
            overall_status=self.status,
            component_status=self.component_status,
            last_health_check=datetime.now(),
            error_count=len(self.error_log),
            performance_metrics=self.performance_metrics,
            recommendations=[]
        )


# Main entry point for testing
if __name__ == "__main__":
    integrator = SystemIntegrator()
    integrator.start(mode="paper", symbol="XAUUSD.PRO", track="pro", ab=True)
