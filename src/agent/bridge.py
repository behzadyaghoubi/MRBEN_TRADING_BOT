#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Agent Bridge Module
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AgentBridge:
    """Simple agent bridge for basic supervision"""
    
    def __init__(self, config: Dict[str, Any], mode: str = "guard"):
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(f"AgentBridge")
        self.logger.info(f"🤖 Agent bridge initialized in {mode} mode")
    
    def review_and_maybe_execute(self, decision_card, context):
        """Review trading decision and optionally modify"""
        try:
            # Basic guard mode - just log decisions
            self.logger.info(f"🔍 Reviewing decision: {decision_card.signal_src} | Conf: {decision_card.adj_conf:.3f}")
            
            # In guard mode, we don't modify decisions, just monitor
            if self.mode == "guard":
                return None  # No action taken
            
            # In other modes, we could implement decision modification
            return None
            
        except Exception as e:
            self.logger.error(f"Error in decision review: {e}")
            return None
    
    def on_health_event(self, event: Dict[str, Any]):
        """Handle health events from the trading system"""
        try:
            severity = event.get("severity", "INFO")
            kind = event.get("kind", "UNKNOWN")
            message = event.get("message", "No message")
            
            if severity == "ERROR":
                self.logger.error(f"🚨 Health Event [{kind}]: {message}")
            elif severity == "WARN":
                self.logger.warning(f"⚠️ Health Event [{kind}]: {message}")
            else:
                self.logger.info(f"ℹ️ Health Event [{kind}]: {message}")
                
        except Exception as e:
            self.logger.error(f"Error handling health event: {e}")
    
    def block_until_stopped(self):
        """Block execution until the agent is stopped"""
        try:
            self.logger.info(f"🤖 Agent running in {self.mode} mode - press Ctrl+C to stop")
            import time
            while True:
                time.sleep(1)
                # In a real implementation, this would check for stop signals
                # For now, just keep running until interrupted
        except KeyboardInterrupt:
            self.logger.info("🛑 Agent stopped by user")
        except Exception as e:
            self.logger.error(f"Agent error: {e}")
    
    def stop(self):
        """Stop the agent"""
        self.logger.info("🛑 Stopping agent")

def maybe_start_agent(config: Dict[str, Any], mode: str = "guard"):
    """Start the AI agent if configured"""
    try:
        # Check if agent is enabled
        agent_config = config.get("agent", {})
        agent_enabled = agent_config.get("enabled", True)  # Default to enabled
        
        if agent_enabled:
            logger.info(f"🤖 Starting AI agent in {mode} mode")
            return AgentBridge(config, mode)
        else:
            logger.info("AI agent supervision disabled")
            return None
    except Exception as e:
        logger.error(f"Error starting agent: {e}")
        return None
