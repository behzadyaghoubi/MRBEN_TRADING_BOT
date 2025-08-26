#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Governor - Layer 3 of MR BEN AI Architecture
Safety and risk management system with kill-switch capabilities
"""
import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

class ExecutionMode(Enum):
    """Execution modes for AI control"""
    SHADOW = "shadow"        # AI only suggests, no execution
    COPILOT = "copilot"      # AI suggests, human/rules can override
    AUTOPILOT = "autopilot"  # AI executes with safety constraints

class SessionType(Enum):
    """Trading session types"""
    LONDON = "london"
    NEWYORK = "newyork" 
    ASIAN = "asian"
    OVERLAP = "overlap"
    CLOSED = "closed"

class RiskGovernor:
    """
    Layer 3 Risk Governor - Final safety and risk management layer
    
    Responsibilities:
    - Enforce all risk limits and constraints
    - Kill-switch functionality
    - Session and time-based filtering
    - News and economic event blocking
    - Execution mode management (Shadow/Co-Pilot/Autopilot)
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.logger = logging.getLogger("RiskGovernor")
        self.config = self._load_config(config_path)
        
        # Current session state
        self.session_state = {
            "equity_start": 10000.0,
            "daily_pnl": 0.0,
            "trades_today": 0,
            "last_trade_time": None,
            "emergency_stop": False,
            "kill_switch_active": False
        }
        
        # Risk metrics tracking
        self.risk_metrics = {
            "max_drawdown_today": 0.0,
            "consecutive_losses": 0,
            "equity_curve": [],
            "hourly_pnl": {}
        }
        
        # News/Event blocking (placeholder)
        self.news_block_active = False
        self.blocked_until = None
        
        self.logger.info(f"RiskGovernor initialized in {self.get_execution_mode().value} mode")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load risk configuration"""
        default_config = {
            "ai_control": {
                "mode": "copilot",
                "emergency_stop_threshold": 0.02,  # 2% daily loss
                "kill_switch_threshold": 0.05,     # 5% total loss
                "max_consecutive_losses": 5
            },
            "risk": {
                "max_daily_loss": 0.02,
                "max_trades_per_day": 10,
                "max_spread_points": 100,
                "session_filter": True,
                "allowed_sessions": ["london", "newyork", "overlap"],
                "cooldown_minutes": 30,
                "position_limit": 2
            },
            "safety": {
                "news_block_enabled": False,
                "high_impact_block_minutes": 60,
                "weekend_trading": False,
                "holiday_trading": False
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return {**default_config, **config}
            except Exception as e:
                self.logger.warning(f"Error loading config: {e}, using defaults")
                return default_config
        else:
            return default_config
    
    def validate_trade(self, proposed_trade: Dict, market_snapshot: Dict) -> Dict:
        """
        Final validation of proposed trade through all safety checks
        
        Args:
            proposed_trade: Decision from PolicyBrain
            market_snapshot: Current market conditions
            
        Returns:
            {
                "approved": bool,
                "reason": str,
                "modified_trade": dict,  # Potentially modified parameters
                "risk_assessment": dict
            }
        """
        try:
            # Skip validation for HOLD decisions
            if proposed_trade.get("decision") == "HOLD":
                return {
                    "approved": True,
                    "reason": "hold_decision",
                    "modified_trade": proposed_trade,
                    "risk_assessment": {}
                }
            
            # Check execution mode
            mode_check = self._check_execution_mode(proposed_trade)
            if not mode_check["approved"]:
                return mode_check
            
            # Emergency and kill-switch checks
            emergency_check = self._check_emergency_conditions()
            if not emergency_check["approved"]:
                return emergency_check
            
            # Risk limit checks
            risk_check = self._check_risk_limits(proposed_trade, market_snapshot)
            if not risk_check["approved"]:
                return risk_check
            
            # Session and time checks
            session_check = self._check_session_constraints(market_snapshot)
            if not session_check["approved"]:
                return session_check
            
            # Market condition checks
            market_check = self._check_market_conditions(market_snapshot)
            if not market_check["approved"]:
                return market_check
            
            # News and event checks
            news_check = self._check_news_events()
            if not news_check["approved"]:
                return news_check
            
            # Position and sizing validation
            position_check = self._validate_position_sizing(proposed_trade, market_snapshot)
            if not position_check["approved"]:
                return position_check
            
            # All checks passed - approve with risk assessment
            risk_assessment = self._calculate_risk_assessment(proposed_trade, market_snapshot)
            
            return {
                "approved": True,
                "reason": "all_checks_passed",
                "modified_trade": proposed_trade,
                "risk_assessment": risk_assessment
            }
            
        except Exception as e:
            self.logger.error(f"Error in trade validation: {e}")
            return {
                "approved": False,
                "reason": "validation_error",
                "modified_trade": proposed_trade,
                "risk_assessment": {}
            }
    
    def _check_execution_mode(self, proposed_trade: Dict) -> Dict:
        """Check if current execution mode allows the trade"""
        mode = self.get_execution_mode()
        
        if mode == ExecutionMode.SHADOW:
            return {
                "approved": False,
                "reason": "shadow_mode_active",
                "modified_trade": proposed_trade,
                "risk_assessment": {}
            }
        
        # COPILOT and AUTOPILOT modes allow execution
        return {
            "approved": True,
            "reason": "execution_mode_ok",
            "modified_trade": proposed_trade,
            "risk_assessment": {}
        }
    
    def _check_emergency_conditions(self) -> Dict:
        """Check for emergency stop and kill-switch conditions"""
        
        # Check kill-switch
        if self.session_state["kill_switch_active"]:
            return {
                "approved": False,
                "reason": "kill_switch_active",
                "modified_trade": {},
                "risk_assessment": {}
            }
        
        # Check emergency stop threshold
        emergency_threshold = self.config["ai_control"]["emergency_stop_threshold"]
        if self.session_state["daily_pnl"] <= -abs(emergency_threshold * self.session_state["equity_start"]):
            self.session_state["emergency_stop"] = True
            self.logger.critical(f"EMERGENCY STOP: Daily loss threshold breached!")
            return {
                "approved": False,
                "reason": "emergency_stop_threshold",
                "modified_trade": {},
                "risk_assessment": {}
            }
        
        # Check kill-switch threshold
        kill_threshold = self.config["ai_control"]["kill_switch_threshold"]
        current_equity = self.session_state["equity_start"] + self.session_state["daily_pnl"]
        total_loss = (self.session_state["equity_start"] - current_equity) / self.session_state["equity_start"]
        
        if total_loss >= kill_threshold:
            self.activate_kill_switch("total_loss_threshold")
            return {
                "approved": False,
                "reason": "kill_switch_triggered",
                "modified_trade": {},
                "risk_assessment": {}
            }
        
        return {
            "approved": True,
            "reason": "emergency_checks_ok",
            "modified_trade": {},
            "risk_assessment": {}
        }
    
    def _check_risk_limits(self, proposed_trade: Dict, market_snapshot: Dict) -> Dict:
        """Check standard risk management limits"""
        
        # Daily loss limit
        max_daily_loss = self.config["risk"]["max_daily_loss"]
        if self.session_state["daily_pnl"] <= -abs(max_daily_loss * self.session_state["equity_start"]):
            return {
                "approved": False,
                "reason": "daily_loss_limit",
                "modified_trade": proposed_trade,
                "risk_assessment": {}
            }
        
        # Daily trade limit
        max_trades = self.config["risk"]["max_trades_per_day"]
        if self.session_state["trades_today"] >= max_trades:
            return {
                "approved": False,
                "reason": "daily_trade_limit",
                "modified_trade": proposed_trade,
                "risk_assessment": {}
            }
        
        # Consecutive loss limit
        max_consecutive = self.config["ai_control"]["max_consecutive_losses"]
        if self.risk_metrics["consecutive_losses"] >= max_consecutive:
            return {
                "approved": False,
                "reason": "consecutive_loss_limit",
                "modified_trade": proposed_trade,
                "risk_assessment": {}
            }
        
        return {
            "approved": True,
            "reason": "risk_limits_ok",
            "modified_trade": proposed_trade,
            "risk_assessment": {}
        }
    
    def _check_session_constraints(self, market_snapshot: Dict) -> Dict:
        """Check trading session and time constraints"""
        
        if not self.config["risk"]["session_filter"]:
            return {
                "approved": True,
                "reason": "session_filter_disabled",
                "modified_trade": {},
                "risk_assessment": {}
            }
        
        current_session = self._get_current_session()
        allowed_sessions = self.config["risk"]["allowed_sessions"]
        
        if current_session.value not in allowed_sessions:
            return {
                "approved": False,
                "reason": f"session_not_allowed_{current_session.value}",
                "modified_trade": {},
                "risk_assessment": {}
            }
        
        # Check cooldown period
        cooldown_minutes = self.config["risk"]["cooldown_minutes"]
        if self.session_state["last_trade_time"]:
            time_since_last = datetime.now() - self.session_state["last_trade_time"]
            if time_since_last.total_seconds() < cooldown_minutes * 60:
                return {
                    "approved": False,
                    "reason": "cooldown_period_active",
                    "modified_trade": {},
                    "risk_assessment": {}
                }
        
        return {
            "approved": True,
            "reason": "session_constraints_ok",
            "modified_trade": {},
            "risk_assessment": {}
        }
    
    def _check_market_conditions(self, market_snapshot: Dict) -> Dict:
        """Check market conditions like spread, volatility, etc."""
        
        # Spread check
        spread = market_snapshot.get("spread", 0)
        max_spread = self.config["risk"]["max_spread_points"]
        
        if spread > max_spread:
            return {
                "approved": False,
                "reason": "spread_too_wide",
                "modified_trade": {},
                "risk_assessment": {"spread": spread, "max_allowed": max_spread}
            }
        
        # Add more market condition checks here (volatility, liquidity, etc.)
        
        return {
            "approved": True,
            "reason": "market_conditions_ok",
            "modified_trade": {},
            "risk_assessment": {"spread": spread}
        }
    
    def _check_news_events(self) -> Dict:
        """Check for news events and economic calendar blocks"""
        
        if not self.config["safety"]["news_block_enabled"]:
            return {
                "approved": True,
                "reason": "news_block_disabled",
                "modified_trade": {},
                "risk_assessment": {}
            }
        
        # Check if currently in news block period
        if self.news_block_active and self.blocked_until:
            if datetime.now() < self.blocked_until:
                return {
                    "approved": False,
                    "reason": "news_block_active",
                    "modified_trade": {},
                    "risk_assessment": {"blocked_until": self.blocked_until.isoformat()}
                }
            else:
                # Block period expired
                self.news_block_active = False
                self.blocked_until = None
        
        return {
            "approved": True,
            "reason": "news_events_ok",
            "modified_trade": {},
            "risk_assessment": {}
        }
    
    def _validate_position_sizing(self, proposed_trade: Dict, market_snapshot: Dict) -> Dict:
        """Validate and potentially modify position sizing"""
        
        size_factor = proposed_trade.get("size_factor", 1.0)
        
        # Position limit check
        position_limit = self.config["risk"]["position_limit"]
        # This would need to check current open positions from MT5
        # For now, just validate the size factor
        
        if size_factor > 2.0:  # Maximum size multiplier
            size_factor = 2.0
            proposed_trade["size_factor"] = size_factor
            self.logger.warning("Position size capped at 2x")
        
        if size_factor < 0.1:  # Minimum size multiplier
            size_factor = 0.1
            proposed_trade["size_factor"] = size_factor
            self.logger.warning("Position size floored at 0.1x")
        
        return {
            "approved": True,
            "reason": "position_sizing_ok",
            "modified_trade": proposed_trade,
            "risk_assessment": {"adjusted_size_factor": size_factor}
        }
    
    def _calculate_risk_assessment(self, proposed_trade: Dict, market_snapshot: Dict) -> Dict:
        """Calculate comprehensive risk assessment for the trade"""
        
        return {
            "daily_pnl": self.session_state["daily_pnl"],
            "trades_today": self.session_state["trades_today"],
            "consecutive_losses": self.risk_metrics["consecutive_losses"],
            "max_drawdown_today": self.risk_metrics["max_drawdown_today"],
            "equity_at_risk": proposed_trade.get("size_factor", 1.0) * self.config["risk"]["base_risk"],
            "expected_value": proposed_trade.get("expected_value", 0.0),
            "session": self._get_current_session().value,
            "spread": market_snapshot.get("spread", 0)
        }
    
    def _get_current_session(self) -> SessionType:
        """Determine current trading session based on time"""
        now = datetime.now()
        hour = now.hour
        
        # Simplified session detection (UTC time)
        if 7 <= hour < 16:
            return SessionType.LONDON
        elif 13 <= hour < 22:
            if 13 <= hour < 16:
                return SessionType.OVERLAP  # London-NY overlap
            else:
                return SessionType.NEWYORK
        elif 22 <= hour or hour < 7:
            return SessionType.ASIAN
        else:
            return SessionType.CLOSED
    
    def get_execution_mode(self) -> ExecutionMode:
        """Get current execution mode"""
        mode_str = self.config["ai_control"]["mode"].lower()
        try:
            return ExecutionMode(mode_str)
        except ValueError:
            self.logger.warning(f"Invalid execution mode: {mode_str}, defaulting to COPILOT")
            return ExecutionMode.COPILOT
    
    def activate_kill_switch(self, reason: str):
        """Activate emergency kill-switch"""
        self.session_state["kill_switch_active"] = True
        self.logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        
        # Save kill-switch state to file for persistence
        kill_switch_file = "logs/kill_switch.json"
        os.makedirs(os.path.dirname(kill_switch_file), exist_ok=True)
        with open(kill_switch_file, 'w') as f:
            json.dump({
                "active": True,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "session_state": self.session_state
            }, f, indent=2)
    
    def deactivate_kill_switch(self, authorized_by: str):
        """Deactivate kill-switch (requires authorization)"""
        self.session_state["kill_switch_active"] = False
        self.logger.info(f"Kill-switch deactivated by: {authorized_by}")
        
        # Update kill-switch file
        kill_switch_file = "logs/kill_switch.json"
        if os.path.exists(kill_switch_file):
            with open(kill_switch_file, 'w') as f:
                json.dump({
                    "active": False,
                    "deactivated_by": authorized_by,
                    "deactivated_at": datetime.now().isoformat()
                }, f, indent=2)
    
    def update_session_state(self, trade_result: Dict):
        """Update session state after a trade"""
        if trade_result.get("executed", False):
            self.session_state["trades_today"] += 1
            self.session_state["last_trade_time"] = datetime.now()
            
            # Update PnL if provided
            pnl = trade_result.get("pnl", 0.0)
            self.session_state["daily_pnl"] += pnl
            
            # Update risk metrics
            if pnl < 0:
                self.risk_metrics["consecutive_losses"] += 1
            else:
                self.risk_metrics["consecutive_losses"] = 0
            
            # Update drawdown
            current_dd = abs(min(0, self.session_state["daily_pnl"]))
            self.risk_metrics["max_drawdown_today"] = max(
                self.risk_metrics["max_drawdown_today"], current_dd
            )
    
    def get_status_report(self) -> Dict:
        """Get comprehensive status report"""
        return {
            "execution_mode": self.get_execution_mode().value,
            "emergency_stop": self.session_state["emergency_stop"],
            "kill_switch_active": self.session_state["kill_switch_active"],
            "session_state": self.session_state,
            "risk_metrics": self.risk_metrics,
            "current_session": self._get_current_session().value,
            "news_block_active": self.news_block_active
        }
