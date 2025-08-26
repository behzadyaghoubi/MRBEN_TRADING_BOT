#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MR BEN Configuration Module
"""

import json
import os
from typing import Dict, Any, Optional

class MT5Config:
    """MT5 configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "MT5": {"login": 12345, "password": "password", "server": "MetaQuotes-Demo"},
            "SYMBOL": "XAUUSD.PRO",
            "TIMEFRAME": "15m",
            "DEMO_MODE": True,
            "AGENT_MODE": "guard",
            "ENABLE_REGIME": True,
            "ENABLE_CONFORMAL": True,
            "LOG_LEVEL": "INFO"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_symbol(self) -> str:
        """Get trading symbol"""
        return self.config.get("SYMBOL", "XAUUSD.PRO")
    
    def get_timeframe(self) -> str:
        """Get trading timeframe"""
        return self.config.get("TIMEFRAME", "15m")
    
    def is_demo_mode(self) -> bool:
        """Check if demo mode is enabled"""
        return self.config.get("DEMO_MODE", True)
