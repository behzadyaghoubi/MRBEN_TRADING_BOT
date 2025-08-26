#!/usr/bin/env python3
"""
MR BEN Advanced Alerting Module
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AdvancedAlerting:
    """Advanced alerting system"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def send_alert(self, message: str, level: str = "INFO"):
        """Send alert"""
        self.logger.info(f"Alert [{level}]: {message}")
