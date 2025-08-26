#!/usr/bin/env python3
"""
MR BEN - Trading Guard
Integrates emergency stop with all trading operations
"""

from __future__ import annotations
from typing import Optional, Callable, Any
from functools import wraps
from .emergency_stop import EmergencyStop, EmergencyState
from .loggingx import logger


class TradingGuard:
    """Guard that prevents trading operations during emergency stop"""
    
    def __init__(self, emergency_stop: EmergencyStop):
        self.emergency_stop = emergency_stop
        self.blocked_operations = 0
        self.last_blocked_operation = None
        
        # Register callbacks
        self.emergency_stop.add_emergency_callback(self._on_emergency_stop)
        self.emergency_stop.add_recovery_callback(self._on_recovery)
        
        logger.bind(evt="GUARD").info("trading_guard_initialized")
    
    def _on_emergency_stop(self, state: EmergencyState) -> None:
        """Handle emergency stop activation"""
        logger.bind(evt="GUARD").warning("trading_operations_blocked",
                                        reason=state.trigger_reason,
                                        source=state.trigger_source)
    
    def _on_recovery(self, state: EmergencyState) -> None:
        """Handle recovery from emergency stop"""
        logger.bind(evt="GUARD").info("trading_operations_resumed",
                                     recovery_source=state.trigger_source)
    
    def guard_trading_operation(self, operation_name: str = "trading_operation"):
        """Decorator to guard trading operations"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.emergency_stop.is_trading_allowed():
                    self.blocked_operations += 1
                    self.last_blocked_operation = operation_name
                    
                    logger.bind(evt="GUARD").warning("trading_operation_blocked",
                                                   operation=operation_name,
                                                   blocked_count=self.blocked_operations)
                    
                    # Return safe default or raise exception
                    return self._get_safe_default(operation_name)
                
                # Trading allowed, proceed with operation
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.bind(evt="GUARD").error("trading_operation_error",
                                                 operation=operation_name,
                                                 error=str(e))
                    raise
            
            return wrapper
        return decorator
    
    def guard_decision_making(self, operation_name: str = "decision_making"):
        """Decorator to guard decision making operations"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.emergency_stop.is_trading_allowed():
                    self.blocked_operations += 1
                    self.last_blocked_operation = operation_name
                    
                    logger.bind(evt="GUARD").warning("decision_operation_blocked",
                                                   operation=operation_name,
                                                   blocked_count=self.blocked_operations)
                    
                    # Return safe HOLD decision
                    return self._get_safe_decision()
                
                # Decision making allowed, proceed with operation
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.bind(evt="GUARD").error("decision_operation_error",
                                                 operation=operation_name,
                                                 error=str(e))
                    raise
            
            return wrapper
        return decorator
    
    def guard_order_execution(self, operation_name: str = "order_execution"):
        """Decorator to guard order execution operations"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.emergency_stop.is_trading_allowed():
                    self.blocked_operations += 1
                    self.last_blocked_operation = operation_name
                    
                    logger.bind(evt="GUARD").warning("order_execution_blocked",
                                                   operation=operation_name,
                                                   blocked_count=self.blocked_operations)
                    
                    # Return safe order result
                    return self._get_safe_order_result()
                
                # Order execution allowed, proceed with operation
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.bind(evt="GUARD").error("order_execution_error",
                                                 operation=operation_name,
                                                 error=str(e))
                    raise
            
            return wrapper
        return decorator
    
    def _get_safe_default(self, operation_name: str) -> Any:
        """Get safe default value for blocked operation"""
        # Return appropriate safe defaults based on operation type
        if "decision" in operation_name.lower():
            return self._get_safe_decision()
        elif "order" in operation_name.lower():
            return self._get_safe_order_result()
        elif "trade" in operation_name.lower():
            return self._get_safe_trade_result()
        else:
            return None
    
    def _get_safe_decision(self):
        """Get safe decision when trading is blocked"""
        from .typesx import DecisionCard
        
        return DecisionCard(
            action="HOLD",
            dir=0,
            reason="emergency_stop_active",
            score=0.0,
            dyn_conf=0.0,
            track="emergency"
        )
    
    def _get_safe_order_result(self):
        """Get safe order result when trading is blocked"""
        from .order_management import OrderResult, OrderStatus
        
        return OrderResult(
            success=False,
            order_id=None,
            status=OrderStatus.REJECTED,
            error_message="Trading blocked by emergency stop",
            execution_time=None,
            slippage_points=0.0
        )
    
    def _get_safe_trade_result(self):
        """Get safe trade result when trading is blocked"""
        return {
            "success": False,
            "reason": "emergency_stop_active",
            "timestamp": None,
            "details": "Trading operation blocked by emergency stop"
        }
    
    def check_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        return self.emergency_stop.is_trading_allowed()
    
    def get_emergency_state(self) -> EmergencyState:
        """Get current emergency state"""
        return self.emergency_stop.get_state()
    
    def get_blocked_operations_count(self) -> int:
        """Get count of blocked operations"""
        return self.blocked_operations
    
    def get_last_blocked_operation(self) -> Optional[str]:
        """Get name of last blocked operation"""
        return self.last_blocked_operation
    
    def reset_blocked_operations_count(self) -> None:
        """Reset blocked operations counter"""
        self.blocked_operations = 0
        self.last_blocked_operation = None
        
        logger.bind(evt="GUARD").info("blocked_operations_counter_reset")
    
    def get_status_summary(self) -> dict:
        """Get trading guard status summary"""
        emergency_state = self.emergency_stop.get_state()
        
        return {
            "trading_allowed": self.emergency_stop.is_trading_allowed(),
            "emergency_active": emergency_state.is_active,
            "emergency_reason": emergency_state.trigger_reason,
            "emergency_source": emergency_state.trigger_source,
            "emergency_triggered_at": emergency_state.triggered_at.isoformat() if emergency_state.triggered_at else None,
            "blocked_operations": self.blocked_operations,
            "last_blocked_operation": self.last_blocked_operation,
            "last_check": emergency_state.last_check.isoformat(),
            "checks_count": emergency_state.checks_count
        }


# Convenience functions for quick checks
def is_trading_allowed(guard: TradingGuard) -> bool:
    """Quick check if trading is allowed"""
    return guard.check_trading_allowed()


def require_trading_allowed(guard: TradingGuard, operation_name: str = "operation"):
    """Decorator that requires trading to be allowed"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not guard.check_trading_allowed():
                logger.bind(evt="GUARD").warning("operation_requires_trading_allowed",
                                               operation=operation_name)
                raise RuntimeError(f"Operation '{operation_name}' requires trading to be allowed")
            return func(*args, **kwargs)
        return wrapper
    return decorator
