#!/usr/bin/env python3
"""
MR BEN - Order Management System
MT5 integration with order filling modes and execution optimization
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .loggingx import logger


class OrderType(Enum):
    """Order types"""
    BUY = "buy"
    SELL = "sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"


class OrderFillingMode(Enum):
    """MT5 order filling modes"""
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    RETURN = "return"  # Return


class OrderStatus(Enum):
    """Order lifecycle status"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = ""
    magic: int = 0
    filling_mode: OrderFillingMode = OrderFillingMode.IOC


@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    ticket: Optional[int] = None
    volume: float = 0.0
    price: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0
    execution_time: Optional[datetime] = None
    error_code: Optional[int] = None
    error_description: Optional[str] = None
    partial_fills: List[Dict] = None
    
    def __post_init__(self):
        if self.partial_fills is None:
            self.partial_fills = []


class MT5Connector:
    """
    MetaTrader 5 Connection Manager
    Handles connection, authentication, and basic operations
    """
    
    def __init__(self, config):
        self.enabled = getattr(config.order_management, 'mt5_enabled', True)
        self.login = getattr(config.order_management, 'mt5_login', 0)
        self.password = getattr(config.order_management, 'mt5_password', "")
        self.server = getattr(config.order_management, 'mt5_server', "")
        self.timeout = getattr(config.order_management, 'mt5_timeout', 60000)
        
        self.connected = False
        self.account_info = None
        
        # Import MT5 only when needed
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
        except ImportError:
            logger.warning("MetaTrader5 not available - using mock mode")
            self.mt5 = None
            self.enabled = False
    
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        if not self.enabled or not self.mt5:
            logger.warning("MT5 not enabled or available")
            return False
        
        try:
            # Initialize MT5
            if not self.mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # Login to account
            if not self.mt5.login(login=self.login, password=self.password, server=self.server):
                logger.error("MT5 login failed")
                return False
            
            # Get account info
            self.account_info = self.mt5.account_info()
            if not self.account_info:
                logger.error("Failed to get account info")
                return False
            
            self.connected = True
            logger.bind(evt="ORDER").info("mt5_connected",
                                        login=self.login,
                                        server=self.server,
                                        balance=self.account_info.balance)
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.mt5 and self.connected:
            self.mt5.shutdown()
            self.connected = False
            logger.bind(evt="ORDER").info("mt5_disconnected")
    
    def get_account_info(self) -> Optional[Dict]:
        """Get current account information"""
        if not self.connected or not self.account_info:
            return None
        
        return {
            "balance": self.account_info.balance,
            "equity": self.account_info.equity,
            "margin": self.account_info.margin,
            "free_margin": self.account_info.margin_free,
            "profit": self.account_info.profit,
            "currency": self.account_info.currency
        }
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if not self.connected or not self.mt5:
            return None
        
        try:
            info = self.mt5.symbol_info(symbol)
            if not info:
                return None
            
            return {
                "symbol": info.name,
                "bid": info.bid,
                "ask": info.ask,
                "spread": info.ask - info.bid,
                "point": info.point,
                "digits": info.digits,
                "trade_mode": info.trade_mode,
                "volume_min": info.volume_min,
                "volume_max": info.volume_max,
                "volume_step": info.volume_step
            }
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None


class OrderFillingOptimizer:
    """
    Order Filling Mode Optimizer
    Selects optimal filling mode based on market conditions
    """
    
    def __init__(self, config):
        self.enabled = getattr(config.order_management, 'filling_optimization_enabled', True)
        self.default_mode = OrderFillingMode(getattr(config.order_management, 'default_filling_mode', 'ioc'))
        
        # Filling mode selection rules
        self.volatility_threshold = getattr(config.order_management, 'volatility_threshold', 0.002)
        self.spread_threshold = getattr(config.order_management, 'spread_threshold', 0.0003)
        self.volume_threshold = getattr(config.order_management, 'volume_threshold', 1.0)
        
    def select_filling_mode(self, 
                           symbol_info: Dict,
                           volume: float,
                           volatility: float,
                           urgency: str = "normal") -> OrderFillingMode:
        """
        Select optimal filling mode based on market conditions
        
        Args:
            symbol_info: Symbol information from MT5
            volume: Order volume
            volatility: Current market volatility (ATR)
            urgency: Order urgency (low, normal, high)
        
        Returns:
            Optimal OrderFillingMode
        """
        if not self.enabled:
            return self.default_mode
        
        # Get current spread
        current_spread = symbol_info.get('spread', 0.0)
        
        # High urgency orders use IOC for immediate execution
        if urgency == "high":
            return OrderFillingMode.IOC
        
        # Large volume orders use FOK to ensure complete execution
        if volume >= self.volume_threshold:
            return OrderFillingMode.FOK
        
        # High volatility markets use IOC to avoid slippage
        if volatility >= self.volatility_threshold:
            return OrderFillingMode.IOC
        
        # Wide spreads use RETURN to wait for better prices
        if current_spread >= self.spread_threshold:
            return OrderFillingMode.RETURN
        
        # Normal conditions use default mode
        return self.default_mode


class SlippageManager:
    """
    Slippage Management System
    Monitors and controls slippage during order execution
    """
    
    def __init__(self, config):
        self.enabled = getattr(config.order_management, 'slippage_control_enabled', True)
        self.max_slippage = getattr(config.order_management, 'max_slippage', 0.0005)
        self.slippage_tolerance = getattr(config.order_management, 'slippage_tolerance', 0.0002)
        
        # Slippage tracking
        self.slippage_history: List[Dict] = []
        self.max_history_size = 1000
        
    def calculate_slippage(self, 
                          requested_price: float, 
                          executed_price: float,
                          order_type: OrderType) -> float:
        """Calculate slippage for order execution"""
        if order_type in [OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP]:
            # For buy orders, slippage is positive if executed above requested
            slippage = executed_price - requested_price
        else:
            # For sell orders, slippage is positive if executed below requested
            slippage = requested_price - executed_price
        
        return slippage
    
    def is_slippage_acceptable(self, slippage: float) -> bool:
        """Check if slippage is within acceptable limits"""
        return abs(slippage) <= self.max_slippage
    
    def record_slippage(self, 
                        symbol: str,
                        order_type: OrderType,
                        requested_price: float,
                        executed_price: float,
                        volume: float):
        """Record slippage for analysis"""
        slippage = self.calculate_slippage(requested_price, executed_price, order_type)
        
        record = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": symbol,
            "order_type": order_type.value,
            "requested_price": requested_price,
            "executed_price": executed_price,
            "slippage": slippage,
            "volume": volume,
            "acceptable": self.is_slippage_acceptable(slippage)
        }
        
        self.slippage_history.append(record)
        
        # Maintain history size
        if len(self.slippage_history) > self.max_history_size:
            self.slippage_history.pop(0)
        
        # Log significant slippage
        if abs(slippage) > self.slippage_tolerance:
            logger.bind(evt="ORDER").warning("high_slippage_detected",
                                           symbol=symbol,
                                           slippage=slippage,
                                           volume=volume)
    
    def get_slippage_stats(self, symbol: str = None) -> Dict:
        """Get slippage statistics"""
        if not self.slippage_history:
            return {"total_orders": 0}
        
        # Filter by symbol if specified
        records = self.slippage_history
        if symbol:
            records = [r for r in records if r["symbol"] == symbol]
        
        if not records:
            return {"total_orders": 0}
        
        slippages = [r["slippage"] for r in records]
        acceptable_count = sum(1 for r in records if r["acceptable"])
        
        return {
            "total_orders": len(records),
            "acceptable_orders": acceptable_count,
            "unacceptable_orders": len(records) - acceptable_count,
            "avg_slippage": np.mean(slippages),
            "max_slippage": max(slippages),
            "min_slippage": min(slippages),
            "std_slippage": np.std(slippages)
        }


class OrderExecutor:
    """
    Main Order Executor
    Coordinates order execution with MT5
    """
    
    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config.order_management, 'enabled', True)
        
        # Initialize components
        self.mt5_connector = MT5Connector(config)
        self.filling_optimizer = OrderFillingOptimizer(config)
        self.slippage_manager = SlippageManager(config)
        
        # Order tracking
        self.active_orders: Dict[int, Dict] = {}
        self.order_history: List[Dict] = []
        
        self.logger = logger
    
    def place_order(self, 
                   order_request: OrderRequest,
                   volatility: float = 0.0,
                   urgency: str = "normal") -> OrderResult:
        """
        Place an order with optimal execution
        
        Args:
            order_request: Order details
            volatility: Current market volatility
            urgency: Order urgency level
        
        Returns:
            OrderResult with execution details
        """
        if not self.enabled:
            return OrderResult(success=False, error_description="Order management disabled")
        
        # Ensure MT5 connection
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                return OrderResult(success=False, error_description="MT5 connection failed")
        
        try:
            # Get symbol information
            symbol_info = self.mt5_connector.get_symbol_info(order_request.symbol)
            if not symbol_info:
                return OrderResult(success=False, error_description="Symbol not found")
            
            # Optimize filling mode
            optimal_filling_mode = self.filling_optimizer.select_filling_mode(
                symbol_info, order_request.volume, volatility, urgency
            )
            
            # Prepare MT5 order request
            mt5_request = self._prepare_mt5_request(order_request, optimal_filling_mode)
            
            # Execute order
            result = self._execute_mt5_order(mt5_request, order_request, symbol_info)
            
            # Record slippage if order was filled
            if result.success and result.volume > 0:
                self.slippage_manager.record_slippage(
                    order_request.symbol,
                    order_request.order_type,
                    order_request.price,
                    result.price,
                    result.volume
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return OrderResult(success=False, error_description=str(e))
    
    def _prepare_mt5_request(self, 
                            order_request: OrderRequest,
                            filling_mode: OrderFillingMode) -> Dict:
        """Prepare MT5 order request"""
        # Convert order type
        mt5_order_type = self._convert_order_type(order_request.order_type)
        
        # Convert filling mode
        mt5_filling = self._convert_filling_mode(filling_mode)
        
        return {
            "action": self.mt5_connector.mt5.TRADE_ACTION_DEAL,
            "symbol": order_request.symbol,
            "volume": order_request.volume,
            "type": mt5_order_type,
            "price": order_request.price,
            "sl": order_request.stop_loss,
            "tp": order_request.take_profit,
            "deviation": 20,  # Allow 2 pips deviation
            "magic": order_request.magic,
            "comment": order_request.comment,
            "type_filling": mt5_filling,
            "type_time": self.mt5_connector.mt5.ORDER_TIME_GTC
        }
    
    def _convert_order_type(self, order_type: OrderType) -> int:
        """Convert OrderType to MT5 constants"""
        if order_type == OrderType.BUY:
            return self.mt5_connector.mt5.ORDER_TYPE_BUY
        elif order_type == OrderType.SELL:
            return self.mt5_connector.mt5.ORDER_TYPE_SELL
        elif order_type == OrderType.BUY_LIMIT:
            return self.mt5_connector.mt5.ORDER_TYPE_BUY_LIMIT
        elif order_type == OrderType.SELL_LIMIT:
            return self.mt5_connector.mt5.ORDER_TYPE_SELL_LIMIT
        elif order_type == OrderType.BUY_STOP:
            return self.mt5_connector.mt5.ORDER_TYPE_BUY_STOP
        elif order_type == OrderType.SELL_STOP:
            return self.mt5_connector.mt5.ORDER_TYPE_SELL_STOP
        else:
            return self.mt5_connector.mt5.ORDER_TYPE_BUY
    
    def _convert_filling_mode(self, filling_mode: OrderFillingMode) -> int:
        """Convert OrderFillingMode to MT5 constants"""
        if filling_mode == OrderFillingMode.IOC:
            return self.mt5_connector.mt5.ORDER_FILLING_IOC
        elif filling_mode == OrderFillingMode.FOK:
            return self.mt5_connector.mt5.ORDER_FILLING_FOK
        elif filling_mode == OrderFillingMode.RETURN:
            return self.mt5_connector.mt5.ORDER_FILLING_RETURN
        else:
            return self.mt5_connector.mt5.ORDER_FILLING_IOC
    
    def _execute_mt5_order(self, 
                          mt5_request: Dict,
                          order_request: OrderRequest,
                          symbol_info: Dict) -> OrderResult:
        """Execute order using MT5"""
        try:
            # Send order
            result = self.mt5_connector.mt5.order_send(mt5_request)
            
            if result.retcode != self.mt5_connector.mt5.TRADE_RETCODE_DONE:
                return OrderResult(
                    success=False,
                    error_code=result.retcode,
                    error_description=f"MT5 error: {result.retcode}"
                )
            
            # Calculate spread
            spread = symbol_info.get('spread', 0.0)
            
            # Calculate slippage
            slippage = self.slippage_manager.calculate_slippage(
                order_request.price, result.price, order_request.order_type
            )
            
            # Create order result
            order_result = OrderResult(
                success=True,
                ticket=result.order,
                volume=result.volume,
                price=result.price,
                spread=spread,
                slippage=slippage,
                execution_time=datetime.now(timezone.utc)
            )
            
            # Log successful order
            self.logger.bind(evt="ORDER").info("order_executed",
                                            symbol=order_request.symbol,
                                            type=order_request.order_type.value,
                                            volume=result.volume,
                                            price=result.price,
                                            slippage=slippage)
            
            return order_result
            
        except Exception as e:
            logger.error(f"MT5 order execution error: {e}")
            return OrderResult(success=False, error_description=str(e))
    
    def modify_order(self, 
                    ticket: int,
                    new_sl: Optional[float] = None,
                    new_tp: Optional[float] = None) -> bool:
        """Modify existing order"""
        if not self.mt5_connector.connected:
            return False
        
        try:
            request = {
                "action": self.mt5_connector.mt5.TRADE_ACTION_SLTP,
                "symbol": "",
                "sl": new_sl,
                "tp": new_tp
            }
            
            result = self.mt5_connector.mt5.order_send(request)
            return result.retcode == self.mt5_connector.mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            logger.error(f"Order modification error: {e}")
            return False
    
    def cancel_order(self, ticket: int) -> bool:
        """Cancel pending order"""
        if not self.mt5_connector.connected:
            return False
        
        try:
            request = {
                "action": self.mt5_connector.mt5.TRADE_ACTION_REMOVE,
                "order": ticket
            }
            
            result = self.mt5_connector.mt5.order_send(request)
            return result.retcode == self.mt5_connector.mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False
    
    def get_order_status(self, ticket: int) -> Optional[Dict]:
        """Get current order status"""
        if not self.mt5_connector.connected:
            return None
        
        try:
            order = self.mt5_connector.mt5.order_get(ticket)
            if not order:
                return None
            
            return {
                "ticket": order.order,
                "symbol": order.symbol,
                "type": order.type,
                "volume": order.volume_initial,
                "volume_filled": order.volume_current,
                "price": order.price_open,
                "sl": order.sl,
                "tp": order.tp,
                "profit": order.profit,
                "status": self._get_order_status(order)
            }
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    def _get_order_status(self, order) -> OrderStatus:
        """Determine order status from MT5 order object"""
        if order.volume_current == 0:
            return OrderStatus.FILLED
        elif order.volume_current < order.volume_initial:
            return OrderStatus.PARTIALLY_FILLED
        else:
            return OrderStatus.PENDING
    
    def get_execution_summary(self) -> Dict:
        """Get order execution summary"""
        slippage_stats = self.slippage_manager.get_slippage_stats()
        
        return {
            "mt5_connected": self.mt5_connector.connected,
            "active_orders": len(self.active_orders),
            "total_orders": len(self.order_history),
            "slippage_stats": slippage_stats
        }
