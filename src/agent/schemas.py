"""
Pydantic schemas for MR BEN AI Agent system.
Defines tool schemas, decision structures, and structured outputs.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime, timedelta
from decimal import Decimal
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ToolPermission(str, Enum):
    """Tool permission levels."""
    READ_ONLY = "read_only"
    WRITE_RESTRICTED = "write_restricted"
    WRITE_FULL = "write_full"


class DecisionStatus(str, Enum):
    """Decision approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPROVED_WITH_CONSTRAINTS = "approved_with_constraints"


class TradingMode(str, Enum):
    """Trading system modes."""
    OBSERVE = "observe"
    PAPER = "paper"
    LIVE = "live"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


# ============================================================================
# TOOL SCHEMAS
# ============================================================================

class ToolSchema(BaseModel):
    """Base schema for all tools."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    permission: ToolPermission = Field(..., description="Tool permission level")
    timeout_seconds: int = Field(default=30, description="Tool timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay_seconds: int = Field(default=5, description="Delay between retries")
    requires_approval: bool = Field(default=True, description="Whether tool requires approval")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key for the tool call")
    audit_tag: Optional[str] = Field(None, description="Audit tag for tracking")


class ReadOnlyToolSchema(ToolSchema):
    """Schema for read-only tools."""
    permission: ToolPermission = Field(default=ToolPermission.READ_ONLY, description="Read-only permission")
    requires_approval: bool = Field(default=False, description="Read-only tools don't require approval")


class WriteToolSchema(ToolSchema):
    """Schema for write tools."""
    permission: ToolPermission = Field(..., description="Write permission level")
    requires_approval: bool = Field(default=True, description="Write tools require approval")
    risk_level: str = Field(..., description="Risk level of the tool")
    max_impact_usd: Optional[Decimal] = Field(None, description="Maximum financial impact in USD")


# ============================================================================
# TOOL INPUT/OUTPUT SCHEMAS
# ============================================================================

class GetMarketSnapshotInput(BaseModel):
    """Input for get_market_snapshot tool."""
    symbol: str = Field(..., description="Trading symbol (e.g., XAUUSD.PRO)")
    timeframe_minutes: Optional[int] = Field(15, description="Timeframe in minutes")
    bars: Optional[int] = Field(100, description="Number of bars to retrieve")


class GetMarketSnapshotOutput(BaseModel):
    """Output for get_market_snapshot tool."""
    symbol: str
    current_price: Decimal
    bid: Decimal
    ask: Decimal
    spread_points: int
    last_update: datetime
    timeframe: str
    bars_count: int
    high_24h: Decimal
    low_24h: Decimal
    volume_24h: Optional[Decimal]
    change_24h: Decimal
    change_percent_24h: Decimal
    atr: Optional[Decimal]
    rsi: Optional[Decimal]
    ema_20: Optional[Decimal]
    ema_50: Optional[Decimal]
    trend_direction: Optional[str]
    volatility_level: Optional[str]


class GetRegimeSnapshotInput(BaseModel):
    """Input for get_regime_snapshot tool."""
    symbol: str = Field(..., description="Trading symbol (e.g., XAUUSD.PRO)")
    include_features: bool = Field(True, description="Include feature scores in output")
    include_history: bool = Field(False, description="Include recent regime history")


class GetRegimeSnapshotOutput(BaseModel):
    """Output for get_regime_snapshot tool."""
    symbol: str
    regime_label: str
    regime_confidence: float
    feature_scores: Dict[str, float]
    session: str
    timestamp: datetime
    prev_regime: Optional[str]
    dwell_bars: int
    regime_summary: Optional[Dict[str, Any]]


class GetPositionsInput(BaseModel):
    """Input for get_positions tool."""
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    magic_number: Optional[int] = Field(None, description="Filter by magic number")


class PositionInfo(BaseModel):
    """Position information."""
    ticket: int
    symbol: str
    type: str
    volume: Decimal
    open_price: Decimal
    current_price: Decimal
    profit: Decimal
    swap: Decimal
    open_time: datetime
    magic_number: int
    comment: Optional[str]
    sl: Optional[Decimal]
    tp: Optional[Decimal]


class GetPositionsOutput(BaseModel):
    """Output for get_positions tool."""
    total_positions: int
    total_profit: Decimal
    total_volume: Decimal
    positions: List[PositionInfo]
    risk_metrics: Dict[str, Any]


class GetOpenOrdersInput(BaseModel):
    """Input for get_open_orders tool."""
    symbol: Optional[str] = Field(None, description="Filter by symbol")


class OrderInfo(BaseModel):
    """Order information."""
    ticket: int
    symbol: str
    type: OrderType
    side: OrderSide
    volume: Decimal
    price: Decimal
    sl: Optional[Decimal]
    tp: Optional[Decimal]
    open_time: datetime
    magic_number: int
    comment: Optional[str]


class GetOpenOrdersOutput(BaseModel):
    """Output for get_open_orders tool."""
    total_orders: int
    orders: List[OrderInfo]


class GetConfigInput(BaseModel):
    """Input for get_config tool."""
    section: Optional[str] = Field(None, description="Configuration section to retrieve")


class GetConfigOutput(BaseModel):
    """Output for get_config tool."""
    trading_mode: TradingMode
    risk_limits: Dict[str, Any]
    trading_params: Dict[str, Any]
    ai_params: Dict[str, Any]
    session_info: Dict[str, Any]


class GetMetricsInput(BaseModel):
    """Input for get_metrics tool."""
    include_performance: bool = Field(True, description="Include performance metrics")
    include_system: bool = Field(True, description="Include system metrics")


class GetMetricsOutput(BaseModel):
    """Output for get_metrics tool."""
    performance: Dict[str, Any]
    system: Dict[str, Any]
    risk: Dict[str, Any]
    trading: Dict[str, Any]
    last_update: datetime


class QuickSimInput(BaseModel):
    """Input for quick_sim tool."""
    action: str = Field(..., description="Action to simulate (buy/sell)")
    symbol: str = Field(..., description="Trading symbol")
    volume: Decimal = Field(..., description="Volume to simulate")
    price: Optional[Decimal] = Field(None, description="Entry price (if not market)")
    sl_points: Optional[int] = Field(None, description="Stop loss in points")
    tp_points: Optional[int] = Field(None, description="Take profit in points")


class QuickSimOutput(BaseModel):
    """Output for quick_sim tool."""
    simulation_id: str
    action: str
    symbol: str
    volume: Decimal
    entry_price: Decimal
    sl_price: Optional[Decimal]
    tp_price: Optional[Decimal]
    risk_usd: Decimal
    potential_profit_usd: Decimal
    risk_reward_ratio: Decimal
    margin_required: Decimal
    max_loss_percent: Decimal
    warnings: List[str]
    recommendations: List[str]


# ============================================================================
# WRITE TOOL SCHEMAS
# ============================================================================

class PlaceOrderInput(BaseModel):
    """Input for place_order tool."""
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    volume: Decimal = Field(..., description="Order volume")
    price: Optional[Decimal] = Field(None, description="Order price (required for limit orders)")
    sl: Optional[Decimal] = Field(None, description="Stop loss price")
    tp: Optional[Decimal] = Field(None, description="Take profit price")
    magic_number: Optional[int] = Field(None, description="Magic number")
    comment: Optional[str] = Field(None, description="Order comment")
    expiration: Optional[datetime] = Field(None, description="Order expiration")


class PlaceOrderOutput(BaseModel):
    """Output for place_order tool."""
    success: bool
    order_ticket: Optional[int]
    message: str
    execution_price: Optional[Decimal]
    execution_time: Optional[datetime]
    warnings: List[str]


class CancelOrderInput(BaseModel):
    """Input for cancel_order tool."""
    order_ticket: int = Field(..., description="Order ticket to cancel")
    reason: Optional[str] = Field(None, description="Cancellation reason")


class CancelOrderOutput(BaseModel):
    """Output for cancel_order tool."""
    success: bool
    message: str
    cancellation_time: Optional[datetime]


class SetRiskLimitsInput(BaseModel):
    """Input for set_risk_limits tool."""
    max_daily_loss_percent: Optional[Decimal] = Field(None, description="Maximum daily loss percentage")
    max_open_trades: Optional[int] = Field(None, description="Maximum open trades")
    max_position_size_usd: Optional[Decimal] = Field(None, description="Maximum position size in USD")
    max_risk_per_trade_percent: Optional[Decimal] = Field(None, description="Maximum risk per trade percentage")
    cooldown_after_loss_minutes: Optional[int] = Field(None, description="Cooldown period after loss")
    halt_on_breach: bool = Field(True, description="Halt trading on risk limit breach")


class SetRiskLimitsOutput(BaseModel):
    """Output for set_risk_limits tool."""
    success: bool
    message: str
    previous_limits: Dict[str, Any]
    new_limits: Dict[str, Any]
    update_time: datetime


class HaltTradingInput(BaseModel):
    """Input for halt_trading tool."""
    reason: str = Field(..., description="Reason for halting trading")
    emergency: bool = Field(False, description="Emergency halt flag")
    close_all_positions: bool = Field(False, description="Close all open positions")
    cancel_all_orders: bool = Field(True, description="Cancel all open orders")


class HaltTradingOutput(BaseModel):
    """Output for halt_trading tool."""
    success: bool
    message: str
    halt_time: datetime
    positions_closed: int
    orders_cancelled: int
    trading_status: str


class ResumeTradingInput(BaseModel):
    """Input for resume_trading tool."""
    reason: str = Field(..., description="Reason for resuming trading")
    risk_review_completed: bool = Field(True, description="Risk review completed flag")
    new_risk_limits: Optional[Dict[str, Any]] = Field(None, description="New risk limits")


class ResumeTradingOutput(BaseModel):
    """Output for resume_trading tool."""
    success: bool
    message: str
    resume_time: datetime
    trading_status: str
    risk_limits: Dict[str, Any]


# ============================================================================
# DECISION SCHEMAS
# ============================================================================

class DecisionContext(BaseModel):
    """Context for a decision."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = Field(None, description="User making the decision")
    session_id: str = Field(..., description="Trading session ID")
    trading_mode: TradingMode = Field(..., description="Current trading mode")
    market_conditions: Dict[str, Any] = Field(default_factory=dict)
    risk_metrics: Dict[str, Any] = Field(default_factory=dict)
    recent_trades: List[Dict[str, Any]] = Field(default_factory=dict)
    
    # Regime-related fields
    regime_label: Optional[str] = Field(None, description="Current market regime label")
    regime_scores: Optional[Dict[str, float]] = Field(None, description="Regime feature scores")
    regime_confidence: Optional[float] = Field(None, description="Regime classification confidence")


class ToolProposal(BaseModel):
    """Proposal for tool execution."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    input_data: Dict[str, Any] = Field(..., description="Tool input data")
    reasoning: str = Field(..., description="Reasoning for the proposal")
    risk_assessment: str = Field(..., description="Risk assessment")
    expected_outcome: str = Field(..., description="Expected outcome")
    urgency: str = Field(default="normal", description="Urgency level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level")


class SupervisorDecision(BaseModel):
    """Supervisor decision output."""
    decision_id: str = Field(..., description="Unique decision ID")
    context: DecisionContext = Field(..., description="Decision context")
    proposal: ToolProposal = Field(..., description="Tool proposal")
    supervisor_analysis: str = Field(..., description="Supervisor analysis")
    recommendation: str = Field(..., description="Supervisor recommendation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Supervisor confidence")
    risk_level: str = Field(..., description="Assessed risk level")
    constraints: List[str] = Field(default_factory=list, description="Recommended constraints")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Regime-aware decision fields
    adj_conf: Optional[float] = Field(None, description="Regime-adjusted confidence")
    threshold: Optional[float] = Field(None, description="Regime-aware threshold")
    allow_trade: Optional[bool] = Field(None, description="Whether trade is allowed by regime")
    regime_notes: Optional[str] = Field(None, description="Regime-related decision notes")


class RiskOfficerDecision(BaseModel):
    """Risk Officer decision output."""
    decision_id: str = Field(..., description="Unique decision ID")
    supervisor_decision: SupervisorDecision = Field(..., description="Supervisor decision")
    risk_officer_analysis: str = Field(..., description="Risk officer analysis")
    approval_status: DecisionStatus = Field(..., description="Approval status")
    approved_constraints: List[str] = Field(default_factory=list, description="Approved constraints")
    risk_mitigation: List[str] = Field(default_factory=list, description="Risk mitigation measures")
    final_approval: bool = Field(..., description="Final approval decision")
    reasoning: str = Field(..., description="Approval/rejection reasoning")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DecisionOutcome(BaseModel):
    """Final decision outcome."""
    decision_id: str = Field(..., description="Decision ID")
    supervisor_decision: SupervisorDecision = Field(..., description="Supervisor decision")
    risk_officer_decision: RiskOfficerDecision = Field(..., description="Risk officer decision")
    execution_result: Optional[Dict[str, Any]] = Field(None, description="Tool execution result")
    execution_time: Optional[datetime] = Field(None, description="Execution time")
    outcome: str = Field(..., description="Outcome description")
    success: bool = Field(..., description="Whether execution was successful")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Regime outcome fields
    regime_label: Optional[str] = Field(None, description="Market regime at decision time")
    regime_impact: Optional[str] = Field(None, description="Impact of regime on decision")
    confidence_adjustment: Optional[float] = Field(None, description="Confidence adjustment applied")


# ============================================================================
# TOOL REGISTRY
# ============================================================================

class ToolRegistry(BaseModel):
    """Registry of available tools."""
    tools: Dict[str, ToolSchema] = Field(default_factory=dict)
    
    def register_tool(self, tool: ToolSchema):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[ToolSchema]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_tools_by_permission(self, permission: ToolPermission) -> List[ToolSchema]:
        """Get tools by permission level."""
        return [tool for tool in self.tools.values() if tool.permission == permission]
    
    def get_available_tools(self, trading_mode: TradingMode) -> List[ToolSchema]:
        """Get available tools for a trading mode."""
        if trading_mode == TradingMode.OBSERVE:
            return self.get_tools_by_permission(ToolPermission.READ_ONLY)
        elif trading_mode == TradingMode.PAPER:
            return [tool for tool in self.tools.values() 
                   if tool.permission in [ToolPermission.READ_ONLY, ToolPermission.WRITE_RESTRICTED]]
        else:  # LIVE
            return self.tools.values()


# ============================================================================
# AGENT STATE SCHEMAS
# ============================================================================

class AgentState(BaseModel):
    """Current state of the AI agent."""
    agent_id: str = Field(..., description="Unique agent ID")
    trading_mode: TradingMode = Field(..., description="Current trading mode")
    is_active: bool = Field(..., description="Whether agent is active")
    current_session: Optional[str] = Field(None, description="Current session ID")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    risk_gate_status: str = Field(..., description="Risk gate status")
    permissions: List[str] = Field(default_factory=list, description="Current permissions")
    constraints: List[str] = Field(default_factory=list, description="Active constraints")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


class AgentConfig(BaseModel):
    """Configuration for the AI agent."""
    model_name: str = Field(default="gpt-5", description="LLM model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=4000, description="Maximum tokens per response")
    structured_output: bool = Field(default=True, description="Use structured outputs")
    risk_gate_enabled: bool = Field(default=True, description="Enable risk gate")
    approval_required: bool = Field(default=True, description="Require approval for write operations")
    max_concurrent_decisions: int = Field(default=5, description="Maximum concurrent decisions")
    decision_timeout_seconds: int = Field(default=300, description="Decision timeout")
    audit_logging: bool = Field(default=True, description="Enable audit logging")
    performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
