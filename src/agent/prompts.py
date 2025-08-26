"""
Structured prompts for MR BEN AI Agent system.
Defines prompts for Supervisor and Risk Officer roles with JSON schema outputs.
"""

from typing import Any

# ============================================================================
# SUPERVISOR PROMPT
# ============================================================================

SUPERVISOR_SYSTEM_PROMPT = """You are the MR BEN Trading System Supervisor, an AI agent responsible for analyzing trading proposals and making initial recommendations.

Your role is to:
1. Analyze market conditions and trading proposals
2. Assess risks and potential outcomes
3. Provide structured recommendations with constraints
4. Ensure all proposals align with trading system rules

You must always respond with valid JSON according to the specified schema.
Never provide explanations outside the JSON structure.
"""

SUPERVISOR_USER_PROMPT_TEMPLATE = """
Analyze the following trading proposal and provide your supervisor decision.

CONTEXT:
{session_info}

MARKET CONDITIONS:
{market_conditions}

MARKET REGIME:
Current Regime: {regime_label}
Regime Confidence: {regime_confidence}
Regime Features: {regime_features}
Session: {trading_session}

RISK METRICS:
{risk_metrics}

RECENT TRADES:
{recent_trades}

PROPOSAL:
Tool: {tool_name}
Input Data: {input_data}
Reasoning: {reasoning}
Risk Assessment: {risk_assessment}
Expected Outcome: {expected_outcome}
Urgency: {urgency}
Confidence: {confidence}

Based on this information, provide your supervisor decision following the exact JSON schema.
Consider:
- Market volatility and conditions
- Current market regime and its implications
- Regime-adjusted confidence thresholds
- Current risk exposure
- Trading system performance
- Risk-reward ratio
- Compliance with trading rules

IMPORTANT: If the current regime is RANGE or HIGH_VOL, apply stricter confidence thresholds.
For TREND regimes, you may apply confidence bonuses.
For ILLIQUID regimes, be extremely cautious.

Respond ONLY with valid JSON matching the SupervisorDecision schema.
"""

SUPERVISOR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "decision_id": {
            "type": "string",
            "description": "Unique decision identifier (UUID format)",
        },
        "context": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Decision timestamp in ISO format",
                },
                "user_id": {"type": "string", "description": "User making the decision"},
                "session_id": {"type": "string", "description": "Trading session identifier"},
                "trading_mode": {
                    "type": "string",
                    "enum": ["observe", "paper", "live"],
                    "description": "Current trading mode",
                },
                "market_conditions": {"type": "object", "description": "Current market conditions"},
                "risk_metrics": {"type": "object", "description": "Current risk metrics"},
                "recent_trades": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of recent trades",
                },
            },
            "required": [
                "timestamp",
                "session_id",
                "trading_mode",
                "market_conditions",
                "risk_metrics",
                "recent_trades",
            ],
        },
        "proposal": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "input_data": {"type": "object"},
                "reasoning": {"type": "string"},
                "risk_assessment": {"type": "string"},
                "expected_outcome": {"type": "string"},
                "urgency": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": [
                "tool_name",
                "input_data",
                "reasoning",
                "risk_assessment",
                "expected_outcome",
                "urgency",
                "confidence",
            ],
        },
        "supervisor_analysis": {
            "type": "string",
            "description": "Detailed analysis of the proposal",
        },
        "recommendation": {
            "type": "string",
            "enum": ["approve", "approve_with_constraints", "reject", "request_more_info"],
            "description": "Supervisor recommendation",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Supervisor confidence level (0.0 to 1.0)",
        },
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "description": "Assessed risk level",
        },
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Recommended constraints if approved",
        },
        "adj_conf": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Regime-adjusted confidence level",
        },
        "threshold": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Regime-aware threshold applied",
        },
        "allow_trade": {
            "type": "boolean",
            "description": "Whether trade is allowed by regime analysis",
        },
        "regime_notes": {"type": "string", "description": "Regime-related decision notes"},
        "timestamp": {"type": "string", "format": "date-time", "description": "Decision timestamp"},
    },
    "required": [
        "decision_id",
        "context",
        "proposal",
        "supervisor_analysis",
        "recommendation",
        "confidence",
        "risk_level",
        "constraints",
        "timestamp",
    ],
}


# ============================================================================
# RISK OFFICER PROMPT
# ============================================================================

RISK_OFFICER_SYSTEM_PROMPT = """You are the MR BEN Trading System Risk Officer, an AI agent responsible for final approval of trading decisions.

Your role is to:
1. Review supervisor decisions and risk assessments
2. Apply final risk controls and constraints
3. Make final approval/rejection decisions
4. Ensure compliance with all risk management rules

You must always respond with valid JSON according to the specified schema.
Never provide explanations outside the JSON structure.
"""

RISK_OFFICER_USER_PROMPT_TEMPLATE = """
Review the following supervisor decision and provide your final risk officer decision.

SUPERVISOR DECISION:
{supervisor_decision}

CURRENT RISK STATUS:
- Daily Loss: {daily_loss_percent}% (Limit: {daily_loss_limit}%)
- Open Positions: {open_positions} (Limit: {max_open_positions})
- Current Risk: {current_risk_level}
- Market Volatility: {market_volatility}
- Trading Session: {trading_session}

MARKET REGIME:
- Current Regime: {regime_label}
- Regime Confidence: {regime_confidence}
- Regime Impact: {regime_impact}

RISK GATE RULES:
- Maximum daily loss: {max_daily_loss}%
- Maximum open trades: {max_open_trades}
- Maximum position size: ${max_position_size_usd}
- Cooldown after loss: {cooldown_minutes} minutes
- Emergency halt threshold: {emergency_threshold}%

Based on this information, provide your final risk officer decision following the exact JSON schema.
Consider:
- Risk limit compliance
- Market conditions and current regime
- Regime-specific risk adjustments
- System performance
- Supervisor analysis quality
- Potential impact on portfolio

IMPORTANT: Consider the market regime when making your decision:
- RANGE/HIGH_VOL regimes require higher confidence thresholds
- ILLIQUID regimes may warrant trade rejection
- TREND regimes may allow more aggressive positions

Respond ONLY with valid JSON matching the RiskOfficerDecision schema.
"""

RISK_OFFICER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "decision_id": {
            "type": "string",
            "description": "Unique decision identifier (UUID format)",
        },
        "supervisor_decision": {
            "type": "object",
            "description": "Complete supervisor decision object",
        },
        "risk_officer_analysis": {
            "type": "string",
            "description": "Detailed risk officer analysis",
        },
        "approval_status": {
            "type": "string",
            "enum": ["pending", "approved", "rejected", "approved_with_constraints"],
            "description": "Final approval status",
        },
        "approved_constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Final approved constraints",
        },
        "risk_mitigation": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Risk mitigation measures",
        },
        "final_approval": {"type": "boolean", "description": "Final approval decision"},
        "reasoning": {"type": "string", "description": "Detailed reasoning for decision"},
        "timestamp": {"type": "string", "format": "date-time", "description": "Decision timestamp"},
    },
    "required": [
        "decision_id",
        "supervisor_decision",
        "risk_officer_analysis",
        "approval_status",
        "approved_constraints",
        "risk_mitigation",
        "final_approval",
        "reasoning",
        "timestamp",
    ],
}


# ============================================================================
# PROMPT UTILITIES
# ============================================================================


def format_supervisor_prompt(
    session_info: str,
    market_conditions: dict[str, Any],
    risk_metrics: dict[str, Any],
    recent_trades: list,
    tool_name: str,
    input_data: dict[str, Any],
    reasoning: str,
    risk_assessment: str,
    expected_outcome: str,
    urgency: str = "normal",
    confidence: float = 0.5,
    regime_label: str = "unknown",
    regime_confidence: float = 0.5,
    regime_features: str = "{}",
    trading_session: str = "unknown",
) -> str:
    """Format the supervisor prompt with actual data."""
    return SUPERVISOR_USER_PROMPT_TEMPLATE.format(
        session_info=session_info,
        market_conditions=str(market_conditions),
        risk_metrics=str(risk_metrics),
        recent_trades=str(recent_trades),
        tool_name=tool_name,
        input_data=str(input_data),
        reasoning=reasoning,
        risk_assessment=risk_assessment,
        expected_outcome=expected_outcome,
        urgency=urgency,
        confidence=confidence,
        regime_label=regime_label,
        regime_confidence=regime_confidence,
        regime_features=regime_features,
        trading_session=trading_session,
    )


def format_risk_officer_prompt(
    supervisor_decision: dict[str, Any],
    daily_loss_percent: float,
    daily_loss_limit: float,
    open_positions: int,
    max_open_positions: int,
    current_risk_level: str,
    market_volatility: str,
    trading_session: str,
    max_daily_loss: float,
    max_open_trades: int,
    max_position_size_usd: float,
    cooldown_minutes: int,
    emergency_threshold: float,
    regime_label: str = "unknown",
    regime_confidence: float = 0.5,
    regime_impact: str = "neutral",
) -> str:
    """Format the risk officer prompt with actual data."""
    return RISK_OFFICER_USER_PROMPT_TEMPLATE.format(
        supervisor_decision=str(supervisor_decision),
        daily_loss_percent=daily_loss_percent,
        daily_loss_limit=daily_loss_limit,
        open_positions=open_positions,
        max_open_positions=max_open_positions,
        current_risk_level=current_risk_level,
        market_volatility=market_volatility,
        trading_session=trading_session,
        max_daily_loss=max_daily_loss,
        max_open_trades=max_open_trades,
        max_position_size_usd=max_position_size_usd,
        cooldown_minutes=cooldown_minutes,
        emergency_threshold=emergency_threshold,
        regime_label=regime_label,
        regime_confidence=regime_confidence,
        regime_impact=regime_impact,
    )


# ============================================================================
# SPECIALIZED PROMPTS
# ============================================================================

EMERGENCY_HALT_PROMPT = """EMERGENCY HALT REQUESTED

The trading system has detected a critical risk condition that requires immediate attention.

RISK CONDITION: {risk_condition}
CURRENT EXPOSURE: {current_exposure}
RECOMMENDED ACTION: {recommended_action}

As Risk Officer, you must immediately:
1. Assess the emergency situation
2. Determine if halt is justified
3. Set appropriate constraints
4. Provide clear reasoning

Respond with your emergency decision using the RiskOfficerDecision schema.
"""

MARKET_REGIME_ANALYSIS_PROMPT = """MARKET REGIME ANALYSIS

Analyze the current market conditions and determine the appropriate trading regime.

MARKET DATA:
- Symbol: {symbol}
- Timeframe: {timeframe}
- Current Price: {current_price}
- ATR: {atr}
- RSI: {rsi}
- Volume: {volume}
- Trend Indicators: {trend_indicators}

HISTORICAL CONTEXT:
- Recent Highs/Lows: {recent_extremes}
- Support/Resistance: {support_resistance}
- Market Sentiment: {sentiment}

Based on this analysis, determine:
1. Market Regime: [trending, ranging, volatile, news-driven]
2. Risk Level: [low, medium, high, critical]
3. Recommended Constraints: [list of constraints]
4. Confidence Level: [0.0 to 1.0]

Respond with structured analysis following the specified schema.
"""

PERFORMANCE_REVIEW_PROMPT = """PERFORMANCE REVIEW REQUEST

Review the trading system performance and provide recommendations.

PERFORMANCE METRICS:
- Total Trades: {total_trades}
- Win Rate: {win_rate}%
- Profit Factor: {profit_factor}
- Max Drawdown: {max_drawdown}%
- Sharpe Ratio: {sharpe_ratio}
- Recent Performance: {recent_performance}

RISK METRICS:
- Current Risk: {current_risk}
- Risk Limits: {risk_limits}
- Breaches: {breaches}
- Cooldown Status: {cooldown_status}

Based on this review, provide:
1. Performance Assessment: [excellent, good, fair, poor]
2. Risk Assessment: [low, medium, high, critical]
3. Recommendations: [list of recommendations]
4. Constraints Adjustment: [list of constraint changes]
5. Confidence Level: [0.0 to 1.0]

Respond with structured review following the specified schema.
"""
