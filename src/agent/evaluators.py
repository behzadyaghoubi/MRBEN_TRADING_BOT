"""
LLM evaluators for MR BEN AI Agent system.
Provides Supervisor and Risk Officer evaluation using structured outputs.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from .prompts import (
    RISK_OFFICER_JSON_SCHEMA,
    RISK_OFFICER_SYSTEM_PROMPT,
    SUPERVISOR_JSON_SCHEMA,
    SUPERVISOR_SYSTEM_PROMPT,
    format_risk_officer_prompt,
    format_supervisor_prompt,
)
from .schemas import (
    DecisionContext,
    RiskOfficerDecision,
    SupervisorDecision,
    ToolProposal,
    TradingMode,
)


class BaseEvaluator:
    """Base class for LLM evaluators."""

    def __init__(self, model_name: str = "gpt-5", temperature: float = 0.1):
        """
        Initialize the evaluator.

        Args:
            model_name: LLM model to use
            temperature: Model temperature for responses
        """
        self.model_name = model_name
        self.temperature = temperature
        self.logger = logging.getLogger(self.__class__.__name__)

        # Mock OpenAI client for now - replace with actual client
        self.client = self._create_mock_client()

    def _create_mock_client(self):
        """Create a mock OpenAI client for testing."""

        # This would be replaced with actual OpenAI client
        class MockOpenAIClient:
            def __init__(self):
                self.chat = self.ChatCompletion()

            class ChatCompletion:
                def create(self, **kwargs):
                    return self.MockResponse(kwargs)

                class MockResponse:
                    def __init__(self, kwargs):
                        self.kwargs = kwargs
                        self.choices = [self.MockChoice()]

                    class MockChoice:
                        def __init__(self):
                            self.message = self.MockMessage()

                        class MockMessage:
                            def __init__(self):
                                self.content = '{"mock": "response"}'

        return MockOpenAIClient()

    def _call_llm(
        self, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Call the LLM with structured output requirements.

        Args:
            system_prompt: System role prompt
            user_prompt: User request prompt
            json_schema: Expected JSON schema

        Returns:
            Parsed JSON response
        """
        try:
            # In real implementation, this would call OpenAI with structured outputs
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=4000,
            )

            # Parse response
            content = response.choices[0].message.content
            parsed_response = json.loads(content)

            # Validate against schema (basic validation)
            self._validate_response(parsed_response, json_schema)

            return parsed_response

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM evaluation failed: {e}")

    def _validate_response(self, response: dict[str, Any], schema: dict[str, Any]):
        """Basic validation of LLM response against schema."""
        # This is a simplified validation - in production, use proper JSON schema validation
        required_fields = schema.get("required", [])

        for field in required_fields:
            if field not in response:
                raise ValueError(f"Missing required field: {field}")

        # Validate enum values if specified
        for field_name, field_spec in schema.get("properties", {}).items():
            if field_name in response:
                if "enum" in field_spec:
                    if response[field_name] not in field_spec["enum"]:
                        raise ValueError(f"Invalid value for {field_name}: {response[field_name]}")

                # Validate numeric ranges
                if "minimum" in field_spec and isinstance(response[field_name], (int, float)):
                    if response[field_name] < field_spec["minimum"]:
                        raise ValueError(
                            f"Value for {field_name} below minimum: {response[field_name]}"
                        )

                if "maximum" in field_spec and isinstance(response[field_name], (int, float)):
                    if response[field_name] > field_spec["maximum"]:
                        raise ValueError(
                            f"Value for {field_name} above maximum: {response[field_name]}"
                        )


class SupervisorEvaluator(BaseEvaluator):
    """Supervisor LLM evaluator for initial decision analysis."""

    def __init__(self, model_name: str = "gpt-5", temperature: float = 0.1):
        """Initialize the supervisor evaluator."""
        super().__init__(model_name, temperature)
        self.logger = logging.getLogger("SupervisorEvaluator")

    def evaluate(
        self, proposal: ToolProposal, context: DecisionContext, risk_status: dict[str, Any]
    ) -> SupervisorDecision:
        """
        Evaluate a tool proposal and provide supervisor decision.

        Args:
            proposal: Tool execution proposal
            context: Decision context
            risk_status: Current risk status

        Returns:
            Supervisor decision
        """
        self.logger.info(f"Evaluating proposal for tool: {proposal.tool_name}")

        try:
            # Format the prompt with actual data
            user_prompt = format_supervisor_prompt(
                session_info=f"Session: {context.session_id}, Mode: {context.trading_mode}",
                market_conditions=context.market_conditions,
                risk_metrics=risk_status,
                recent_trades=context.recent_trades,
                tool_name=proposal.tool_name,
                input_data=proposal.input_data,
                reasoning=proposal.reasoning,
                risk_assessment=proposal.risk_assessment,
                expected_outcome=proposal.expected_outcome,
                urgency=proposal.urgency,
                confidence=proposal.confidence,
            )

            # Call LLM for evaluation
            llm_response = self._call_llm(
                SUPERVISOR_SYSTEM_PROMPT, user_prompt, SUPERVISOR_JSON_SCHEMA
            )

            # Create structured decision
            decision = self._create_supervisor_decision(llm_response, proposal, context)

            self.logger.info(f"Supervisor decision completed: {decision.recommendation}")
            return decision

        except Exception as e:
            self.logger.error(f"Supervisor evaluation failed: {e}")
            # Return a safe default decision
            return self._create_default_decision(proposal, context, str(e))

    def _create_supervisor_decision(
        self, llm_response: dict[str, Any], proposal: ToolProposal, context: DecisionContext
    ) -> SupervisorDecision:
        """Create a SupervisorDecision from LLM response."""
        # Generate unique decision ID
        decision_id = str(uuid.uuid4())

        # Create decision context
        decision_context = DecisionContext(
            timestamp=datetime.utcnow(),
            session_id=context.session_id,
            trading_mode=context.trading_mode,
            market_conditions=context.market_conditions,
            risk_metrics=context.risk_metrics,
            recent_trades=context.recent_trades,
        )

        # Create supervisor decision
        decision = SupervisorDecision(
            decision_id=decision_id,
            context=decision_context,
            proposal=proposal,
            supervisor_analysis=llm_response.get("supervisor_analysis", ""),
            recommendation=llm_response.get("recommendation", "request_more_info"),
            confidence=llm_response.get("confidence", 0.5),
            risk_level=llm_response.get("risk_level", "medium"),
            constraints=llm_response.get("constraints", []),
            timestamp=datetime.utcnow(),
        )

        return decision

    def _create_default_decision(
        self, proposal: ToolProposal, context: DecisionContext, error_reason: str
    ) -> SupervisorDecision:
        """Create a default decision when evaluation fails."""
        decision_id = str(uuid.uuid4())

        decision_context = DecisionContext(
            timestamp=datetime.utcnow(),
            session_id=context.session_id,
            trading_mode=context.trading_mode,
            market_conditions=context.market_conditions,
            risk_metrics=context.risk_metrics,
            recent_trades=context.recent_trades,
        )

        return SupervisorDecision(
            decision_id=decision_id,
            context=decision_context,
            proposal=proposal,
            supervisor_analysis=f"Evaluation failed: {error_reason}",
            recommendation="request_more_info",
            confidence=0.0,
            risk_level="high",
            constraints=["Requires manual review due to evaluation failure"],
            timestamp=datetime.utcnow(),
        )


class RiskOfficerEvaluator(BaseEvaluator):
    """Risk Officer LLM evaluator for final approval decisions."""

    def __init__(self, model_name: str = "gpt-5", temperature: float = 0.05):
        """Initialize the risk officer evaluator."""
        super().__init__(model_name, temperature)
        self.logger = logging.getLogger("RiskOfficerEvaluator")

    def evaluate(
        self, supervisor_decision: SupervisorDecision, risk_status: dict[str, Any]
    ) -> RiskOfficerDecision:
        """
        Evaluate supervisor decision and provide final risk officer decision.

        Args:
            supervisor_decision: Supervisor's decision
            risk_status: Current risk status

        Returns:
            Risk officer decision
        """
        self.logger.info(f"Evaluating supervisor decision: {supervisor_decision.decision_id}")

        try:
            # Format the prompt with actual data
            user_prompt = format_risk_officer_prompt(
                supervisor_decision=supervisor_decision.dict(),
                daily_loss_percent=risk_status.get("daily_loss_percent", 0.0),
                daily_loss_limit=risk_status.get("daily_loss_limit", 2.0),
                open_positions=risk_status.get("open_positions", 0),
                max_open_positions=risk_status.get("max_open_positions", 3),
                current_risk_level=risk_status.get("current_risk_level", "low"),
                market_volatility=risk_status.get("market_volatility", "medium"),
                trading_session=risk_status.get("trading_session", "active"),
                max_daily_loss=risk_status.get("max_daily_loss", 2.0),
                max_open_trades=risk_status.get("max_open_trades", 3),
                max_position_size_usd=risk_status.get("max_position_size_usd", 10000.0),
                cooldown_minutes=risk_status.get("cooldown_minutes", 30),
                emergency_threshold=risk_status.get("emergency_threshold", 5.0),
            )

            # Call LLM for evaluation
            llm_response = self._call_llm(
                RISK_OFFICER_SYSTEM_PROMPT, user_prompt, RISK_OFFICER_JSON_SCHEMA
            )

            # Create structured decision
            decision = self._create_risk_officer_decision(llm_response, supervisor_decision)

            self.logger.info(f"Risk officer decision completed: {decision.approval_status}")
            return decision

        except Exception as e:
            self.logger.error(f"Risk officer evaluation failed: {e}")
            # Return a safe default decision
            return self._create_default_decision(supervisor_decision, str(e))

    def _create_risk_officer_decision(
        self, llm_response: dict[str, Any], supervisor_decision: SupervisorDecision
    ) -> RiskOfficerDecision:
        """Create a RiskOfficerDecision from LLM response."""
        # Generate unique decision ID
        decision_id = str(uuid.uuid4())

        # Create risk officer decision
        decision = RiskOfficerDecision(
            decision_id=decision_id,
            supervisor_decision=supervisor_decision,
            risk_officer_analysis=llm_response.get("risk_officer_analysis", ""),
            approval_status=llm_response.get("approval_status", "pending"),
            approved_constraints=llm_response.get("approved_constraints", []),
            risk_mitigation=llm_response.get("risk_mitigation", []),
            final_approval=llm_response.get("final_approval", False),
            reasoning=llm_response.get("reasoning", ""),
            timestamp=datetime.utcnow(),
        )

        return decision

    def _create_default_decision(
        self, supervisor_decision: SupervisorDecision, error_reason: str
    ) -> RiskOfficerDecision:
        """Create a default decision when evaluation fails."""
        decision_id = str(uuid.uuid4())

        return RiskOfficerDecision(
            decision_id=decision_id,
            supervisor_decision=supervisor_decision,
            risk_officer_analysis=f"Evaluation failed: {error_reason}",
            approval_status="rejected",
            approved_constraints=[],
            risk_mitigation=["Requires manual review due to evaluation failure"],
            final_approval=False,
            reasoning="Automatic rejection due to evaluation failure",
            timestamp=datetime.utcnow(),
        )

    def evaluate_emergency_halt(
        self, risk_condition: str, current_exposure: dict[str, Any], recommended_action: str
    ) -> RiskOfficerDecision:
        """
        Evaluate emergency halt request.

        Args:
            risk_condition: Description of risk condition
            current_exposure: Current exposure details
            recommended_action: Recommended action

        Returns:
            Emergency halt decision
        """
        self.logger.warning(f"Emergency halt evaluation: {risk_condition}")

        # For emergency situations, we might want to bypass LLM and use rule-based logic
        # or use a specialized emergency prompt

        # Create emergency decision context
        emergency_context = DecisionContext(
            session_id="emergency",
            trading_mode=TradingMode.OBSERVE,  # Force observe mode
            market_conditions={"emergency": True},
            risk_metrics=current_exposure,
            recent_trades=[],
        )

        # Create emergency proposal
        emergency_proposal = ToolProposal(
            tool_name="halt_trading",
            input_data={"reason": risk_condition, "emergency": True},
            reasoning="Emergency risk condition detected",
            risk_assessment="Critical risk requiring immediate action",
            expected_outcome="Trading halted to prevent further losses",
            urgency="critical",
            confidence=1.0,
        )

        # Create emergency supervisor decision
        emergency_supervisor = SupervisorDecision(
            decision_id=str(uuid.uuid4()),
            context=emergency_context,
            proposal=emergency_proposal,
            supervisor_analysis="Emergency condition requires immediate halt",
            recommendation="approve",
            confidence=1.0,
            risk_level="critical",
            constraints=["Emergency halt with full position closure"],
            timestamp=datetime.utcnow(),
        )

        # Create emergency risk officer decision
        emergency_decision = RiskOfficerDecision(
            decision_id=str(uuid.uuid4()),
            supervisor_decision=emergency_supervisor,
            risk_officer_analysis="Emergency halt approved due to critical risk",
            approval_status="approved",
            approved_constraints=["Emergency halt with full position closure"],
            risk_mitigation=["Immediate halt", "Close all positions", "Cancel all orders"],
            final_approval=True,
            reasoning="Emergency halt approved to prevent catastrophic losses",
            timestamp=datetime.utcnow(),
        )

        return emergency_decision
