"""
Integration tests for AI Agent regime integration.

Tests end-to-end regime-aware decision making in the Supervisor/Risk Officer pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai.regime import RegimeLabel, RegimeSnapshot, RegimeClassifier, RegimeConfig
from strategies.scorer import AdaptiveScorer, AdaptiveConfig
from agent.schemas import (
    DecisionContext, ToolProposal, SupervisorDecision, 
    RiskOfficerDecision, DecisionOutcome
)
from agent.prompts import format_supervisor_prompt, format_risk_officer_prompt


class TestAgentRegimeIntegration:
    """Test integration between AI Agent and regime detection."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create regime classifier
        self.regime_config = RegimeConfig(
            windows={"vol_atr": 14, "adx": 14, "return_ms": 20, "rv": 60},
            thresholds={
                "trend_adx": 20.0,
                "vol_high": 0.012,
                "spread_wide_bp": 15.0,
                "chop_z": 1.0
            },
            smoothing={"method": "ema", "alpha": 0.2, "min_dwell_bars": 5}
        )
        
        self.regime_classifier = RegimeClassifier(self.regime_config)
        
        # Create adaptive scorer
        self.scorer_config = AdaptiveConfig(
            base_conf=0.60,
            trend_bonus=0.10,
            chop_penalty=0.15,
            high_vol_penalty=0.10,
            regime_thresholds={
                "trend": 0.65,
                "range": 0.75,
                "high_vol": 0.80,
                "illiquid": 0.85,
                "unknown": 0.70
            }
        )
        
        self.adaptive_scorer = AdaptiveScorer(self.scorer_config)
        
        # Create test market data
        self.market_data = self._create_test_market_data()
        
        # Create test regime snapshots
        self.trend_regime = RegimeSnapshot(
            label=RegimeLabel.TREND,
            scores={"adx": 25.0, "rv": 0.008, "z": 1.2, "spread_bp": 8.0},
            session="london",
            ts=datetime.now(),
            confidence=0.85
        )
        
        self.range_regime = RegimeSnapshot(
            label=RegimeLabel.RANGE,
            scores={"adx": 15.0, "rv": 0.006, "z": 0.3, "spread_bp": 6.0},
            session="ny",
            ts=datetime.now(),
            confidence=0.75
        )
        
        self.high_vol_regime = RegimeSnapshot(
            label=RegimeLabel.HIGH_VOL,
            scores={"adx": 18.0, "rv": 0.025, "z": 0.8, "spread_bp": 10.0},
            session="london",
            ts=datetime.now(),
            confidence=0.80
        )
    
    def _create_test_market_data(self):
        """Create realistic test market data."""
        np.random.seed(42)
        n_bars = 100
        
        # Generate price data
        base_price = 2000.0
        trend = np.linspace(0, 0.03, n_bars)  # 3% trend
        volatility = 0.015  # 1.5% daily volatility
        returns = np.random.normal(trend, volatility, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.004, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.004, n_bars))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
        
        # Ensure high >= low
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        
        # Add timestamps
        start_time = datetime.now() - timedelta(days=n_bars)
        data.index = pd.date_range(start=start_time, periods=n_bars, freq='H')
        
        return data
    
    def test_regime_detection_integration(self):
        """Test that regime detection works with real market data."""
        # Test with trend data
        trend_data = self._create_trend_data(100)
        trend_snapshot = self.regime_classifier.infer_regime(trend_data)
        
        assert trend_snapshot.label in [RegimeLabel.TREND, RegimeLabel.UNKNOWN]
        assert trend_snapshot.confidence > 0.5
        assert trend_snapshot.scores['adx'] > 15
        
        # Test with range data
        range_data = self._create_range_data(100)
        range_snapshot = self.regime_classifier.infer_regime(range_data)
        
        assert range_snapshot.label in [RegimeLabel.RANGE, RegimeLabel.UNKNOWN]
        assert range_snapshot.confidence > 0.4
        assert range_snapshot.scores['adx'] < 25
    
    def _create_trend_data(self, n_bars):
        """Create trending market data."""
        base_price = 2000.0
        trend = np.linspace(0, 0.08, n_bars)  # 8% trend
        noise = np.random.normal(0, 0.003, n_bars)
        prices = base_price * np.exp(trend + noise)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
        
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        
        start_time = datetime.now() - timedelta(days=n_bars)
        data.index = pd.date_range(start=start_time, periods=n_bars, freq='H')
        
        return data
    
    def _create_range_data(self, n_bars):
        """Create ranging market data."""
        base_price = 2000.0
        cycle = np.sin(np.linspace(0, 6*np.pi, n_bars)) * 0.015
        prices = base_price * (1 + cycle)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0005, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_bars))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
        
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        
        start_time = datetime.now() - timedelta(days=n_bars)
        data.index = pd.date_range(start=start_time, periods=n_bars, freq='H')
        
        return data
    
    def test_adaptive_confidence_integration(self):
        """Test that adaptive confidence works with regime snapshots."""
        # Test trend regime bonus
        decision = self.adaptive_scorer.adapt_confidence(0.70, self.trend_regime)
        
        assert decision.regime == RegimeLabel.TREND
        assert decision.adj_conf == 0.70  # 0.60 + 0.10
        assert decision.allow_trade == True
        
        # Test range regime penalty
        decision = self.adaptive_scorer.adapt_confidence(0.75, self.range_regime)
        
        assert decision.regime == RegimeLabel.RANGE
        assert decision.adj_conf == 0.45  # 0.60 - 0.15
        assert decision.threshold == 0.75  # Regime threshold
        assert decision.allow_trade == False  # 0.75 < 0.75
    
    def test_supervisor_prompt_regime_integration(self):
        """Test that supervisor prompts include regime information."""
        # Test with trend regime
        prompt = format_supervisor_prompt(
            session_info="Test session",
            market_conditions={"volatility": "medium"},
            risk_metrics={"current_risk": "low"},
            recent_trades=[],
            tool_name="place_order",
            input_data={"symbol": "XAUUSD.PRO", "volume": 0.1},
            reasoning="Strong trend signal",
            risk_assessment="Low risk",
            expected_outcome="Profitable trade",
            urgency="normal",
            confidence=0.75,
            regime_label="trend",
            regime_confidence=0.85,
            regime_features="{'adx': 25.0, 'rv': 0.008}",
            trading_session="london"
        )
        
        assert "Current Regime: trend" in prompt
        assert "Regime Confidence: 0.85" in prompt
        assert "Regime Features: {'adx': 25.0, 'rv': 0.008}" in prompt
        assert "Session: london" in prompt
        assert "regime-adjusted confidence thresholds" in prompt
    
    def test_risk_officer_prompt_regime_integration(self):
        """Test that risk officer prompts include regime information."""
        # Test with range regime
        prompt = format_risk_officer_prompt(
            supervisor_decision={"confidence": 0.75},
            daily_loss_percent=1.5,
            daily_loss_limit=2.0,
            open_positions=2,
            max_open_positions=3,
            current_risk_level="medium",
            market_volatility="medium",
            trading_session="ny",
            max_daily_loss=2.0,
            max_open_trades=3,
            max_position_size_usd=10000.0,
            cooldown_minutes=30,
            emergency_threshold=5.0,
            regime_label="range",
            regime_confidence=0.75,
            regime_impact="restrictive"
        )
        
        assert "Current Regime: range" in prompt
        assert "Regime Confidence: 0.75" in prompt
        assert "Regime Impact: restrictive" in prompt
        assert "RANGE/HIGH_VOL regimes require higher confidence thresholds" in prompt
    
    def test_decision_context_regime_integration(self):
        """Test that decision context includes regime information."""
        context = DecisionContext(
            session_id="test_session_123",
            trading_mode="paper",
            regime_label="trend",
            regime_scores={"adx": 25.0, "rv": 0.008, "z": 1.2},
            regime_confidence=0.85
        )
        
        assert context.regime_label == "trend"
        assert context.regime_scores["adx"] == 25.0
        assert context.regime_confidence == 0.85
    
    def test_supervisor_decision_regime_integration(self):
        """Test that supervisor decisions include regime-aware fields."""
        context = DecisionContext(
            session_id="test_session_123",
            trading_mode="paper",
            regime_label="range",
            regime_scores={"adx": 15.0, "rv": 0.006},
            regime_confidence=0.75
        )
        
        proposal = ToolProposal(
            tool_name="place_order",
            input_data={"symbol": "XAUUSD.PRO", "volume": 0.1},
            reasoning="Range breakout signal",
            risk_assessment="Medium risk",
            expected_outcome="Range breakout trade",
            confidence=0.75
        )
        
        decision = SupervisorDecision(
            decision_id="dec_123",
            context=context,
            proposal=proposal,
            supervisor_analysis="Range market with breakout potential",
            recommendation="approve_with_constraints",
            confidence=0.75,
            risk_level="medium",
            constraints=["Reduce position size", "Tight stop loss"],
            adj_conf=0.45,  # Regime-adjusted confidence
            threshold=0.75,  # Regime threshold
            allow_trade=False,  # Blocked by regime
            regime_notes="Range regime requires 0.75 confidence, adjusted to 0.45"
        )
        
        assert decision.adj_conf == 0.45
        assert decision.threshold == 0.75
        assert decision.allow_trade == False
        assert "Range regime requires 0.75 confidence" in decision.regime_notes
    
    def test_decision_outcome_regime_integration(self):
        """Test that decision outcomes include regime information."""
        # Create supervisor decision
        context = DecisionContext(
            session_id="test_session_123",
            trading_mode="paper",
            regime_label="trend",
            regime_scores={"adx": 25.0, "rv": 0.008},
            regime_confidence=0.85
        )
        
        proposal = ToolProposal(
            tool_name="place_order",
            input_data={"symbol": "XAUUSD.PRO", "volume": 0.1},
            reasoning="Strong trend signal",
            risk_assessment="Low risk",
            expected_outcome="Trend following trade",
            confidence=0.80
        )
        
        supervisor_decision = SupervisorDecision(
            decision_id="dec_123",
            context=context,
            proposal=proposal,
            supervisor_analysis="Strong uptrend with good risk/reward",
            recommendation="approve",
            confidence=0.80,
            risk_level="low",
            constraints=[],
            adj_conf=0.70,  # 0.60 + 0.10 trend bonus
            threshold=0.70,
            allow_trade=True,
            regime_notes="Trend regime allows trade with confidence bonus"
        )
        
        # Create risk officer decision
        risk_officer_decision = RiskOfficerDecision(
            decision_id="dec_123",
            supervisor_decision=supervisor_decision,
            risk_officer_analysis="Approved by supervisor, risk acceptable",
            approval_status="approved",
            approved_constraints=[],
            risk_mitigation=["Standard position sizing"],
            final_approval=True,
            reasoning="Trend regime with good signal quality"
        )
        
        # Create decision outcome
        outcome = DecisionOutcome(
            decision_id="dec_123",
            supervisor_decision=supervisor_decision,
            risk_officer_decision=risk_officer_decision,
            execution_result={"success": True, "order_ticket": 12345},
            execution_time=datetime.now(),
            outcome="Order placed successfully",
            success=True,
            errors=[],
            regime_label="trend",
            regime_impact="positive",
            confidence_adjustment=0.10
        )
        
        assert outcome.regime_label == "trend"
        assert outcome.regime_impact == "positive"
        assert outcome.confidence_adjustment == 0.10
    
    def test_end_to_end_regime_decision_flow(self):
        """Test complete regime-aware decision flow."""
        # 1. Detect regime
        regime_snapshot = self.regime_classifier.infer_regime(self.market_data)
        
        # 2. Adapt confidence based on regime
        raw_confidence = 0.75
        adaptive_decision = self.adaptive_scorer.adapt_confidence(
            raw_confidence, regime_snapshot
        )
        
        # 3. Create decision context with regime
        context = DecisionContext(
            session_id="test_session_123",
            trading_mode="paper",
            regime_label=regime_snapshot.label.value,
            regime_scores=regime_snapshot.scores,
            regime_confidence=regime_snapshot.confidence
        )
        
        # 4. Create tool proposal
        proposal = ToolProposal(
            tool_name="place_order",
            input_data={"symbol": "XAUUSD.PRO", "volume": 0.1},
            reasoning="Market analysis suggests trade opportunity",
            risk_assessment="Medium risk",
            expected_outcome="Profitable trade",
            confidence=raw_confidence
        )
        
        # 5. Create supervisor decision
        supervisor_decision = SupervisorDecision(
            decision_id="dec_123",
            context=context,
            proposal=proposal,
            supervisor_analysis="Regime-aware analysis completed",
            recommendation="approve" if adaptive_decision.allow_trade else "reject",
            confidence=raw_confidence,
            risk_level="medium",
            constraints=[],
            adj_conf=adaptive_decision.adj_conf,
            threshold=adaptive_decision.threshold,
            allow_trade=adaptive_decision.allow_trade,
            regime_notes=adaptive_decision.notes
        )
        
        # Verify integration
        assert supervisor_decision.regime_label == regime_snapshot.label.value
        assert supervisor_decision.adj_conf == adaptive_decision.adj_conf
        assert supervisor_decision.threshold == adaptive_decision.threshold
        assert supervisor_decision.allow_trade == adaptive_decision.allow_trade
        assert len(supervisor_decision.regime_notes) > 0
    
    def test_regime_blocking_mechanism(self):
        """Test that regime analysis can block trades."""
        # Test with range regime (should block trades)
        range_decision = self.adaptive_scorer.adapt_confidence(0.75, self.range_regime)
        
        assert range_decision.allow_trade == False
        assert range_decision.threshold == 0.75
        assert range_decision.adj_conf == 0.45
        
        # Test with trend regime (should allow trades)
        trend_decision = self.adaptive_scorer.adapt_confidence(0.75, self.trend_regime)
        
        assert trend_decision.allow_trade == True
        assert trend_decision.threshold == 0.70
        assert trend_decision.adj_conf == 0.70
    
    def test_regime_confidence_adjustments(self):
        """Test that regime confidence adjustments are applied correctly."""
        # Test trend bonus
        trend_decision = self.adaptive_scorer.adapt_confidence(0.65, self.trend_regime)
        assert trend_decision.adj_conf == 0.70  # 0.60 + 0.10
        
        # Test range penalty
        range_decision = self.adaptive_scorer.adapt_confidence(0.65, self.range_regime)
        assert range_decision.adj_conf == 0.45  # 0.60 - 0.15
        
        # Test high volatility penalty
        vol_decision = self.adaptive_scorer.adapt_confidence(0.65, self.high_vol_regime)
        assert vol_decision.adj_conf == 0.50  # 0.60 - 0.10
    
    def test_regime_threshold_override(self):
        """Test that regime thresholds override adjusted confidence."""
        # Create config with high regime thresholds
        high_threshold_config = AdaptiveConfig(
            regime_thresholds={
                "trend": 0.80,
                "range": 0.85,
                "high_vol": 0.90,
                "illiquid": 0.95,
                "unknown": 0.75
            }
        )
        
        high_threshold_scorer = AdaptiveScorer(high_threshold_config)
        
        # Even with trend bonus, should hit regime threshold
        decision = high_threshold_scorer.adapt_confidence(0.75, self.trend_regime)
        
        assert decision.threshold == 0.80  # Regime threshold
        assert decision.allow_trade == False  # 0.75 < 0.80
    
    def test_regime_session_filtering(self):
        """Test session-based filtering in regime analysis."""
        # Test non-optimal session
        asia_trend_regime = RegimeSnapshot(
            label=RegimeLabel.TREND,
            scores={"adx": 25.0, "rv": 0.008, "z": 1.2},
            session="asia",  # Non-optimal
            ts=datetime.now(),
            confidence=0.85
        )
        
        decision = self.adaptive_scorer.adapt_confidence(0.70, asia_trend_regime)
        
        # Should get additional session penalty
        assert decision.adj_conf < 0.70
        assert "session_asia" in decision.session_filter
    
    def test_regime_persistence_in_decisions(self):
        """Test that regime information persists through decision chain."""
        # Create regime snapshot
        regime_snapshot = self.regime_classifier.infer_regime(self.market_data)
        
        # Create decision context
        context = DecisionContext(
            session_id="test_session_123",
            trading_mode="paper",
            regime_label=regime_snapshot.label.value,
            regime_scores=regime_snapshot.scores,
            regime_confidence=regime_snapshot.confidence
        )
        
        # Verify regime information is preserved
        assert context.regime_label == regime_snapshot.label.value
        assert context.regime_scores == regime_snapshot.scores
        assert context.regime_confidence == regime_snapshot.confidence
        
        # Create proposal and decision
        proposal = ToolProposal(
            tool_name="place_order",
            input_data={"symbol": "XAUUSD.PRO", "volume": 0.1},
            reasoning="Test proposal",
            risk_assessment="Low risk",
            expected_outcome="Test outcome",
            confidence=0.75
        )
        
        decision = SupervisorDecision(
            decision_id="dec_123",
            context=context,
            proposal=proposal,
            supervisor_analysis="Test analysis",
            recommendation="approve",
            confidence=0.75,
            risk_level="low",
            constraints=[],
            adj_conf=0.70,
            threshold=0.70,
            allow_trade=True,
            regime_notes="Test regime notes"
        )
        
        # Verify regime information flows through
        assert decision.context.regime_label == regime_snapshot.label.value
        assert decision.regime_notes is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
