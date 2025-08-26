"""
Unit tests for adaptive confidence scorer.

Tests regime-aware confidence adjustments and threshold calculations.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategies.scorer import (
    AdaptiveDecision, AdaptiveConfig, AdaptiveScorer, 
    create_adaptive_scorer, quick_adapt_confidence, get_regime_penalty
)
from ai.regime import RegimeLabel, RegimeSnapshot


class TestAdaptiveConfig:
    """Test adaptive configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        config = AdaptiveConfig(
            base_conf=0.65,
            trend_bonus=0.12,
            chop_penalty=0.18,
            high_vol_penalty=0.12
        )
        
        assert config.base_conf == 0.65
        assert config.trend_bonus == 0.12
        assert config.chop_penalty == 0.18
        assert config.high_vol_penalty == 0.12
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptiveConfig()
        
        assert config.base_conf == 0.60
        assert config.trend_bonus == 0.10
        assert config.chop_penalty == 0.15
        assert config.high_vol_penalty == 0.10
        assert config.min_conf_floor == 0.50
        assert config.max_conf_cap == 0.90
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid base confidence
        with pytest.raises(ValueError):
            AdaptiveConfig(base_conf=1.5)
        
        # Invalid min/max bounds
        with pytest.raises(ValueError):
            AdaptiveConfig(min_conf_floor=0.8, max_conf_cap=0.6)
        
        # Valid bounds
        config = AdaptiveConfig(min_conf_floor=0.4, max_conf_cap=0.95)
        assert config.min_conf_floor == 0.4
        assert config.max_conf_cap == 0.95


class TestAdaptiveScorer:
    """Test adaptive confidence scoring functionality."""
    
    def setup_method(self):
        """Set up test scorer and data."""
        self.config = AdaptiveConfig(
            base_conf=0.60,
            trend_bonus=0.10,
            chop_penalty=0.15,
            high_vol_penalty=0.10,
            min_conf_floor=0.50,
            max_conf_cap=0.90
        )
        
        self.scorer = AdaptiveScorer(self.config)
        
        # Create test regime snapshots
        self.trend_regime = RegimeSnapshot(
            label=RegimeLabel.TREND,
            scores={"adx": 25.0, "rv": 0.008, "z": 1.2},
            session="london",
            ts=datetime.now(),
            confidence=0.85
        )
        
        self.range_regime = RegimeSnapshot(
            label=RegimeLabel.RANGE,
            scores={"adx": 15.0, "rv": 0.006, "z": 0.3},
            session="ny",
            ts=datetime.now(),
            confidence=0.75
        )
        
        self.high_vol_regime = RegimeSnapshot(
            label=RegimeLabel.HIGH_VOL,
            scores={"adx": 18.0, "rv": 0.020, "z": 0.8},
            session="london",
            ts=datetime.now(),
            confidence=0.80
        )
        
        self.illiquid_regime = RegimeSnapshot(
            label=RegimeLabel.ILLIQUID,
            scores={"adx": 12.0, "rv": 0.005, "z": 0.1, "spread_bp": 25.0},
            session="asia",
            ts=datetime.now(),
            confidence=0.70
        )
    
    def test_trend_regime_bonus(self):
        """Test confidence bonus for trend regime."""
        raw_conf = 0.70
        
        decision = self.scorer.adapt_confidence(raw_conf, self.trend_regime)
        
        assert decision.regime == RegimeLabel.TREND
        assert decision.adj_conf == 0.70  # 0.60 + 0.10
        assert decision.threshold == 0.70
        assert decision.allow_trade == True
        assert decision.trend_bonus == 0.10
        assert decision.chop_penalty == 0.0
        assert decision.vol_penalty == 0.0
    
    def test_range_regime_penalty(self):
        """Test confidence penalty for range regime."""
        raw_conf = 0.75
        
        decision = self.scorer.adapt_confidence(raw_conf, self.range_regime)
        
        assert decision.regime == RegimeLabel.RANGE
        assert decision.adj_conf == 0.45  # 0.60 - 0.15
        assert decision.threshold == 0.75  # Regime threshold applies
        assert decision.allow_trade == False  # raw_conf < threshold
        assert decision.trend_bonus == 0.0
        assert decision.chop_penalty == 0.15
        assert decision.vol_penalty == 0.0
    
    def test_high_vol_regime_penalty(self):
        """Test confidence penalty for high volatility regime."""
        raw_conf = 0.80
        
        decision = self.scorer.adapt_confidence(raw_conf, self.high_vol_regime)
        
        assert decision.regime == RegimeLabel.HIGH_VOL
        assert decision.adj_conf == 0.50  # 0.60 - 0.10
        assert decision.threshold == 0.80  # Regime threshold applies
        assert decision.allow_trade == False  # raw_conf < threshold
        assert decision.trend_bonus == 0.0
        assert decision.chop_penalty == 0.0
        assert decision.vol_penalty == 0.10
    
    def test_illiquid_regime_severe_penalty(self):
        """Test severe penalty for illiquid regime."""
        raw_conf = 0.85
        
        decision = self.scorer.adapt_confidence(raw_conf, self.illiquid_regime)
        
        assert decision.regime == RegimeLabel.ILLIQUID
        assert decision.adj_conf == 0.35  # 0.60 - 0.25 (severe penalty)
        assert decision.threshold == 0.85  # Regime threshold applies
        assert decision.allow_trade == False  # raw_conf < threshold
        assert decision.trend_bonus == 0.0
        assert decision.chop_penalty == 0.0
        assert decision.vol_penalty == 0.0
    
    def test_confidence_bounds(self):
        """Test confidence floor and cap enforcement."""
        # Test floor
        raw_conf = 0.30
        decision = self.scorer.adapt_confidence(raw_conf, self.range_regime)
        assert decision.adj_conf == 0.50  # Should hit floor
        
        # Test cap
        raw_conf = 0.95
        decision = self.scorer.adapt_confidence(raw_conf, self.trend_regime)
        assert decision.adj_conf == 0.90  # Should hit cap
    
    def test_session_filters(self):
        """Test session-based filtering."""
        # Non-optimal session
        non_optimal_regime = RegimeSnapshot(
            label=RegimeLabel.TREND,
            scores={"adx": 25.0, "rv": 0.008, "z": 1.2},
            session="asia",  # Non-optimal
            ts=datetime.now(),
            confidence=0.85
        )
        
        raw_conf = 0.70
        decision = self.scorer.adapt_confidence(raw_conf, non_optimal_regime)
        
        # Should get additional session penalty
        assert decision.adj_conf < 0.70
        assert "session_asia" in decision.session_filter
    
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
        
        scorer_high = AdaptiveScorer(high_threshold_config)
        
        # Even with trend bonus, should hit regime threshold
        raw_conf = 0.75
        decision = scorer_high.adapt_confidence(raw_conf, self.trend_regime)
        
        assert decision.threshold == 0.80  # Regime threshold
        assert decision.allow_trade == False  # 0.75 < 0.80
    
    def test_position_size_multipliers(self):
        """Test position size multipliers by regime."""
        assert self.scorer.get_position_size_multiplier(RegimeLabel.TREND) == 1.0
        assert self.scorer.get_position_size_multiplier(RegimeLabel.RANGE) == 0.5
        assert self.scorer.get_position_size_multiplier(RegimeLabel.HIGH_VOL) == 0.3
        assert self.scorer.get_position_size_multiplier(RegimeLabel.ILLIQUID) == 0.1
        assert self.scorer.get_position_size_multiplier(RegimeLabel.UNKNOWN) == 0.7
    
    def test_regime_summary(self):
        """Test regime summary generation."""
        summary = self.scorer.get_regime_summary()
        
        assert "base_confidence" in summary
        assert "regime_thresholds" in summary
        assert "position_limits" in summary
        assert "adjustments" in summary
        assert "bounds" in summary
        
        assert summary["base_confidence"] == 0.60
        assert summary["adjustments"]["trend_bonus"] == 0.10
        assert summary["adjustments"]["chop_penalty"] == 0.15
    
    def test_config_update(self):
        """Test configuration update functionality."""
        new_config = {
            "base_conf": 0.65,
            "trend_bonus": 0.15,
            "chop_penalty": 0.20
        }
        
        self.scorer.update_config(new_config)
        
        assert self.scorer.config.base_conf == 0.65
        assert self.scorer.config.trend_bonus == 0.15
        assert self.scorer.config.chop_penalty == 0.20
    
    def test_reset_functionality(self):
        """Test reset to default configuration."""
        # Change config
        self.scorer.update_config({"base_conf": 0.75})
        assert self.scorer.config.base_conf == 0.75
        
        # Reset
        self.scorer.reset()
        assert self.scorer.config.base_conf == 0.60  # Default value


class TestConvenienceFunctions:
    """Test convenience functions for quick scoring."""
    
    def test_quick_adapt_confidence(self):
        """Test quick confidence adaptation."""
        decision = quick_adapt_confidence(
            raw_conf=0.75,
            regime_label="trend",
            config_dict={"base_conf": 0.65, "trend_bonus": 0.12}
        )
        
        assert decision.raw_conf == 0.75
        assert decision.regime == RegimeLabel.TREND
        assert decision.adj_conf == 0.77  # 0.65 + 0.12
        assert decision.allow_trade == True
    
    def test_get_regime_penalty(self):
        """Test regime penalty lookup."""
        assert get_regime_penalty("trend") == 0.0
        assert get_regime_penalty("range") == 0.15
        assert get_regime_penalty("high_vol") == 0.10
        assert get_regime_penalty("illiquid") == 0.25
        assert get_regime_penalty("unknown") == 0.05
        assert get_regime_penalty("nonexistent") == 0.05  # Default


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_confidence(self):
        """Test handling of zero confidence."""
        scorer = AdaptiveScorer()
        regime = RegimeSnapshot(
            label=RegimeLabel.TREND,
            scores={},
            session="london",
            ts=datetime.now(),
            confidence=0.8
        )
        
        decision = scorer.adapt_confidence(0.0, regime)
        
        assert decision.raw_conf == 0.0
        assert decision.adj_conf == 0.70  # 0.60 + 0.10
        assert decision.allow_trade == False
    
    def test_maximum_confidence(self):
        """Test handling of maximum confidence."""
        scorer = AdaptiveScorer()
        regime = RegimeSnapshot(
            label=RegimeLabel.RANGE,
            scores={},
            session="london",
            ts=datetime.now(),
            confidence=0.8
        )
        
        decision = scorer.adapt_confidence(1.0, regime)
        
        assert decision.raw_conf == 1.0
        assert decision.adj_conf == 0.45  # 0.60 - 0.15
        assert decision.threshold == 0.75  # Regime threshold
        assert decision.allow_trade == True  # 1.0 > 0.75
    
    def test_extreme_regime_confidence(self):
        """Test handling of extreme regime confidence values."""
        # Very low regime confidence
        low_conf_regime = RegimeSnapshot(
            label=RegimeLabel.TREND,
            scores={},
            session="london",
            ts=datetime.now(),
            confidence=0.1
        )
        
        scorer = AdaptiveScorer()
        decision = scorer.adapt_confidence(0.8, low_conf_regime)
        
        # Should still apply trend bonus
        assert decision.adj_conf == 0.70  # 0.60 + 0.10
        assert decision.allow_trade == True
    
    def test_mixed_regime_conditions(self):
        """Test complex regime scenarios."""
        # Create a regime with mixed characteristics
        mixed_regime = RegimeSnapshot(
            label=RegimeLabel.HIGH_VOL,  # High volatility
            scores={"adx": 25.0, "rv": 0.025, "z": 1.5, "spread_bp": 8.0},
            session="asia",  # Non-optimal session
            ts=datetime.now(),
            confidence=0.8
        )
        
        scorer = AdaptiveScorer()
        decision = scorer.adapt_confidence(0.85, mixed_regime)
        
        # Should get multiple penalties
        assert decision.regime == RegimeLabel.HIGH_VOL
        assert decision.adj_conf < 0.60  # Multiple penalties
        assert decision.vol_penalty > 0
        assert "session_asia" in decision.session_filter


class TestIntegration:
    """Test integration with regime system."""
    
    def test_regime_snapshot_integration(self):
        """Test integration with RegimeSnapshot objects."""
        scorer = AdaptiveScorer()
        
        # Test with all regime types
        regimes = [
            (RegimeLabel.TREND, 0.70, True),
            (RegimeLabel.RANGE, 0.75, False),
            (RegimeLabel.HIGH_VOL, 0.80, False),
            (RegimeLabel.ILLIQUID, 0.85, False),
            (RegimeLabel.UNKNOWN, 0.70, True)
        ]
        
        for regime_label, raw_conf, expected_allow in regimes:
            regime = RegimeSnapshot(
                label=regime_label,
                scores={},
                session="london",
                ts=datetime.now(),
                confidence=0.8
            )
            
            decision = scorer.adapt_confidence(raw_conf, regime)
            
            assert decision.regime == regime_label
            assert decision.allow_trade == expected_allow
    
    def test_config_consistency(self):
        """Test that configuration is consistent across operations."""
        config = AdaptiveConfig(
            base_conf=0.55,
            trend_bonus=0.08,
            chop_penalty=0.12,
            high_vol_penalty=0.08
        )
        
        scorer = AdaptiveScorer(config)
        
        # Test multiple operations
        regime = RegimeSnapshot(
            label=RegimeLabel.TREND,
            scores={},
            session="london",
            ts=datetime.now(),
            confidence=0.8
        )
        
        decision1 = scorer.adapt_confidence(0.7, regime)
        decision2 = scorer.adapt_confidence(0.8, regime)
        
        # Both should use same config
        assert decision1.adj_conf == 0.63  # 0.55 + 0.08
        assert decision2.adj_conf == 0.63  # 0.55 + 0.08


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
