"""
Simple tests for agent regime integration.
This is a placeholder that tests basic functionality.
"""

from src.ai.regime import RegimeClassifier, RegimeConfig, RegimeLabel, RegimeSnapshot


class TestAgentRegimeIntegration:
    """Test basic integration between AI Agent and regime detection."""

    def test_basic_regime_classification(self):
        """Test that basic regime classification works."""
        classifier = RegimeClassifier()

        # Test low volatility
        snapshot = classifier.classify_by_volatility(0.005)
        assert snapshot.label == RegimeLabel.LOW

        # Test high volatility
        snapshot = classifier.classify_by_volatility(0.018)
        assert snapshot.label == RegimeLabel.HIGH

    def test_regime_config_integration(self):
        """Test that regime config can be customized."""
        config = RegimeConfig(thr_extreme=0.025, thr_high=0.020, thr_medium=0.015)

        classifier = RegimeClassifier(config)
        snapshot = classifier.classify_by_volatility(0.018)
        assert snapshot.label == RegimeLabel.MEDIUM  # 0.018 is > 0.015 but not > 0.020

    def test_regime_snapshot_attributes(self):
        """Test that regime snapshots have expected attributes."""
        snapshot = RegimeSnapshot(volatility=0.015, label=RegimeLabel.HIGH)

        assert hasattr(snapshot, 'volatility')
        assert hasattr(snapshot, 'label')
        assert snapshot.volatility == 0.015
        assert snapshot.label == RegimeLabel.HIGH

    def test_regime_label_enum(self):
        """Test that regime labels are properly defined."""
        assert RegimeLabel.LOW.value == "LOW"
        assert RegimeLabel.MEDIUM.value == "MEDIUM"
        assert RegimeLabel.HIGH.value == "HIGH"
        assert RegimeLabel.EXTREME.value == "EXTREME"
        assert RegimeLabel.UNKNOWN.value == "UNKNOWN"

    def test_volatility_edge_cases(self):
        """Test edge cases in volatility classification."""
        classifier = RegimeClassifier()

        # Test None volatility
        snapshot = classifier.classify_by_volatility(None)
        assert snapshot.label == RegimeLabel.UNKNOWN
        assert snapshot.volatility == 0.0

        # Test zero volatility
        snapshot = classifier.classify_by_volatility(0.0)
        assert snapshot.label == RegimeLabel.LOW

        # Test very high volatility
        snapshot = classifier.classify_by_volatility(0.100)
        assert snapshot.label == RegimeLabel.EXTREME
