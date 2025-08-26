"""
Simple tests for the simplified regime detection system.
"""

from src.ai.regime import RegimeClassifier, RegimeConfig, RegimeLabel, RegimeSnapshot


class TestRegimeClassifier:
    """Test the simplified regime classifier."""

    def test_regime_config_defaults(self):
        """Test default configuration values."""
        config = RegimeConfig()
        assert config.thr_extreme == 0.020
        assert config.thr_high == 0.015
        assert config.thr_medium == 0.010

    def test_regime_config_custom(self):
        """Test custom configuration values."""
        config = RegimeConfig(thr_extreme=0.030, thr_high=0.020, thr_medium=0.015)
        assert config.thr_extreme == 0.030
        assert config.thr_high == 0.020
        assert config.thr_medium == 0.015

    def test_regime_classifier_creation(self):
        """Test classifier creation with default and custom config."""
        # Default config
        classifier = RegimeClassifier()
        assert classifier.cfg.thr_extreme == 0.020

        # Custom config
        custom_config = RegimeConfig(thr_extreme=0.025)
        classifier = RegimeClassifier(custom_config)
        assert classifier.cfg.thr_extreme == 0.025

    def test_classify_low_volatility(self):
        """Test classification of low volatility."""
        classifier = RegimeClassifier()
        snapshot = classifier.classify_by_volatility(0.005)

        assert snapshot.volatility == 0.005
        assert snapshot.label == RegimeLabel.LOW

    def test_classify_medium_volatility(self):
        """Test classification of medium volatility."""
        classifier = RegimeClassifier()
        snapshot = classifier.classify_by_volatility(0.012)

        assert snapshot.volatility == 0.012
        assert snapshot.label == RegimeLabel.MEDIUM

    def test_classify_high_volatility(self):
        """Test classification of high volatility."""
        classifier = RegimeClassifier()
        snapshot = classifier.classify_by_volatility(0.018)

        assert snapshot.volatility == 0.018
        assert snapshot.label == RegimeLabel.HIGH

    def test_classify_extreme_volatility(self):
        """Test classification of extreme volatility."""
        classifier = RegimeClassifier()
        snapshot = classifier.classify_by_volatility(0.025)

        assert snapshot.volatility == 0.025
        assert snapshot.label == RegimeLabel.EXTREME

    def test_classify_boundary_values(self):
        """Test classification at boundary values."""
        classifier = RegimeClassifier()

        # At medium threshold (0.010) - should be LOW because not > 0.010
        snapshot = classifier.classify_by_volatility(0.010)
        assert snapshot.label == RegimeLabel.LOW

        # Just above medium threshold
        snapshot = classifier.classify_by_volatility(0.0101)
        assert snapshot.label == RegimeLabel.MEDIUM

        # At high threshold (0.015) - should be MEDIUM because not > 0.015
        snapshot = classifier.classify_by_volatility(0.015)
        assert snapshot.label == RegimeLabel.MEDIUM

        # Just above high threshold
        snapshot = classifier.classify_by_volatility(0.0151)
        assert snapshot.label == RegimeLabel.HIGH

    def test_classify_none_volatility(self):
        """Test classification with None volatility."""
        classifier = RegimeClassifier()
        snapshot = classifier.classify_by_volatility(None)

        assert snapshot.volatility == 0.0
        assert snapshot.label == RegimeLabel.UNKNOWN

    def test_regime_snapshot_creation(self):
        """Test RegimeSnapshot creation and attributes."""
        snapshot = RegimeSnapshot(volatility=0.015, label=RegimeLabel.HIGH)

        assert snapshot.volatility == 0.015
        assert snapshot.label == RegimeLabel.HIGH

    def test_regime_label_values(self):
        """Test that all regime labels have expected values."""
        assert RegimeLabel.UNKNOWN.value == "UNKNOWN"
        assert RegimeLabel.LOW.value == "LOW"
        assert RegimeLabel.MEDIUM.value == "MEDIUM"
        assert RegimeLabel.HIGH.value == "HIGH"
        assert RegimeLabel.EXTREME.value == "EXTREME"

    def test_float_conversion(self):
        """Test that string volatility is converted to float."""
        classifier = RegimeClassifier()
        snapshot = classifier.classify_by_volatility("0.015")

        assert isinstance(snapshot.volatility, float)
        assert snapshot.volatility == 0.015
        assert snapshot.label == RegimeLabel.MEDIUM  # 0.015 is not > 0.015, so MEDIUM
