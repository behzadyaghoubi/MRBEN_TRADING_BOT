"""
Unit tests for market regime detection system.

Tests features, regime classification, and smoothing functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai.features import (
    atr, adx, realized_vol, rolling_z, session_tag, 
    spread_bp, calculate_returns
)
from ai.regime import (
    RegimeLabel, RegimeSnapshot, RegimeConfig, 
    RegimeClassifier, create_regime_classifier
)


class TestFeatures:
    """Test market feature calculations."""
    
    def setup_method(self):
        """Set up test data."""
        # Create synthetic OHLC data
        np.random.seed(42)
        n_bars = 100
        
        # Generate realistic price data
        base_price = 2000.0
        returns = np.random.normal(0, 0.01, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.test_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_bars))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
        
        # Ensure high >= low
        self.test_data['high'] = np.maximum(self.test_data['high'], self.test_data['close'])
        self.test_data['low'] = np.minimum(self.test_data['low'], self.test_data['close'])
        
        # Create timestamps
        start_time = datetime.now() - timedelta(days=n_bars)
        self.test_data.index = pd.date_range(start=start_time, periods=n_bars, freq='H')
    
    def test_atr_calculation(self):
        """Test ATR calculation."""
        high = self.test_data['high'].values
        low = self.test_data['low'].values
        close = self.test_data['close'].values
        
        atr_values = atr(high, low, close, n=14)
        
        assert len(atr_values) == len(high)
        assert not np.any(np.isnan(atr_values[13:]))  # After window
        assert np.all(atr_values >= 0)  # ATR should be non-negative
        assert atr_values[0] == high[0] - low[0]  # First value
    
    def test_adx_calculation(self):
        """Test ADX calculation."""
        high = self.test_data['high'].values
        low = self.test_data['low'].values
        close = self.test_data['close'].values
        
        adx_values = adx(high, low, close, n=14)
        
        assert len(adx_values) == len(high)
        assert not np.any(np.isnan(adx_values[13:]))  # After window
        assert np.all(adx_values >= 0)  # ADX should be non-negative
        assert np.all(adx_values <= 100)  # ADX should be <= 100
    
    def test_realized_volatility(self):
        """Test realized volatility calculation."""
        returns = calculate_returns(self.test_data['close'].values)
        rv_values = realized_vol(returns, n=20)
        
        assert len(rv_values) == len(returns)
        assert not np.any(np.isnan(rv_values[19:]))  # After window
        assert np.all(rv_values >= 0)  # Volatility should be non-negative
    
    def test_rolling_zscore(self):
        """Test rolling Z-score calculation."""
        data = self.test_data['close'].values
        z_scores = rolling_z(data, n=20)
        
        assert len(z_scores) == len(data)
        assert not np.any(np.isnan(z_scores[19:]))  # After window
        # Z-scores should be roughly centered around 0
        assert abs(np.mean(z_scores[19:])) < 0.1
    
    def test_session_tagging(self):
        """Test session tagging."""
        timestamps = self.test_data.index
        sessions = session_tag(timestamps)
        
        assert len(sessions) == len(timestamps)
        assert np.all(np.isin(sessions, [0, 1, 2, 3]))  # Valid session codes
        assert sessions.dtype == np.int64
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        bid = np.array([1999.0, 2000.0, 2001.0])
        ask = np.array([2001.0, 2002.0, 2003.0])
        
        spread = spread_bp(bid, ask)
        
        assert len(spread) == len(bid)
        assert np.all(spread > 0)  # Spread should be positive
        # Expected: (2001-1999)/1999 * 10000 = 10.0 bp
        assert abs(spread[0] - 10.0) < 0.1
    
    def test_returns_calculation(self):
        """Test log returns calculation."""
        prices = self.test_data['close'].values
        returns = calculate_returns(prices)
        
        assert len(returns) == len(prices)
        assert np.isnan(returns[0])  # First return should be NaN
        assert not np.any(np.isnan(returns[1:]))  # Rest should be valid
    
    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError):
            atr([1, 2, 3], [1, 2, 3], [1, 2, 3], n=0)  # Invalid n
        
        with pytest.raises(ValueError):
            atr([1, 2], [1, 2, 3], [1, 2, 3], n=2)  # Mismatched lengths
        
        with pytest.raises(ValueError):
            realized_vol([1, 2, 3], n=5)  # n > data length


class TestRegimeClassification:
    """Test regime classification functionality."""
    
    def setup_method(self):
        """Set up test data and classifier."""
        self.config = RegimeConfig(
            windows={"vol_atr": 14, "adx": 14, "return_ms": 20, "rv": 60},
            thresholds={
                "trend_adx": 20.0,
                "vol_high": 0.012,
                "spread_wide_bp": 15.0,
                "chop_z": 1.0
            },
            smoothing={"method": "ema", "alpha": 0.2, "min_dwell_bars": 5}
        )
        
        self.classifier = RegimeClassifier(self.config)
        
        # Create test data
        np.random.seed(42)
        n_bars = 100
        
        # Generate different regime scenarios
        self.trend_data = self._create_trend_data(n_bars)
        self.range_data = self._create_range_data(n_bars)
        self.volatile_data = self._create_volatile_data(n_bars)
    
    def _create_trend_data(self, n_bars):
        """Create trending market data."""
        base_price = 2000.0
        trend = np.linspace(0, 0.1, n_bars)  # 10% trend
        noise = np.random.normal(0, 0.005, n_bars)
        prices = base_price * np.exp(trend + noise)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_bars))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
    
    def _create_range_data(self, n_bars):
        """Create ranging market data."""
        base_price = 2000.0
        # Oscillating pattern
        cycle = np.sin(np.linspace(0, 4*np.pi, n_bars)) * 0.02
        prices = base_price * (1 + cycle)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
    
    def _create_volatile_data(self, n_bars):
        """Create high volatility market data."""
        base_price = 2000.0
        # High volatility with jumps
        volatility = 0.03  # 3% daily volatility
        returns = np.random.normal(0, volatility, n_bars)
        # Add occasional jumps
        jumps = np.random.choice([0, 1], n_bars, p=[0.95, 0.05])
        returns += jumps * np.random.normal(0, 0.05, n_bars)
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.008, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.008, n_bars))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
    
    def test_trend_regime_detection(self):
        """Test trend regime detection."""
        snapshot = self.classifier.infer_regime(self.trend_data)
        
        assert snapshot.label in [RegimeLabel.TREND, RegimeLabel.UNKNOWN]
        assert snapshot.confidence > 0.5
        assert snapshot.scores['adx'] > 15  # Should have some trend strength
    
    def test_range_regime_detection(self):
        """Test range regime detection."""
        snapshot = self.classifier.infer_regime(self.range_data)
        
        assert snapshot.label in [RegimeLabel.RANGE, RegimeLabel.UNKNOWN]
        assert snapshot.confidence > 0.4
        # ADX should be low in ranging markets
        assert snapshot.scores['adx'] < 25
    
    def test_volatile_regime_detection(self):
        """Test high volatility regime detection."""
        snapshot = self.classifier.infer_regime(self.volatile_data)
        
        # Should detect either HIGH_VOL or TREND
        assert snapshot.label in [RegimeLabel.HIGH_VOL, RegimeLabel.TREND, RegimeLabel.UNKNOWN]
        assert snapshot.confidence > 0.4
    
    def test_regime_smoothing(self):
        """Test regime smoothing functionality."""
        # Test EMA smoothing
        config_ema = RegimeConfig(
            smoothing={"method": "ema", "alpha": 0.3, "min_dwell_bars": 3}
        )
        classifier_ema = RegimeClassifier(config_ema)
        
        # Test majority smoothing
        config_majority = RegimeConfig(
            smoothing={"method": "majority", "alpha": 0.2, "min_dwell_bars": 3}
        )
        classifier_majority = RegimeClassifier(config_majority)
        
        # Both should work without errors
        snapshot1 = classifier_ema.infer_regime(self.trend_data)
        snapshot2 = classifier_majority.infer_regime(self.range_data)
        
        assert snapshot1 is not None
        assert snapshot2 is not None
    
    def test_dwell_time_enforcement(self):
        """Test minimum dwell time enforcement."""
        config_dwell = RegimeConfig(
            smoothing={"method": "ema", "alpha": 0.2, "min_dwell_bars": 10}
        )
        classifier_dwell = RegimeClassifier(config_dwell)
        
        # First classification
        snapshot1 = classifier_dwell.infer_regime(self.trend_data)
        initial_regime = snapshot1.label
        
        # Force regime change (this would normally happen with different data)
        classifier_dwell.prev_snapshot = snapshot1
        
        # Create a snapshot with different regime
        new_snapshot = RegimeSnapshot(
            label=RegimeLabel.RANGE if initial_regime == RegimeLabel.TREND else RegimeLabel.TREND,
            scores={"adx": 15.0, "rv": 0.008, "z": 0.5, "spread_bp": 8.0},
            session="london",
            ts=pd.Timestamp.now(),
            confidence=0.8
        )
        
        # Apply smoothing
        smoothed = classifier_dwell._smooth_regime(new_snapshot)
        
        # Should enforce dwell time
        assert smoothed.label == initial_regime
        assert smoothed.dwell_bars > 0
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid alpha
        with pytest.raises(ValueError):
            RegimeConfig(smoothing={"method": "ema", "alpha": 1.5, "min_dwell_bars": 5})
        
        # Invalid dwell bars
        with pytest.raises(ValueError):
            RegimeConfig(smoothing={"method": "ema", "alpha": 0.2, "min_dwell_bars": -1})
        
        # Invalid windows
        with pytest.raises(ValueError):
            RegimeConfig(windows={"vol_atr": 0, "adx": 14, "return_ms": 20, "rv": 60})
    
    def test_regime_summary(self):
        """Test regime summary functionality."""
        # Add some regime history
        for _ in range(5):
            self.classifier.infer_regime(self.trend_data)
        
        summary = self.classifier.get_regime_summary(lookback_days=1)
        
        assert "total_snapshots" in summary
        assert "regime_counts" in summary
        assert "confidence_by_regime" in summary
        assert "current_regime" in summary
    
    def test_factory_function(self):
        """Test factory function for creating classifier."""
        config_dict = {
            "windows": {"vol_atr": 10, "adx": 10},
            "thresholds": {"trend_adx": 25.0},
            "smoothing": {"method": "majority", "alpha": 0.1, "min_dwell_bars": 5}
        }
        
        classifier = create_regime_classifier(config_dict)
        
        assert isinstance(classifier, RegimeClassifier)
        assert classifier.config.windows["vol_atr"] == 10
        assert classifier.config.thresholds["trend_adx"] == 25.0
        assert classifier.config.smoothing["method"] == "majority"


class TestRegimeIntegration:
    """Test integration between features and regime classification."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_bars = 100
        
        # Create realistic market data
        base_price = 2000.0
        trend = np.linspace(0, 0.05, n_bars)  # 5% trend
        volatility = 0.015  # 1.5% daily volatility
        returns = np.random.normal(trend, volatility, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.market_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.004, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.004, n_bars))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
        
        # Ensure high >= low
        self.market_data['high'] = np.maximum(self.market_data['high'], self.market_data['close'])
        self.market_data['low'] = np.minimum(self.market_data['low'], self.market_data['close'])
        
        # Add timestamps
        start_time = datetime.now() - timedelta(days=n_bars)
        self.market_data.index = pd.date_range(start=start_time, periods=n_bars, freq='H')
    
    def test_end_to_end_regime_detection(self):
        """Test complete regime detection pipeline."""
        config = RegimeConfig(
            windows={"vol_atr": 14, "adx": 14, "return_ms": 20, "rv": 60},
            thresholds={
                "trend_adx": 20.0,
                "vol_high": 0.012,
                "spread_wide_bp": 15.0,
                "chop_z": 1.0
            },
            smoothing={"method": "ema", "alpha": 0.2, "min_dwell_bars": 5}
        )
        
        classifier = RegimeClassifier(config)
        
        # Test with microstructure data
        micro = {
            'bid': self.market_data['close'].values * 0.9999,  # 1bp spread
            'ask': self.market_data['close'].values * 1.0001
        }
        
        snapshot = classifier.infer_regime(self.market_data, micro)
        
        assert snapshot is not None
        assert snapshot.label in RegimeLabel
        assert snapshot.confidence > 0
        assert snapshot.session in ["closed", "asia", "london", "ny"]
        assert "spread_bp" in snapshot.scores
    
    def test_regime_persistence(self):
        """Test that regime state persists correctly."""
        config = RegimeConfig(
            smoothing={"method": "ema", "alpha": 0.1, "min_dwell_bars": 3}
        )
        
        classifier = RegimeClassifier(config)
        
        # First classification
        snapshot1 = classifier.infer_regime(self.market_data)
        regime1 = snapshot1.label
        confidence1 = snapshot1.confidence
        
        # Second classification with same data
        snapshot2 = classifier.infer_regime(self.market_data)
        regime2 = snapshot2.label
        confidence2 = snapshot2.confidence
        
        # Regime should persist
        assert regime2 == regime1
        # Confidence should be smoothed
        assert abs(confidence2 - confidence1) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
