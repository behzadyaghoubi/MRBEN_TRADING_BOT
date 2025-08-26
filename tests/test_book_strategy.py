"""
Test script for BookStrategy class.
Tests signal generation with sample price data.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.strategies.book_strategy import BookStrategy
from src.config.settings import settings


class TestBookStrategy(unittest.TestCase):
    """Test cases for BookStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = BookStrategy()
        self.sample_data = self._create_sample_data()
        
    def _create_sample_data(self, symbol: str = "XAUUSD", periods: int = 200) -> pd.DataFrame:
        """Create realistic sample OHLCV data for testing."""
        np.random.seed(42)  # For reproducible results
        
        # Generate base price movement with some trend
        base_price = 2000.0  # Starting price for XAUUSD
        prices = [base_price]
        
        # Create some trending movement
        trend = np.linspace(0, 50, periods)  # Upward trend
        noise = np.random.normal(0, 0.3, periods)  # Random noise
        
        for i in range(1, periods):
            # Combine trend and noise
            price_change = (trend[i] - trend[i-1]) + noise[i]
            new_price = prices[-1] * (1 + price_change/100)
            prices.append(new_price)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            # Create realistic OHLC from base price
            volatility = abs(np.random.normal(0, 0.003))  # Price volatility
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = price * (1 + np.random.normal(0, 0.001))
            close_price = price * (1 + np.random.normal(0, 0.001))
            volume = np.random.randint(5000, 15000)
            
            # Ensure OHLC relationships are correct
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            data.append({
                'timestamp': datetime.now() - timedelta(minutes=15*(periods-i)),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'tick_volume': volume
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        print("\n=== Testing Strategy Initialization ===")
        
        # Test basic initialization
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.name, "BookStrategy")
        self.assertIsInstance(self.strategy.parameters, dict)
        
        # Test settings integration
        self.assertEqual(self.strategy.symbol, settings.trading.symbol)
        self.assertEqual(self.strategy.timeframe, settings.trading.timeframe)
        self.assertEqual(self.strategy.base_risk, settings.trading.base_risk)
        
        print(f"✅ Strategy initialized successfully")
        print(f"   Symbol: {self.strategy.symbol}")
        print(f"   Timeframe: {self.strategy.timeframe}")
        print(f"   Base Risk: {self.strategy.base_risk}")
        print(f"   Parameters: {len(self.strategy.parameters)} configured")
    
    def test_indicator_calculation(self):
        """Test technical indicator calculation."""
        print("\n=== Testing Indicator Calculation ===")
        
        # Calculate indicators
        df_with_indicators = self.strategy.calculate_indicators(self.sample_data)
        
        # Check that indicators were added
        expected_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'sma_fast', 'sma_slow', 'ema_fast', 'ema_slow',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
            'volume_ma', 'volume_ratio',
            'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing', 'doji',
            'support', 'resistance'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, df_with_indicators.columns, 
                         f"Indicator {indicator} not found in dataframe")
        
        # Check that indicators have reasonable values
        latest = df_with_indicators.iloc[-1]
        
        # RSI should be between 0 and 100
        self.assertGreaterEqual(latest['rsi'], 0)
        self.assertLessEqual(latest['rsi'], 100)
        
        # Bollinger Bands should have proper relationships
        self.assertGreater(latest['bb_upper'], latest['bb_middle'])
        self.assertLess(latest['bb_lower'], latest['bb_middle'])
        
        # Volume ratio should be positive
        self.assertGreater(latest['volume_ratio'], 0)
        
        print(f"✅ All indicators calculated successfully")
        print(f"   RSI: {latest['rsi']:.2f}")
        print(f"   MACD: {latest['macd']:.5f}")
        print(f"   BB Width: {latest['bb_width']:.4f}")
        print(f"   Volume Ratio: {latest['volume_ratio']:.2f}")
    
    def test_signal_generation(self):
        """Test signal generation functionality."""
        print("\n=== Testing Signal Generation ===")
        
        # Calculate indicators first
        df_with_indicators = self.strategy.calculate_indicators(self.sample_data)
        
        # Generate signal
        signal = self.strategy.generate_signal(df_with_indicators)
        
        # Test signal structure
        self.assertIsInstance(signal, dict)
        required_keys = [
            'signal', 'confidence', 'entry_price', 'stop_loss', 
            'take_profit', 'risk_reward_ratio', 'reasons', 'timestamp',
            'symbol', 'timeframe'
        ]
        
        for key in required_keys:
            self.assertIn(key, signal, f"Signal missing key: {key}")
        
        # Test signal values
        self.assertIn(signal['signal'], ['BUY', 'SELL', 'HOLD'])
        self.assertGreaterEqual(signal['confidence'], 0.0)
        self.assertLessEqual(signal['confidence'], 1.0)
        
        # Test price values based on signal type
        if signal['signal'] in ['BUY', 'SELL']:
            self.assertIsNotNone(signal['entry_price'])
            self.assertGreater(signal['entry_price'], 0)
            self.assertIsNotNone(signal['stop_loss'])
            self.assertIsNotNone(signal['take_profit'])
        else:  # HOLD signal
            self.assertIsNone(signal['entry_price'])
            self.assertIsNone(signal['stop_loss'])
            self.assertIsNone(signal['take_profit'])
        
        self.assertIsInstance(signal['reasons'], list)
        self.assertIsInstance(signal['timestamp'], datetime)
        
        print(f"✅ Signal generated successfully")
        print(f"   Signal: {signal['signal']}")
        print(f"   Confidence: {signal['confidence']:.2f}")
        entry_price_str = f"{signal['entry_price']:.5f}" if signal['entry_price'] is not None else "None"
        stop_loss_str = f"{signal['stop_loss']:.5f}" if signal['stop_loss'] is not None else "None"
        take_profit_str = f"{signal['take_profit']:.5f}" if signal['take_profit'] is not None else "None"
        print(f"   Entry Price: {entry_price_str}")
        print(f"   Stop Loss: {stop_loss_str}")
        print(f"   Take Profit: {take_profit_str}")
        print(f"   Risk/Reward: {signal['risk_reward_ratio']:.2f}")
        print(f"   Reasons: {', '.join(signal['reasons'])}")
    
    def test_batch_signal_generation(self):
        """Test generating signals for entire dataset."""
        print("\n=== Testing Batch Signal Generation ===")
        
        # Generate signals for entire dataset
        df_with_signals = self.strategy.generate_signals(self.sample_data)
        
        # Check that signal column was added
        self.assertIn('signal', df_with_signals.columns)
        
        # Check signal distribution
        signal_counts = df_with_signals['signal'].value_counts()
        total_signals = len(df_with_signals)
        
        print(f"✅ Batch signal generation completed")
        print(f"   Total data points: {total_signals}")
        for signal_type, count in signal_counts.items():
            percentage = (count / total_signals) * 100
            print(f"   {signal_type}: {count} ({percentage:.1f}%)")
        
        # Should have some signals (not all HOLD)
        self.assertGreater(len(signal_counts), 1, "Should have multiple signal types")
    
    def test_compatibility_methods(self):
        """Test compatibility with existing trading engine."""
        print("\n=== Testing Compatibility Methods ===")
        
        # Calculate indicators
        df_with_indicators = self.strategy.calculate_indicators(self.sample_data)
        
        # Test get_latest_signal method
        signal_result = self.strategy.get_latest_signal(df_with_indicators)
        
        # Check SignalResult structure
        self.assertIsInstance(signal_result.signal, str)
        self.assertIsInstance(signal_result.confidence, float)
        self.assertIsInstance(signal_result.features, dict)
        self.assertIsInstance(signal_result.metadata, dict)
        
        # Check features
        expected_features = ['entry_price', 'stop_loss', 'take_profit', 'risk_reward_ratio', 'reasons']
        for feature in expected_features:
            self.assertIn(feature, signal_result.features)
        
        # Check metadata
        expected_metadata = ['symbol', 'timeframe', 'timestamp']
        for meta in expected_metadata:
            self.assertIn(meta, signal_result.metadata)
        
        print(f"✅ Compatibility methods work correctly")
        print(f"   SignalResult: {signal_result.signal} (confidence: {signal_result.confidence:.2f})")
        print(f"   Features: {len(signal_result.features)} items")
        print(f"   Metadata: {len(signal_result.metadata)} items")
    
    def test_parameter_customization(self):
        """Test strategy with custom parameters."""
        print("\n=== Testing Parameter Customization ===")
        
        # Custom parameters
        custom_params = {
            'rsi_period': 21,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'min_confidence': 0.7,
            'risk_reward_ratio': 3.0,
        }
        
        # Create strategy with custom parameters
        custom_strategy = BookStrategy(custom_params)
        
        # Check that custom parameters were applied
        for key, value in custom_params.items():
            self.assertEqual(custom_strategy.parameters[key], value)
        
        # Test signal generation with custom parameters
        df_with_indicators = custom_strategy.calculate_indicators(self.sample_data)
        signal = custom_strategy.generate_signal(df_with_indicators)
        
        print(f"✅ Custom parameters applied successfully")
        print(f"   Custom RSI period: {custom_strategy.parameters['rsi_period']}")
        print(f"   Custom confidence threshold: {custom_strategy.parameters['min_confidence']}")
        print(f"   Signal with custom params: {signal['signal']} (confidence: {signal['confidence']:.2f})")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n=== Testing Edge Cases ===")
        
        # Test with insufficient data
        small_df = self.sample_data.head(10)  # Less than minimum required
        signal = self.strategy.generate_signal(small_df)
        self.assertEqual(signal['signal'], 'HOLD')
        self.assertIn('Insufficient data', signal['reasons'])
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        signal = self.strategy.generate_signal(empty_df)
        self.assertEqual(signal['signal'], 'HOLD')
        
        # Test with missing columns
        invalid_df: pd.DataFrame = self.sample_data[['open', 'high']].copy()  # Missing close, low, volume
        signal = self.strategy.generate_signal(invalid_df)
        self.assertEqual(signal['signal'], 'HOLD')
        
        print(f"✅ Edge cases handled correctly")
        print(f"   Insufficient data: {signal['signal']}")
    
    def test_data_validation(self):
        """Test data validation functionality."""
        print("\n=== Testing Data Validation ===")
        
        # Test valid data
        self.assertTrue(self.strategy.validate_data(self.sample_data))
        
        # Test invalid data (missing required columns)
        invalid_data: pd.DataFrame = self.sample_data[['open', 'high']].copy()  # Missing close, low
        self.assertFalse(self.strategy.validate_data(invalid_data))
        
        # Test insufficient data
        small_data = self.sample_data.head(20)  # Less than 50 rows
        self.assertFalse(self.strategy.validate_data(small_data))
        
        print(f"✅ Data validation works correctly")


def run_comprehensive_test():
    """Run comprehensive test with detailed output."""
    print("🧪 BookStrategy Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestBookStrategy)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run comprehensive test
    success = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 