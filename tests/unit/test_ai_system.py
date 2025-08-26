"""
Unit tests for MRBENAdvancedAISystem class.
"""

import os
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ai.system import MRBENAdvancedAISystem


class TestMRBENAdvancedAISystem(unittest.TestCase):
    """Test cases for MRBENAdvancedAISystem class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ai_system = MRBENAdvancedAISystem()

    def test_initialization(self):
        """Test AI system initialization."""
        self.assertIsNotNone(self.ai_system.logger)
        self.assertEqual(self.ai_system.ensemble_weights, [0.4, 0.3, 0.3])
        self.assertIsInstance(self.ai_system.models, dict)
        self.assertIsInstance(self.ai_system.scalers, dict)
        self.assertIsInstance(self.ai_system.label_encoders, dict)

    def test_load_models(self):
        """Test model loading functionality."""
        # Mock file existence and model loading
        with patch('os.path.exists', return_value=True):
            with patch('tensorflow.keras.models.load_model') as mock_load_lstm:
                with patch('joblib.load') as mock_load_ml:
                    mock_load_lstm.return_value = Mock()
                    mock_load_ml.return_value = {'model': Mock(), 'scaler': Mock()}

                    self.ai_system.load_models()

                    # Check if models were loaded
                    self.assertIn('lstm', self.ai_system.models)
                    self.assertIn('ml_filter', self.ai_system.models)
                    self.assertIn('ml_filter', self.ai_system.scalers)

    def test_load_models_with_missing_files(self):
        """Test model loading when files are missing."""
        # Mock file existence to return False
        with patch('os.path.exists', return_value=False):
            self.ai_system.load_models()

            # Should handle missing files gracefully
            self.assertEqual(len(self.ai_system.models), 0)

    def test_load_models_with_errors(self):
        """Test model loading error handling."""
        # Mock file existence but cause loading error
        with patch('os.path.exists', return_value=True):
            with patch('tensorflow.keras.models.load_model', side_effect=Exception("Load error")):
                self.ai_system.load_models()

                # Should handle errors gracefully
                self.assertEqual(len(self.ai_system.models), 0)

    def test_calculate_atr(self):
        """Test ATR calculation."""
        # Create test data
        df = pd.DataFrame(
            {
                'high': [2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
                'low': [1999.0, 2000.0, 2001.0, 2002.0, 2003.0],
                'close': [1999.5, 2000.5, 2001.5, 2002.5, 2003.5],
            }
        )

        atr = self.ai_system._calculate_atr(df, period=3)
        self.assertIsInstance(atr, pd.Series)
        self.assertEqual(len(atr), 5)

    def test_generate_meta_features(self):
        """Test meta-feature generation."""
        # Create test data
        df = pd.DataFrame(
            {
                'time': pd.date_range('2024-01-01', periods=5, freq='H'),
                'close': [2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
                'high': [2000.5, 2001.5, 2002.5, 2003.5, 2004.5],
                'low': [1999.5, 2000.5, 2001.5, 2002.5, 2003.5],
            }
        )

        result_df = self.ai_system.generate_meta_features(df)

        # Check if new features were added
        self.assertIn('hour', result_df.columns)
        self.assertIn('day_of_week', result_df.columns)
        self.assertIn('session', result_df.columns)
        self.assertIn('session_encoded', result_df.columns)
        self.assertIn('rsi', result_df.columns)
        self.assertIn('macd', result_df.columns)
        self.assertIn('atr', result_df.columns)
        self.assertIn('sma_20', result_df.columns)
        self.assertIn('sma_50', result_df.columns)

    def test_session_classification(self):
        """Test trading session classification."""
        # Create test data with different hours
        df = pd.DataFrame(
            {'time': pd.date_range('2024-01-01', periods=24, freq='H'), 'close': range(24)}
        )

        result_df = self.ai_system.generate_meta_features(df)

        # Check session classification
        sessions = result_df['session'].unique()
        self.assertIn('Asia', sessions)
        self.assertIn('London', sessions)
        self.assertIn('NY', sessions)

    def test_technical_prediction(self):
        """Test technical analysis prediction."""
        # Test with single row
        df_single = pd.DataFrame(
            {
                'open': [2000.0],
                'close': [2005.0],
                'rsi': [30.0],
                'macd': [0.05],
                'macd_signal': [0.02],
            }
        )

        result = self.ai_system._tech_pred(df_single)
        self.assertIn('signal', result)
        self.assertIn('confidence', result)
        self.assertIn('score', result)
        self.assertEqual(result['signal'], 1)  # Should be bullish

        # Test with multiple rows
        df_multi = pd.DataFrame(
            {
                'rsi': [30.0, 70.0, 50.0],
                'macd': [0.05, -0.05, 0.0],
                'macd_signal': [0.02, -0.02, 0.0],
            }
        )

        result = self.ai_system._tech_pred(df_multi)
        self.assertIn('signal', result)
        self.assertIn('confidence', result)
        self.assertIn('score', result)

    def test_lstm_simple_prediction(self):
        """Test LSTM prediction."""
        # Test with single row
        df_single = pd.DataFrame({'open': [2000.0], 'close': [2005.0], 'rsi': [30.0]})

        result = self.ai_system._lstm_simple(df_single)
        self.assertIn('signal', result)
        self.assertIn('confidence', result)
        self.assertIn('score', result)

        # Test with multiple rows
        df_multi = pd.DataFrame({'rsi': [30.0, 70.0, 50.0]})

        result = self.ai_system._lstm_simple(df_multi)
        self.assertIn('signal', result)
        self.assertIn('confidence', result)
        self.assertIn('score', result)

    def test_ml_prediction(self):
        """Test ML model prediction."""
        # Test without ML model
        df = pd.DataFrame({'close': [2000.0]})
        result = self.ai_system._ml_pred(df)
        self.assertEqual(result['signal'], 0)
        self.assertEqual(result['confidence'], 0.5)

        # Test with ML model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        mock_scaler = Mock()
        mock_scaler.feature_names_in_ = ['close', 'rsi']
        mock_scaler.transform.return_value = np.array([[0.5, 0.6]])

        self.ai_system.models['ml_filter'] = mock_model
        self.ai_system.scalers['ml_filter'] = mock_scaler

        df_with_features = pd.DataFrame({'close': [2000.0], 'rsi': [50.0]})

        result = self.ai_system._ml_pred(df_with_features)
        self.assertEqual(result['signal'], 1)
        self.assertEqual(result['confidence'], 0.7)

    def test_ensemble_signal_generation(self):
        """Test ensemble signal generation."""
        # Mock market data
        market_data = {
            'time': datetime.now(),
            'open': 2000.0,
            'high': 2005.0,
            'low': 1995.0,
            'close': 2002.0,
            'volume': 100,
        }

        # Mock AI predictions
        with patch.object(
            self.ai_system,
            '_tech_pred',
            return_value={'signal': 1, 'confidence': 0.8, 'score': 0.6},
        ):
            with patch.object(
                self.ai_system,
                '_lstm_simple',
                return_value={'signal': 1, 'confidence': 0.9, 'score': 0.7},
            ):
                with patch.object(
                    self.ai_system,
                    '_ml_pred',
                    return_value={'signal': 1, 'confidence': 0.7, 'score': 0.5},
                ):
                    signal = self.ai_system.generate_ensemble_signal(market_data)

                    self.assertIn('signal', signal)
                    self.assertIn('confidence', signal)
                    self.assertIn('score', signal)
                    self.assertIn('source', signal)
                    self.assertEqual(signal['signal'], 1)  # Should be bullish

    def test_ensemble_signal_with_mixed_predictions(self):
        """Test ensemble signal with mixed predictions."""
        market_data = {
            'time': datetime.now(),
            'open': 2000.0,
            'high': 2005.0,
            'low': 1995.0,
            'close': 2002.0,
            'volume': 100,
        }

        # Mock mixed predictions
        with patch.object(
            self.ai_system,
            '_tech_pred',
            return_value={'signal': 1, 'confidence': 0.8, 'score': 0.6},
        ):
            with patch.object(
                self.ai_system,
                '_lstm_simple',
                return_value={'signal': -1, 'confidence': 0.9, 'score': -0.7},
            ):
                with patch.object(
                    self.ai_system,
                    '_ml_pred',
                    return_value={'signal': 0, 'confidence': 0.7, 'score': 0.0},
                ):
                    signal = self.ai_system.generate_ensemble_signal(market_data)

                    # Should handle mixed signals gracefully
                    self.assertIn('signal', signal)
                    self.assertIn('confidence', signal)
                    self.assertIn('score', signal)

    def test_ensemble_proba_win(self):
        """Test ensemble probability calculation."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                'time': pd.date_range('2024-01-01', periods=5, freq='H'),
                'close': [2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
            }
        )

        # Mock predictions
        with patch.object(
            self.ai_system,
            '_tech_pred',
            return_value={'signal': 1, 'confidence': 0.8, 'score': 0.6},
        ):
            with patch.object(
                self.ai_system,
                '_lstm_simple',
                return_value={'signal': 1, 'confidence': 0.9, 'score': 0.7},
            ):
                with patch.object(
                    self.ai_system,
                    '_ml_pred',
                    return_value={'signal': 1, 'confidence': 0.7, 'score': 0.5},
                ):
                    proba = self.ai_system.ensemble_proba_win(df)

                    self.assertIsInstance(proba, float)
                    self.assertGreaterEqual(proba, 0.0)
                    self.assertLessEqual(proba, 1.0)

    def test_error_handling_in_ensemble(self):
        """Test error handling in ensemble signal generation."""
        market_data = {
            'time': datetime.now(),
            'open': 2000.0,
            'high': 2005.0,
            'low': 1995.0,
            'close': 2002.0,
            'volume': 100,
        }

        # Mock error in prediction
        with patch.object(self.ai_system, '_tech_pred', side_effect=Exception("Test error")):
            signal = self.ai_system.generate_ensemble_signal(market_data)

            # Should return default signal on error
            self.assertEqual(signal['signal'], 0)
            self.assertEqual(signal['confidence'], 0.5)
            self.assertEqual(signal['source'], 'Error')

    def test_feature_engineering_robustness(self):
        """Test robustness of feature engineering."""
        # Test with minimal data
        df_minimal = pd.DataFrame({'time': [datetime.now()], 'close': [2000.0]})

        result = self.ai_system.generate_meta_features(df_minimal)

        # Should handle minimal data gracefully
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), len(df_minimal.columns))

    def test_model_availability_handling(self):
        """Test handling of missing models."""
        # Clear models
        self.ai_system.models.clear()

        market_data = {
            'time': datetime.now(),
            'open': 2000.0,
            'high': 2005.0,
            'low': 1995.0,
            'close': 2002.0,
            'volume': 100,
        }

        # Should fall back to technical analysis
        signal = self.ai_system.generate_ensemble_signal(market_data)
        self.assertIn('signal', signal)
        self.assertIn('confidence', signal)

    def test_scaler_output_format_handling(self):
        """Test handling of different scaler output formats."""
        # Test with sklearn >= 1.2 format
        mock_scaler = Mock()
        mock_scaler.feature_names_in_ = ['close', 'rsi']
        mock_scaler.transform.return_value = np.array([[0.5, 0.6]])

        # Test set_output method availability
        if hasattr(mock_scaler, 'set_output'):
            mock_scaler.set_output.return_value = None

        self.ai_system.scalers['ml_filter'] = mock_scaler

        # Should handle both old and new sklearn versions
        df = pd.DataFrame({'close': [2000.0], 'rsi': [50.0]})
        result = self.ai_system._ml_pred(df)
        self.assertIn('signal', result)

    def test_memory_efficiency(self):
        """Test memory efficiency of feature generation."""
        # Create large DataFrame
        large_df = pd.DataFrame(
            {
                'time': pd.date_range('2024-01-01', periods=1000, freq='H'),
                'close': np.random.randn(1000) + 2000.0,
                'high': np.random.randn(1000) + 2000.0,
                'low': np.random.randn(1000) + 2000.0,
            }
        )

        # Should handle large datasets efficiently
        start_memory = self._get_memory_usage()
        result = self.ai_system.generate_meta_features(large_df)
        end_memory = self._get_memory_usage()

        # Memory increase should be reasonable
        memory_increase = end_memory - start_memory
        self.assertLess(memory_increase, 100)  # Less than 100MB increase

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except ImportError:
            return 0

    def test_logging_functionality(self):
        """Test logging functionality."""
        # Test that logger is properly configured
        self.assertIsNotNone(self.ai_system.logger)
        self.assertEqual(self.ai_system.logger.level, 20)  # INFO level

        # Test that logger doesn't propagate to root
        self.assertFalse(self.ai_system.logger.propagate)

    def tearDown(self):
        """Clean up after tests."""
        pass


if __name__ == '__main__':
    unittest.main()
