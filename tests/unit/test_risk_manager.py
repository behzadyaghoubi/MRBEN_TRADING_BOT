"""
Unit tests for EnhancedRiskManager class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from risk.manager import EnhancedRiskManager


class TestEnhancedRiskManager(unittest.TestCase):
    """Test cases for EnhancedRiskManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = EnhancedRiskManager(
            base_risk=0.01,
            min_lot=0.01,
            max_lot=2.0,
            max_open_trades=3,
            max_drawdown=0.10,
            atr_period=14,
            sl_atr_multiplier=2.0,
            tp_atr_multiplier=4.0,
            trailing_atr_multiplier=1.5,
            base_confidence_threshold=0.35,
            adaptive_confidence=True,
            performance_window=20,
            confidence_adjustment_factor=0.1,
            tf_minutes=15
        )

    def test_initialization(self):
        """Test risk manager initialization."""
        self.assertEqual(self.risk_manager.base_risk, 0.01)
        self.assertEqual(self.risk_manager.min_lot, 0.01)
        self.assertEqual(self.risk_manager.max_lot, 2.0)
        self.assertEqual(self.risk_manager.max_open_trades, 3)
        self.assertEqual(self.risk_manager.max_drawdown, 0.10)
        self.assertEqual(self.risk_manager.base_confidence_threshold, 0.35)
        self.assertTrue(self.risk_manager.adaptive_confidence)

    def test_get_atr(self):
        """Test ATR calculation and caching."""
        # Mock MT5 data
        mock_rates = []
        for i in range(20):
            mock_rates.append({
                'high': 2000.0 + i * 0.1,
                'low': 1999.0 + i * 0.1,
                'close': 1999.5 + i * 0.1
            })
        
        with patch('src.risk.manager.mt5.copy_rates_from_pos', return_value=mock_rates):
            with patch('src.risk.manager.mt5.TIMEFRAME_M15', 15):
                atr = self.risk_manager.get_atr("XAUUSD.PRO")
                self.assertIsNotNone(atr)
                self.assertGreater(atr, 0)
                
                # Test caching
                atr2 = self.risk_manager.get_atr("XAUUSD.PRO")
                self.assertEqual(atr, atr2)

    def test_calculate_dynamic_sl_tp(self):
        """Test dynamic SL/TP calculation."""
        # Mock ATR
        with patch.object(self.risk_manager, 'get_atr', return_value=2.0):
            sl, tp = self.risk_manager.calculate_dynamic_sl_tp("XAUUSD.PRO", 2000.0, "BUY")
            
            # For BUY: SL below entry, TP above entry
            self.assertLess(sl, 2000.0)
            self.assertGreater(tp, 2000.0)
            
            # Check distances
            sl_distance = 2000.0 - sl
            tp_distance = tp - 2000.0
            expected_sl_distance = 2.0 * 2.0  # ATR * multiplier
            expected_tp_distance = 2.0 * 4.0  # ATR * multiplier
            
            self.assertAlmostEqual(sl_distance, expected_sl_distance, places=1)
            self.assertAlmostEqual(tp_distance, expected_tp_distance, places=1)
            
            # Test SELL
            sl, tp = self.risk_manager.calculate_dynamic_sl_tp("XAUUSD.PRO", 2000.0, "SELL")
            self.assertGreater(sl, 2000.0)
            self.assertLess(tp, 2000.0)

    def test_calculate_lot_size(self):
        """Test lot size calculation."""
        # Mock symbol info
        mock_info = Mock()
        mock_info.trade_tick_size = 0.1
        mock_info.trade_tick_value = 1.0
        mock_info.volume_min = 0.01
        mock_info.volume_max = 10.0
        mock_info.volume_step = 0.01
        
        with patch('src.risk.manager.mt5.symbol_info', return_value=mock_info):
            lot_size = self.risk_manager.calculate_lot_size(10000.0, 0.01, 2.0, "XAUUSD.PRO")
            
            # Should be within bounds
            self.assertGreaterEqual(lot_size, 0.01)
            self.assertLessEqual(lot_size, 2.0)
            
            # Should be aligned with volume step
            self.assertAlmostEqual(lot_size % 0.01, 0, places=2)

    def test_can_open_new_trade(self):
        """Test trade opening eligibility."""
        # Test basic conditions
        self.assertTrue(self.risk_manager.can_open_new_trade(10000.0, 10000.0, 0))
        
        # Test max open trades limit
        self.assertFalse(self.risk_manager.can_open_new_trade(10000.0, 10000.0, 3))
        
        # Test drawdown limit
        self.assertFalse(self.risk_manager.can_open_new_trade(9500.0, 10000.0, 0))

    def test_trailing_stop_management(self):
        """Test trailing stop functionality."""
        # Add trailing stop
        self.risk_manager.add_trailing_stop(1, 2000.0, 1990.0, True)
        self.assertIn(1, self.risk_manager.trailing_stops)
        
        # Check trailing stop data
        trailing_data = self.risk_manager.trailing_stops[1]
        self.assertEqual(trailing_data['entry_price'], 2000.0)
        self.assertEqual(trailing_data['current_sl'], 1990.0)
        self.assertTrue(trailing_data['is_buy'])
        
        # Remove trailing stop
        self.risk_manager.remove_trailing_stop(1)
        self.assertNotIn(1, self.risk_manager.trailing_stops)

    def test_update_trailing_stops(self):
        """Test trailing stop updates."""
        # Add a trailing stop
        self.risk_manager.add_trailing_stop(1, 2000.0, 1990.0, True)
        
        # Mock ATR and tick data
        with patch.object(self.risk_manager, 'get_atr', return_value=2.0):
            with patch('src.risk.manager.mt5.symbol_info_tick') as mock_tick:
                mock_tick.return_value.bid = 2010.0
                mock_tick.return_value.ask = 2010.1
                
                # Update trailing stops
                mods = self.risk_manager.update_trailing_stops("XAUUSD.PRO")
                
                # Should have modifications for profitable position
                if mods:
                    self.assertIsInstance(mods, list)
                    for mod in mods:
                        self.assertIn('ticket', mod)
                        self.assertIn('new_sl', mod)

    def test_confidence_threshold_management(self):
        """Test confidence threshold functionality."""
        # Test initial threshold
        self.assertEqual(self.risk_manager.get_current_confidence_threshold(), 0.35)
        
        # Test threshold update
        self.risk_manager.current_confidence_threshold = 0.40
        self.assertEqual(self.risk_manager.get_current_confidence_threshold(), 0.40)

    def test_performance_update(self):
        """Test performance-based confidence adjustment."""
        # Mock MT5 history
        mock_deal = Mock()
        mock_deal.symbol = "XAUUSD.PRO"
        mock_deal.entry = 1  # DEAL_ENTRY_OUT
        mock_deal.profit = 100.0  # Profitable trade
        mock_deal.time = datetime.now()
        
        with patch('src.risk.manager.mt5.history_deals_get', return_value=[mock_deal]):
            self.risk_manager.update_performance_from_history("XAUUSD.PRO")
            
            # Should have recorded performance
            self.assertIn(100.0, self.risk_manager.recent_performances)

    def test_adaptive_confidence(self):
        """Test adaptive confidence adjustment."""
        # Set up performance data
        self.risk_manager.recent_performances = [100.0] * 20  # 100% win rate
        
        # Mock performance update
        with patch.object(self.risk_manager, 'get_atr', return_value=2.0):
            with patch('src.risk.manager.mt5.history_deals_get', return_value=[]):
                self.risk_manager.update_performance_from_history("XAUUSD.PRO")
                
                # With high win rate, confidence should decrease
                if self.risk_manager.adaptive_confidence:
                    self.assertLessEqual(
                        self.risk_manager.current_confidence_threshold,
                        self.risk_manager.base_confidence_threshold
                    )

    def test_error_handling(self):
        """Test error handling in various methods."""
        # Test ATR calculation with no data
        with patch('src.risk.manager.mt5.copy_rates_from_pos', return_value=None):
            atr = self.risk_manager.get_atr("XAUUSD.PRO")
            self.assertIsNone(atr)
        
        # Test lot size calculation with invalid data
        with patch('src.risk.manager.mt5.symbol_info', return_value=None):
            lot_size = self.risk_manager.calculate_lot_size(10000.0, 0.01, 0, "XAUUSD.PRO")
            self.assertEqual(lot_size, 0.01)  # Should return min_lot
        
        # Test trailing stop update with no MT5
        with patch('src.risk.manager.mt5.symbol_info_tick', side_effect=Exception("MT5 error")):
            mods = self.risk_manager.update_trailing_stops("XAUUSD.PRO")
            self.assertEqual(mods, [])

    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases."""
        # Test with zero ATR
        with patch.object(self.risk_manager, 'get_atr', return_value=0):
            sl, tp = self.risk_manager.calculate_dynamic_sl_tp("XAUUSD.PRO", 2000.0, "BUY")
            # Should handle gracefully
            self.assertIsInstance(sl, float)
            self.assertIsInstance(tp, float)
        
        # Test with very large ATR
        with patch.object(self.risk_manager, 'get_atr', return_value=1000.0):
            sl, tp = self.risk_manager.calculate_dynamic_sl_tp("XAUUSD.PRO", 2000.0, "BUY")
            # Should handle gracefully
            self.assertIsInstance(sl, float)
            self.assertIsInstance(tp, float)
        
        # Test with invalid lot size parameters
        lot_size = self.risk_manager.calculate_lot_size(0, 0.01, 2.0, "XAUUSD.PRO")
        self.assertEqual(lot_size, 0.01)  # Should return min_lot

    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Test with invalid parameters
        with self.assertRaises(ValueError):
            EnhancedRiskManager(base_risk=-0.01)
        
        with self.assertRaises(ValueError):
            EnhancedRiskManager(min_lot=-0.01)
        
        with self.assertRaises(ValueError):
            EnhancedRiskManager(max_lot=0)
        
        # Test with valid parameters
        try:
            risk_mgr = EnhancedRiskManager(
                base_risk=0.02,
                min_lot=0.05,
                max_lot=5.0,
                max_open_trades=5
            )
            self.assertIsNotNone(risk_mgr)
        except Exception as e:
            self.fail(f"Valid configuration should not raise exception: {e}")

    def test_memory_management(self):
        """Test memory management in trailing stops."""
        # Add many trailing stops
        for i in range(100):
            self.risk_manager.add_trailing_stop(i, 2000.0, 1990.0, True)
        
        self.assertEqual(len(self.risk_manager.trailing_stops), 100)
        
        # Remove all
        for i in range(100):
            self.risk_manager.remove_trailing_stop(i)
        
        self.assertEqual(len(self.risk_manager.trailing_stops), 0)

    def tearDown(self):
        """Clean up after tests."""
        pass


if __name__ == '__main__':
    unittest.main()
