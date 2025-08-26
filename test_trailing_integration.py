import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
import time

# Add the current directory to Python path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock MT5 before importing our modules
sys.modules['MetaTrader5'] = Mock()
sys.modules['MetaTrader5'].TRADE_RETCODE_DONE = 10009
sys.modules['MetaTrader5'].ORDER_TYPE_BUY = 0
sys.modules['MetaTrader5'].ORDER_TYPE_SELL = 1
sys.modules['MetaTrader5'].TIMEFRAME_M15 = 15
sys.modules['MetaTrader5'].TIMEFRAME_M5 = 5
sys.modules['MetaTrader5'].TRADE_ACTION_DEAL = 1
sys.modules['MetaTrader5'].ORDER_TIME_GTC = 0
sys.modules['MetaTrader5'].ORDER_FILLING_IOC = 1

# Now import our modules
from live_trader_clean import MT5LiveTrader, MT5Config

class TestTrailingIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock config for testing
        self.mock_config_data = {
            "credentials": {
                "login": 12345,
                "password": "test",
                "server": "test-server"
            },
            "trading": {
                "symbol": "XAUUSD.PRO",
                "timeframe": 15,
                "bars": 500,
                "magic_number": 20250721,
                "sessions": ["24h"],
                "max_spread_points": 200,
                "use_risk_based_volume": False,
                "fixed_volume": 0.1,
                "sleep_seconds": 12,
                "retry_delay": 5,
                "consecutive_signals_required": 1,
                "lstm_timesteps": 50,
                "cooldown_seconds": 180
            },
            "risk": {
                "base_risk": 0.01,
                "min_lot": 0.01,
                "max_lot": 2.0,
                "max_open_trades": 3,
                "max_daily_loss": 0.02,
                "max_trades_per_day": 10
            },
            "flags": {
                "demo_mode": True
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "logs/trading_bot.log",
                "trade_log_path": "data/trade_log_gold.csv"
            },
            "models": {
                "use_rl": True,
                "use_lstm": True,
                "use_technical": True,
                "ml_filter_threshold": 0.5
            },
            "notifications": {
                "telegram_enabled": False,
                "email_enabled": False,
                "daily_summary": True
            }
        }
        
        # Mock the config file
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(self.mock_config_data)
            self.config = MT5Config()
    
    @patch('live_trader_clean.mt5')
    def test_trailing_stop_integration_no_real_orders(self, mock_mt5):
        """Test trailing stop integration without real orders."""
        print("\n🧪 Testing Trailing Stop Integration (No Real Orders)")
        
        # Create trader instance
        trader = MT5LiveTrader()
        
        # Mock MT5 initialization
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        mock_mt5.account_info.return_value = Mock()
        
        # Mock symbol info
        mock_symbol_info = Mock()
        mock_symbol_info.point = 0.01
        mock_symbol_info.trade_stops_level = 10
        mock_symbol_info.trade_freeze_level = 5
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        # Mock order send result
        mock_order_result = Mock()
        mock_order_result.retcode = 10009  # TRADE_RETCODE_DONE
        mock_order_result.order = 12345  # Order ID
        mock_order_result.comment = "Order executed successfully"
        mock_mt5.order_send.return_value = mock_order_result
        
        # Mock position that should be found after order execution
        mock_position = Mock()
        mock_position.ticket = 67890  # Position ticket (different from order ID)
        mock_position.volume = 0.1
        mock_position.price_open = 2000.0
        mock_position.sl = 1990.0
        mock_position.tp = 2020.0
        mock_position.type = 0  # Buy position
        
        # Mock positions_get to return our position
        mock_mt5.positions_get.return_value = [mock_position]
        
        # Mock the risk manager's add_trailing_stop method
        trader.risk_manager.add_trailing_stop = Mock()
        
        # Create mock signal data
        signal_data = {
            'signal': 1,  # Buy signal
            'confidence': 0.75,
            'ensemble_score': 0.8
        }
        
        # Create mock market data
        import pandas as pd
        mock_df = pd.DataFrame({
            'close': [2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
            'high': [2005.0, 2006.0, 2007.0, 2008.0, 2009.0],
            'low': [1995.0, 1996.0, 1997.0, 1998.0, 1999.0],
            'open': [1998.0, 1999.0, 2000.0, 2001.0, 2002.0]
        })
        
        # Mock spread check to pass
        with patch('live_trader_clean.is_spread_ok_dynamic') as mock_spread:
            mock_spread.return_value = (True, 0.2, 0.3)  # Spread OK
            
            # Mock ATR calculation for SL/TP
            trader.risk_manager.get_atr = Mock(return_value=2.0)
            
            # Mock get_current_tick to return None (use last close price)
            trader.data_manager.get_current_tick = Mock(return_value=None)
            
            # Mock MT5 availability to ensure real order execution
            with patch('live_trader_clean.MT5_AVAILABLE', True):
                trader.mt5_connected = True
                
                # Execute the trade
                result = trader._execute_trade(signal_data, mock_df)
            
            print(f"✅ Trade execution result: {result}")
            
            # Verify order was sent
            mock_mt5.order_send.assert_called_once()
            
            # Verify the order parameters
            order_call_args = mock_mt5.order_send.call_args[1]
            print(f"📋 Order parameters:")
            print(f"   - Symbol: {order_call_args.get('symbol')}")
            print(f"   - Volume: {order_call_args.get('volume')}")
            print(f"   - Type: {order_call_args.get('type')}")
            print(f"   - Price: {order_call_args.get('price')}")
            print(f"   - SL: {order_call_args.get('sl')}")
            print(f"   - TP: {order_call_args.get('tp')}")
            
            # Verify position was searched for
            mock_mt5.positions_get.assert_called_with(symbol=trader.config.SYMBOL)
            
            # Verify trailing stop was added with correct position ticket
            trader.risk_manager.add_trailing_stop.assert_called_once()
            trailing_call_args = trader.risk_manager.add_trailing_stop.call_args[0]
            
            print(f"🎯 Trailing stop call:")
            print(f"   - Position ticket: {trailing_call_args[0]}")
            print(f"   - Entry price: {trailing_call_args[1]}")
            print(f"   - Initial SL: {trailing_call_args[2]}")
            print(f"   - Is buy: {trailing_call_args[3]}")
            
            # Verify the position ticket is correct (not the order ID)
            self.assertEqual(trailing_call_args[0], 67890)  # Position ticket
            self.assertNotEqual(trailing_call_args[0], 12345)  # Not order ID
            
            print(f"✅ Position ticket verification: PASSED")
            print(f"   - Expected: 67890 (position ticket)")
            print(f"   - Actual: {trailing_call_args[0]}")
            print(f"   - Order ID was: 12345 (correctly different)")
    
    @patch('live_trader_clean.mt5')
    def test_trailing_stop_position_not_found(self, mock_mt5):
        """Test behavior when position is not found after order execution."""
        print("\n🧪 Testing Trailing Stop - Position Not Found Scenario")
        
        # Create trader instance
        trader = MT5LiveTrader()
        
        # Mock MT5 initialization
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        mock_mt5.account_info.return_value = Mock()
        
        # Mock symbol info
        mock_symbol_info = Mock()
        mock_symbol_info.point = 0.01
        mock_symbol_info.trade_stops_level = 10
        mock_symbol_info.trade_freeze_level = 5
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        # Mock order send result
        mock_order_result = Mock()
        mock_order_result.retcode = 10009  # TRADE_RETCODE_DONE
        mock_order_result.order = 12345
        mock_order_result.comment = "Order executed successfully"
        mock_mt5.order_send.return_value = mock_order_result
        
        # Mock positions_get to return empty list (position not found)
        mock_mt5.positions_get.return_value = []
        
        # Mock the risk manager's add_trailing_stop method
        trader.risk_manager.add_trailing_stop = Mock()
        
        # Create mock signal data
        signal_data = {
            'signal': 1,
            'confidence': 0.75,
            'ensemble_score': 0.8
        }
        
        # Create mock market data
        import pandas as pd
        mock_df = pd.DataFrame({
            'close': [2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
            'high': [2005.0, 2006.0, 2007.0, 2008.0, 2009.0],
            'low': [1995.0, 1996.0, 1997.0, 1998.0, 1999.0],
            'open': [1998.0, 1999.0, 2000.0, 2001.0, 2002.0]
        })
        
        # Mock spread check to pass
        with patch('live_trader_clean.is_spread_ok_dynamic') as mock_spread:
            mock_spread.return_value = (True, 0.2, 0.3)
            
            # Mock ATR calculation
            trader.risk_manager.get_atr = Mock(return_value=2.0)
            
            # Mock get_current_tick to return None (use last close price)
            trader.data_manager.get_current_tick = Mock(return_value=None)
            
            # Mock MT5 availability to ensure real order execution
            with patch('live_trader_clean.MT5_AVAILABLE', True):
                trader.mt5_connected = True
                
                # Execute the trade
                result = trader._execute_trade(signal_data, mock_df)
            
            print(f"✅ Trade execution result: {result}")
            
            # Verify order was sent
            mock_mt5.order_send.assert_called_once()
            
            # Verify position was searched for
            mock_mt5.positions_get.assert_called_with(symbol=trader.config.SYMBOL)
            
            # Verify trailing stop was NOT added (position not found)
            trader.risk_manager.add_trailing_stop.assert_not_called()
            
            print(f"✅ Trailing stop correctly NOT added when position not found")
    
    @patch('live_trader_clean.mt5')
    def test_trailing_stop_multiple_positions(self, mock_mt5):
        """Test trailing stop with multiple positions to ensure correct matching."""
        print("\n🧪 Testing Trailing Stop - Multiple Positions Scenario")
        
        # Create trader instance
        trader = MT5LiveTrader()
        
        # Mock MT5 initialization
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        mock_mt5.account_info.return_value = Mock()
        
        # Mock symbol info
        mock_symbol_info = Mock()
        mock_symbol_info.point = 0.01
        mock_symbol_info.trade_stops_level = 10
        mock_symbol_info.trade_freeze_level = 5
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        # Mock order send result
        mock_order_result = Mock()
        mock_order_result.retcode = 10009
        mock_order_result.order = 12345
        mock_order_result.comment = "Order executed successfully"
        mock_mt5.order_send.return_value = mock_order_result
        
        # Mock multiple positions
        mock_position1 = Mock()
        mock_position1.ticket = 11111
        mock_position1.volume = 0.05  # Different volume
        mock_position1.price_open = 1995.0  # Different price
        
        mock_position2 = Mock()
        mock_position2.ticket = 67890  # This should match
        mock_position2.volume = 0.1  # Matches order volume
        mock_position2.price_open = 2000.0  # Matches order price
        
        mock_position3 = Mock()
        mock_position3.ticket = 22222
        mock_position3.volume = 0.2  # Different volume
        mock_position3.price_open = 2005.0  # Different price
        
        mock_mt5.positions_get.return_value = [mock_position1, mock_position2, mock_position3]
        
        # Mock the risk manager's add_trailing_stop method
        trader.risk_manager.add_trailing_stop = Mock()
        
        # Create mock signal data
        signal_data = {
            'signal': 1,
            'confidence': 0.75,
            'ensemble_score': 0.8
        }
        
        # Create mock market data
        import pandas as pd
        mock_df = pd.DataFrame({
            'close': [2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
            'high': [2005.0, 2006.0, 2007.0, 2008.0, 2009.0],
            'low': [1995.0, 1996.0, 1997.0, 1998.0, 1999.0],
            'open': [1998.0, 1999.0, 2000.0, 2001.0, 2002.0]
        })
        
        # Mock spread check to pass
        with patch('live_trader_clean.is_spread_ok_dynamic') as mock_spread:
            mock_spread.return_value = (True, 0.2, 0.3)
            
            # Mock ATR calculation
            trader.risk_manager.get_atr = Mock(return_value=2.0)
            
            # Mock get_current_tick to return None (use last close price)
            trader.data_manager.get_current_tick = Mock(return_value=None)
            
            # Mock MT5 availability to ensure real order execution
            with patch('live_trader_clean.MT5_AVAILABLE', True):
                trader.mt5_connected = True
                
                # Execute the trade
                result = trader._execute_trade(signal_data, mock_df)
            
            print(f"✅ Trade execution result: {result}")
            
            # Verify trailing stop was added with correct position ticket
            trader.risk_manager.add_trailing_stop.assert_called_once()
            trailing_call_args = trader.risk_manager.add_trailing_stop.call_args[0]
            
            print(f"🎯 Trailing stop call:")
            print(f"   - Position ticket: {trailing_call_args[0]}")
            print(f"   - Expected: 67890 (matching position)")
            
            # Verify the correct position was selected
            self.assertEqual(trailing_call_args[0], 67890)  # Should match position2
            
            print(f"✅ Correct position selected from multiple positions")

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    print("🚀 Starting Trailing Integration Tests (No Real Orders)")
    print("=" * 60)
    
    # Run the tests
    unittest.main(verbosity=2)
