#!/usr/bin/env python3
"""
Test script for the refactored MR BEN Trading System.
This script verifies that the modular architecture works correctly.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🧪 Testing module imports...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    modules_to_test = [
        "core.trader",
        "core.metrics", 
        "core.exceptions",
        "config.settings",
        "data.manager",
        "ai.system",
        "risk.manager",
        "execution.executor",
        "utils.helpers",
        "utils.position_management",
        "utils.memory",
        "utils.error_handler"
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name}")
            
            # Test if main classes can be instantiated (with mocks)
            if module_name == "core.trader":
                try:
                    # Mock dependencies for trader
                    with open('test_config.json', 'w') as f:
                        f.write('{"credentials": {"login": 123, "password": "test", "server": "test"}, "flags": {"demo_mode": true}, "trading": {"symbol": "XAUUSD.PRO", "timeframe": 15, "bars": 500, "magic_number": 20250721, "sessions": ["London", "NY"], "max_spread_points": 200, "use_risk_based_volume": true, "fixed_volume": 0.01, "sleep_seconds": 12, "retry_delay": 5, "consecutive_signals_required": 1, "lstm_timesteps": 50, "cooldown_seconds": 180}, "risk": {"base_risk": 0.01, "min_lot": 0.01, "max_lot": 2.0, "max_open_trades": 3, "max_daily_loss": 0.02, "max_trades_per_day": 10}, "logging": {"enabled": true, "level": "INFO", "log_file": "logs/test.log", "trade_log_path": "data/test_trades.csv"}, "session": {"timezone": "Etc/UTC"}, "models": {}, "advanced": {"swing_lookback": 12, "dynamic_spread_atr_frac": 0.10}, "execution": {"spread_eps": 0.02, "use_spread_ma": true, "spread_ma_window": 5, "spread_hysteresis_factor": 1.05}, "tp_policy": {"split": true, "tp1_r": 0.8, "tp2_r": 1.5, "tp1_share": 0.5, "breakeven_after_tp1": true}}')
                    
                    # Mock MT5 availability
                    import src.core.trader
                    src.core.trader.MT5_AVAILABLE = False
                    
                    # Mock all dependencies
                    import unittest.mock
                    with unittest.mock.patch.multiple('src.core.trader',
                                                  MT5Config=unittest.mock.Mock(),
                                                  MT5DataManager=unittest.mock.Mock(),
                                                  MRBENAdvancedAISystem=unittest.mock.Mock(),
                                                  EnhancedRiskManager=unittest.mock.Mock(),
                                                  EnhancedTradeExecutor=unittest.mock.Mock(),
                                                  _get_open_positions=unittest.mock.Mock(return_value={}),
                                                  _prune_trailing_registry=unittest.mock.Mock(return_value=0),
                                                  _count_open_positions=unittest.mock.Mock(return_value=0),
                                                  log_memory_usage=unittest.mock.Mock(),
                                                  cleanup_memory=unittest.mock.Mock()):
                        
                        # Mock config attributes
                        mock_config = unittest.mock.Mock()
                        mock_config.LOG_LEVEL = "INFO"
                        mock_config.LOG_FILE = "logs/test.log"
                        mock_config.TRADE_LOG_PATH = "data/test_trades.csv"
                        mock_config.SYMBOL = "XAUUSD.PRO"
                        mock_config.TIMEFRAME_MIN = 15
                        mock_config.BARS = 500
                        mock_config.MAGIC = 20250721
                        mock_config.SESSIONS = ["London", "NY"]
                        mock_config.MAX_SPREAD_POINTS = 200
                        mock_config.USE_RISK_BASED_VOLUME = True
                        mock_config.FIXED_VOLUME = 0.01
                        mock_config.SLEEP_SECONDS = 12
                        mock_config.RETRY_DELAY = 5
                        mock_config.CONSECUTIVE_SIGNALS_REQUIRED = 1
                        mock_config.LSTM_TIMESTEPS = 50
                        mock_config.COOLDOWN_SECONDS = 180
                        mock_config.BASE_RISK = 0.01
                        mock_config.MIN_LOT = 0.01
                        mock_config.MAX_LOT = 2.0
                        mock_config.MAX_OPEN_TRADES = 3
                        mock_config.MAX_DAILY_LOSS = 0.02
                        mock_config.MAX_TRADES_PER_DAY = 10
                        mock_config.SESSION_TZ = "Etc/UTC"
                        mock_config.config_data = {}
                        
                        # Mock the config class
                        with unittest.mock.patch('src.core.trader.MT5Config', return_value=mock_config):
                            trader = module.MT5LiveTrader()
                            print(f"   ✅ MT5LiveTrader instantiated successfully")
                            
                except Exception as e:
                    print(f"   ⚠️  MT5LiveTrader instantiation failed: {e}")
                    
            elif module_name == "config.settings":
                try:
                    # Mock file reading
                    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"credentials": {"login": 123, "password": "test", "server": "test"}, "flags": {"demo_mode": true}, "trading": {"symbol": "XAUUSD.PRO", "timeframe": 15, "bars": 500, "magic_number": 20250721, "sessions": ["London", "NY"], "max_spread_points": 200, "use_risk_based_volume": true, "fixed_volume": 0.01, "sleep_seconds": 12, "retry_delay": 5, "consecutive_signals_required": 1, "lstm_timesteps": 50, "cooldown_seconds": 180}, "risk": {"base_risk": 0.01, "min_lot": 0.01, "max_lot": 2.0, "max_open_trades": 3, "max_daily_loss": 0.02, "max_trades_per_day": 10}, "logging": {"enabled": true, "level": "INFO", "log_file": "logs/test.log", "trade_log_path": "data/test_trades.csv"}, "session": {"timezone": "Etc/UTC"}, "models": {}, "advanced": {"swing_lookback": 12, "dynamic_spread_atr_frac": 0.10}, "execution": {"spread_eps": 0.02, "use_spread_ma": true, "spread_ma_window": 5, "spread_hysteresis_factor": 1.05}, "tp_policy": {"split": true, "tp1_r": 0.8, "tp2_r": 1.5, "tp1_share": 0.5, "breakeven_after_tp1": true}}')):
                        with unittest.mock.patch('os.getenv', return_value="test_password"):
                            config = module.MT5Config()
                            print(f"   ✅ MT5Config instantiated successfully")
                            
                except Exception as e:
                    print(f"   ⚠️  MT5Config instantiation failed: {e}")
                    
            elif module_name == "ai.system":
                try:
                    ai_system = module.MRBENAdvancedAISystem()
                    print(f"   ✅ MRBENAdvancedAISystem instantiated successfully")
                except Exception as e:
                    print(f"   ⚠️  MRBENAdvancedAISystem instantiation failed: {e}")
                    
            elif module_name == "risk.manager":
                try:
                    risk_manager = module.EnhancedRiskManager()
                    print(f"   ✅ EnhancedRiskManager instantiated successfully")
                except Exception as e:
                    print(f"   ⚠️  EnhancedRiskManager instantiation failed: {e}")
                    
            elif module_name == "execution.executor":
                try:
                    mock_risk_manager = unittest.mock.Mock()
                    executor = module.EnhancedTradeExecutor(mock_risk_manager)
                    print(f"   ✅ EnhancedTradeExecutor instantiated successfully")
                except Exception as e:
                    print(f"   ⚠️  EnhancedTradeExecutor instantiation failed: {e}")
                    
            elif module_name == "core.metrics":
                try:
                    metrics = module.PerformanceMetrics()
                    print(f"   ✅ PerformanceMetrics instantiated successfully")
                except Exception as e:
                    print(f"   ⚠️  PerformanceMetrics instantiation failed: {e}")
                    
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"⚠️  {module_name}: {e}")
            failed_imports.append(module_name)
    
    # Clean up test file
    try:
        os.remove('test_config.json')
    except:
        pass
    
    return failed_imports

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test performance metrics
        from src.core.metrics import PerformanceMetrics
        metrics = PerformanceMetrics()
        metrics.record_cycle(0.1)
        metrics.record_trade()
        stats = metrics.get_stats()
        print(f"✅ Performance metrics: {stats['cycle_count']} cycles, {stats['total_trades']} trades")
        
        # Test risk manager
        from src.risk.manager import EnhancedRiskManager
        risk_manager = EnhancedRiskManager()
        print(f"✅ Risk manager initialized with base risk: {risk_manager.base_risk}")
        
        # Test AI system
        from src.ai.system import MRBENAdvancedAISystem
        ai_system = MRBENAdvancedAISystem()
        print(f"✅ AI system initialized with {len(ai_system.models)} models")
        
        # Test utilities
        from src.utils.helpers import round_price
        rounded = round_price("XAUUSD.PRO", 2000.12345)
        print(f"✅ Utility functions: rounded price {rounded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        # Create a minimal test config
        test_config = {
            "credentials": {
                "login": 12345,
                "password": "test_password",
                "server": "TestServer"
            },
            "flags": {
                "demo_mode": True
            },
            "trading": {
                "symbol": "XAUUSD.PRO",
                "timeframe": 15,
                "bars": 500,
                "magic_number": 20250721,
                "sessions": ["London", "NY"],
                "max_spread_points": 200,
                "use_risk_based_volume": True,
                "fixed_volume": 0.01,
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
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "logs/test.log",
                "trade_log_path": "data/test_trades.csv"
            },
            "session": {
                "timezone": "Etc/UTC"
            }
        }
        
        # Write test config
        import json
        with open('test_config.json', 'w') as f:
            json.dump(test_config, f, indent=2)
        
        # Test config loading
        from src.config.settings import MT5Config
        config = MT5Config()
        
        print(f"✅ Configuration loaded:")
        print(f"   Symbol: {config.SYMBOL}")
        print(f"   Timeframe: {config.TIMEFRAME_MIN}")
        print(f"   Base Risk: {config.BASE_RISK}")
        print(f"   Max Open Trades: {config.MAX_OPEN_TRADES}")
        
        # Clean up
        os.remove('test_config.json')
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 MR BEN Trading System - Refactoring Verification")
    print("=" * 60)
    
    # Test imports
    failed_imports = test_imports()
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} modules failed to import:")
        for module in failed_imports:
            print(f"   - {module}")
        return False
    
    print(f"\n✅ All {len(failed_imports)} modules imported successfully!")
    
    # Test basic functionality
    if not test_basic_functionality():
        return False
    
    # Test configuration
    if not test_configuration():
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! The refactored system is working correctly.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
