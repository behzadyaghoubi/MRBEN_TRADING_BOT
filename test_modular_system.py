#!/usr/bin/env python3
"""
Test script for MR BEN Modular Trading System
Tests the basic functionality of modular components
"""

import os
import sys
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_config_module():
    """Test configuration module functionality"""
    print("🧪 Testing Configuration Module...")

    try:
        from config import MT5Config

        # Test with mock config data
        mock_config_data = {
            "credentials": {
                "login": "test_login",
                "password": "test_password",
                "server": "test_server",
            },
            "flags": {"demo_mode": True},
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
                "cooldown_seconds": 180,
            },
            "risk": {
                "base_risk": 0.01,
                "min_lot": 0.01,
                "max_lot": 2.0,
                "max_open_trades": 3,
                "max_daily_loss": 0.02,
                "max_trades_per_day": 10,
                "sl_atr_multiplier": 1.6,
                "tp_atr_multiplier": 2.2,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "logs/test.log",
                "trade_log_path": "data/test_trades.csv",
            },
            "session": {"timezone": "Etc/UTC"},
            "advanced": {
                "swing_lookback": 12,
                "dynamic_spread_atr_frac": 0.10,
                "deviation_multiplier": 1.5,
                "inbar_eval_seconds": 10,
                "inbar_min_conf": 0.66,
                "inbar_min_score": 0.12,
                "inbar_min_struct_buffer_atr": 0.8,
                "startup_warmup_seconds": 90,
                "startup_min_conf": 0.62,
                "startup_min_score": 0.10,
                "reentry_window_seconds": 90,
            },
            "execution": {
                "spread_eps": 0.02,
                "use_spread_ma": True,
                "spread_ma_window": 5,
                "spread_hysteresis_factor": 1.05,
            },
            "tp_policy": {
                "split": True,
                "tp1_r": 0.8,
                "tp2_r": 1.5,
                "tp1_share": 0.5,
                "breakeven_after_tp1": True,
            },
            "conformal": {
                "enabled": True,
                "soft_gate": True,
                "emergency_bypass": False,
                "min_p": 0.10,
                "hard_floor": 0.05,
                "penalty_small": 0.05,
                "penalty_big": 0.10,
                "extra_consecutive": 1,
                "treat_zero_as_block": True,
                "max_conf_bump_floor": 0.05,
                "extra_consec_floor": 2,
                "cap_final_thr": 0.90,
            },
        }

        # Mock file reading
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = str(mock_config_data)
            mock_open.return_value.__enter__.return_value.__iter__ = lambda x: iter([])

            # Mock json.loads
            with patch('json.loads', return_value=mock_config_data):
                config = MT5Config()

                # Test basic attributes
                assert config.SYMBOL == "XAUUSD.PRO"
                assert config.TIMEFRAME_MIN == 15
                assert config.BARS == 500
                assert config.MAGIC == 20250721
                assert config.DEMO_MODE == True

                # Test structured config access
                assert config.trading.symbol == "XAUUSD.PRO"
                assert config.risk.base_risk == 0.01
                assert config.logging.level == "INFO"

                # Test config summary
                summary = config.get_config_summary()
                assert summary["symbol"] == "XAUUSD.PRO"
                assert summary["demo_mode"] == True

                print("✅ Configuration module test passed")
                return True

    except Exception as e:
        print(f"❌ Configuration module test failed: {e}")
        return False


def test_telemetry_module():
    """Test telemetry module functionality"""
    print("🧪 Testing Telemetry Module...")

    try:
        from telemetry import EventLogger, MemoryMonitor, MFELogger, PerformanceMetrics

        # Test EventLogger
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.write = Mock()
            logger = EventLogger("test_events.jsonl", "test_run", "XAUUSD.PRO")
            logger.emit("test_event", test_data="test_value")
            logger.close()

        # Test MFELogger
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.write = Mock()
            mfe_logger = MFELogger("test_mfe.jsonl")
            mfe_logger.log_tick("test_run", 123, 1800.0, 1795.0, 1790.0, 1810.0)
            mfe_logger.close()

        # Test PerformanceMetrics
        metrics = PerformanceMetrics()
        metrics.record_cycle(0.1)
        metrics.record_trade()
        metrics.record_error()

        stats = metrics.get_stats()
        assert stats["cycle_count"] == 1
        assert stats["total_trades"] == 1
        assert stats["error_count"] == 1

        # Test MemoryMonitor
        memory_monitor = MemoryMonitor()
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
            memory_mb = memory_monitor.check_memory(force=True)
            assert memory_mb == 100.0

        print("✅ Telemetry module test passed")
        return True

    except Exception as e:
        print(f"❌ Telemetry module test failed: {e}")
        return False


def test_trading_system_module():
    """Test trading system module functionality"""
    print("🧪 Testing Trading System Module...")

    try:
        from trading_system import TradingState, TradingSystem

        # Test TradingState
        state = TradingState("test_run")
        assert state.run_id == "test_run"
        assert state.running == False
        assert state.consecutive_signals == 0

        # Test TradingSystem (with mocked dependencies)
        with patch('config.MT5Config') as mock_config_class:
            mock_config = Mock()
            mock_config.BASE_RISK = 0.01
            mock_config.MIN_LOT = 0.01
            mock_config.MAX_LOT = 2.0
            mock_config.MAX_OPEN_TRADES = 3
            mock_config.TIMEFRAME_MIN = 15
            mock_config.SYMBOL = "XAUUSD.PRO"
            mock_config.SESSION_TZ = "Etc/UTC"
            mock_config.config_data = {}

            mock_config_class.return_value = mock_config

            # Mock all component dependencies
            with patch.multiple(
                'trading_system',
                EnhancedRiskManager=Mock,
                EnhancedTradeExecutor=Mock,
                MT5DataManager=Mock,
                MRBENAdvancedAISystem=Mock,
                EventLogger=Mock,
                MFELogger=Mock,
            ):

                trading_system = TradingSystem(mock_config)

                # Test basic initialization
                assert trading_system.config == mock_config
                assert trading_system.state.run_id is not None
                assert hasattr(trading_system, 'trailing_registry')

                # Test can_trade method
                can_trade, reason = trading_system.can_trade()
                assert isinstance(can_trade, bool)
                assert isinstance(reason, str)

                # Test generate_signal method
                mock_df = Mock()
                mock_df.__getitem__.return_value.iloc.__getitem__.return_value = 1800.0
                mock_df.__getitem__.return_value.iloc.__getitem__.return_value = 100

                signal = trading_system.generate_signal(mock_df)
                assert isinstance(signal, dict)
                assert 'signal' in signal
                assert 'confidence' in signal

                print("✅ Trading system module test passed")
                return True

    except Exception as e:
        print(f"❌ Trading system module test failed: {e}")
        return False


def test_trading_loop_module():
    """Test trading loop module functionality"""
    print("🧪 Testing Trading Loop Module...")

    try:
        from trading_loop import TradingLoopManager
        from trading_system import TradingSystem

        # Mock dependencies
        with patch('config.MT5Config') as mock_config_class:
            mock_config = Mock()
            mock_config.RETRY_DELAY = 1
            mock_config.SLEEP_SECONDS = 1
            mock_config.BARS = 500
            mock_config.TIMEFRAME_MIN = 15
            mock_config.MAX_OPEN_TRADES = 3
            mock_config.CONSECUTIVE_SIGNALS_REQUIRED = 1
            mock_config_class.return_value = mock_config

            with patch.multiple(
                'trading_system',
                EnhancedRiskManager=Mock,
                EnhancedTradeExecutor=Mock,
                MT5DataManager=Mock,
                MRBENAdvancedAISystem=Mock,
                EventLogger=Mock,
                MFELogger=Mock,
            ):

                trading_system = TradingSystem(mock_config)
                loop_manager = TradingLoopManager(trading_system)

                # Test initialization
                assert loop_manager.trading_system == trading_system
                assert loop_manager.config == mock_config
                assert loop_manager.running == False
                assert loop_manager.cycle == 0

                # Test start/stop
                loop_manager.start()
                assert loop_manager.running == True

                # Give it a moment to start
                import time

                time.sleep(0.1)

                loop_manager.stop()
                assert loop_manager.running == False

                # Test status
                status = loop_manager.get_status()
                assert isinstance(status, dict)
                assert 'running' in status
                assert 'cycle' in status

                print("✅ Trading loop module test passed")
                return True

    except Exception as e:
        print(f"❌ Trading loop module test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting MR BEN Modular System Tests")
    print("=" * 60)

    tests = [
        test_config_module,
        test_telemetry_module,
        test_trading_system_module,
        test_trading_loop_module,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Modular system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
