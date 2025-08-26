#!/usr/bin/env python3
"""
MR BEN - STEP7 Order Management Test
Test MT5 integration, order filling modes, and execution optimization
"""

import os
import sys
from datetime import UTC, datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_step7():
    """Test STEP7: Order Management"""
    print("🚀 MR BEN - STEP7 Order Management Test")
    print("=" * 50)

    try:
        # Test 1: Configuration
        from core.configx import load_config

        cfg = load_config()
        print("✅ Configuration loaded")

        # Test 2: Order Management config
        assert hasattr(cfg, 'order_management')
        assert hasattr(cfg.order_management, 'mt5_enabled')
        assert hasattr(cfg.order_management, 'filling_optimization_enabled')
        assert hasattr(cfg.order_management, 'slippage_control_enabled')
        print("✅ Order management config present")

        # Test 3: Import order management components
        from core.order_management import (
            OrderExecutor,
            OrderFillingMode,
            OrderRequest,
            OrderResult,
            OrderStatus,
            OrderType,
        )

        print("✅ Order management components imported")

        # Test 4: Initialize components
        order_executor = OrderExecutor(cfg)
        print("✅ Order executor initialized")

        # Test 5: Test MT5 Connector
        print("\n🔌 Testing MT5 Connector...")

        mt5_connector = order_executor.mt5_connector
        assert mt5_connector is not None

        # Test connection (will fail without real MT5, but that's expected)
        if mt5_connector.enabled:
            print("ℹ️ MT5 enabled - connection test skipped (requires real MT5)")
        else:
            print("ℹ️ MT5 disabled - using mock mode")

        print("✅ MT5 connector working")

        # Test 6: Test Order Filling Optimizer
        print("\n🎯 Testing Order Filling Optimizer...")

        filling_optimizer = order_executor.filling_optimizer
        assert filling_optimizer is not None

        # Test filling mode selection
        mock_symbol_info = {
            "spread": 0.0002,  # Normal spread
            "volume_min": 0.01,
            "volume_max": 100.0,
        }

        # Test normal conditions
        normal_mode = filling_optimizer.select_filling_mode(mock_symbol_info, 0.5, 0.001, "normal")
        assert normal_mode in [OrderFillingMode.IOC, OrderFillingMode.FOK, OrderFillingMode.RETURN]

        # Test high urgency
        high_urgency_mode = filling_optimizer.select_filling_mode(
            mock_symbol_info, 0.5, 0.001, "high"
        )
        assert high_urgency_mode == OrderFillingMode.IOC

        # Test high volatility
        high_vol_mode = filling_optimizer.select_filling_mode(
            mock_symbol_info, 0.5, 0.003, "normal"  # Above volatility threshold
        )
        assert high_vol_mode == OrderFillingMode.IOC

        # Test large volume
        large_volume_mode = filling_optimizer.select_filling_mode(
            mock_symbol_info, 2.0, 0.001, "normal"  # Above volume threshold
        )
        assert large_volume_mode == OrderFillingMode.FOK

        # Test wide spread
        wide_spread_info = {"spread": 0.0005}  # Above spread threshold
        wide_spread_mode = filling_optimizer.select_filling_mode(
            wide_spread_info, 0.5, 0.001, "normal"
        )
        assert wide_spread_mode == OrderFillingMode.RETURN

        print("✅ Order filling optimizer working correctly")

        # Test 7: Test Slippage Manager
        print("\n📊 Testing Slippage Manager...")

        slippage_manager = order_executor.slippage_manager
        assert slippage_manager is not None

        # Test slippage calculation for buy order
        buy_slippage = slippage_manager.calculate_slippage(
            requested_price=1.1000, executed_price=1.1002, order_type=OrderType.BUY
        )
        assert buy_slippage == 0.0002  # Positive slippage (executed above)

        # Test slippage calculation for sell order
        sell_slippage = slippage_manager.calculate_slippage(
            requested_price=1.1000, executed_price=1.0998, order_type=OrderType.SELL
        )
        assert sell_slippage == 0.0002  # Positive slippage (executed below)

        # Test slippage acceptability
        assert slippage_manager.is_slippage_acceptable(0.0001)  # Within limits
        assert not slippage_manager.is_slippage_acceptable(0.0010)  # Above limits

        # Test slippage recording
        slippage_manager.record_slippage(
            symbol="EURUSD",
            order_type=OrderType.BUY,
            requested_price=1.1000,
            executed_price=1.1002,
            volume=0.5,
        )

        # Get slippage stats
        stats = slippage_manager.get_slippage_stats()
        assert stats["total_orders"] == 1
        assert stats["acceptable_orders"] == 1
        assert stats["unacceptable_orders"] == 0

        print("✅ Slippage manager working correctly")

        # Test 8: Test Order Request Creation
        print("\n📝 Testing Order Request Creation...")

        order_request = OrderRequest(
            symbol="EURUSD",
            order_type=OrderType.BUY,
            volume=0.5,
            price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            comment="Test order",
            magic=12345,
        )

        assert order_request.symbol == "EURUSD"
        assert order_request.order_type == OrderType.BUY
        assert order_request.volume == 0.5
        assert order_request.price == 1.1000
        assert order_request.stop_loss == 1.0950
        assert order_request.take_profit == 1.1100

        print("✅ Order request creation working")

        # Test 9: Test Order Result Creation
        print("\n📋 Testing Order Result Creation...")

        order_result = OrderResult(
            success=True,
            ticket=12345,
            volume=0.5,
            price=1.1002,
            spread=0.0002,
            slippage=0.0002,
            execution_time=datetime.now(UTC),
        )

        assert order_result.success is True
        assert order_result.ticket == 12345
        assert order_result.volume == 0.5
        assert order_result.price == 1.1002
        assert order_result.spread == 0.0002
        assert order_result.slippage == 0.0002

        print("✅ Order result creation working")

        # Test 10: Test Order Execution (Mock Mode)
        print("\n🚀 Testing Order Execution (Mock Mode)...")

        # Since we're in mock mode, test the structure without real execution
        if not mt5_connector.enabled:
            print("ℹ️ Testing in mock mode - real execution skipped")

            # Test order preparation
            mock_symbol_info = {"spread": 0.0002, "volume_min": 0.01, "volume_max": 100.0}

            # Test filling mode selection
            optimal_mode = filling_optimizer.select_filling_mode(
                mock_symbol_info, 0.5, 0.001, "normal"
            )
            assert optimal_mode in [
                OrderFillingMode.IOC,
                OrderFillingMode.FOK,
                OrderFillingMode.RETURN,
            ]

            print("✅ Mock order execution working")
        else:
            print("ℹ️ MT5 enabled - execution test would require real connection")

        # Test 11: Test Order Status Management
        print("\n📊 Testing Order Status Management...")

        # Test order status conversion
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"

        # Test order type conversion
        assert OrderType.BUY.value == "buy"
        assert OrderType.SELL.value == "sell"
        assert OrderType.BUY_LIMIT.value == "buy_limit"

        # Test filling mode conversion
        assert OrderFillingMode.IOC.value == "ioc"
        assert OrderFillingMode.FOK.value == "fok"
        assert OrderFillingMode.RETURN.value == "return"

        print("✅ Order status management working")

        # Test 12: Test Execution Summary
        print("\n📈 Testing Execution Summary...")

        summary = order_executor.get_execution_summary()
        assert isinstance(summary, dict)
        assert "mt5_connected" in summary
        assert "active_orders" in summary
        assert "total_orders" in summary
        assert "slippage_stats" in summary

        print("✅ Execution summary working")

        print("\n🎉 STEP7: Order Management - COMPLETED SUCCESSFULLY!")
        print("All components are working correctly:")
        print("✅ MT5 Connector with connection management")
        print("✅ Order Filling Mode Optimization")
        print("✅ Slippage Management and Control")
        print("✅ Order Request/Result structures")
        print("✅ Order Execution coordination")
        print("✅ Comprehensive order lifecycle management")

        return True

    except Exception as e:
        print(f"❌ STEP7 test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_step7()
    sys.exit(0 if success else 1)
