#!/usr/bin/env python3
"""
MR BEN - STEP5 Simple Test
Quick verification that Risk Management is working
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_step5():
    """Simple test for STEP5"""
    print("🚀 MR BEN - STEP5 Simple Test")
    print("=" * 40)

    try:
        # Test 1: Configuration
        from core.configx import load_config

        cfg = load_config()
        print("✅ Configuration loaded")

        # Test 2: Risk Management config
        assert hasattr(cfg, 'risk_management')
        assert hasattr(cfg.risk_management, 'gates')
        print("✅ Risk management config present")

        # Test 3: Import risk gates
        from core.risk_gates import RiskManager

        print("✅ Risk gates imported")

        # Test 4: Import position sizing
        from core.position_sizing import PositionSizer

        print("✅ Position sizing imported")

        # Test 5: Initialize components
        risk_manager = RiskManager(cfg.risk_management)
        position_sizer = PositionSizer(cfg)
        print("✅ Components initialized")

        # Test 6: Basic functionality
        symbol = "EURUSD"
        bid = 1.1000
        ask = 1.1002
        position_size = 1.0
        signal_direction = 1
        current_positions = {}
        account_balance = 10000.0

        # Test risk gates
        allowed, responses = risk_manager.evaluate_all_gates(
            symbol, bid, ask, position_size, signal_direction, current_positions, account_balance
        )

        assert isinstance(allowed, bool)
        assert isinstance(responses, list)
        print("✅ Risk gates working")

        # Test position sizing
        position_info = position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            account_balance=account_balance,
            confidence=0.75,
            atr_value=0.0030,
        )

        assert position_info.size > 0
        print("✅ Position sizing working")

        print("\n🎉 STEP5: Risk Management - COMPLETED SUCCESSFULLY!")
        print("All components are working correctly.")
        return True

    except Exception as e:
        print(f"❌ STEP5 test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_step5()
    sys.exit(0 if success else 1)
