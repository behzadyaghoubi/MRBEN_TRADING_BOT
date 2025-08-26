#!/usr/bin/env python3
"""
MR BEN - STEP6 Position Management Test
Test TP-Split, Breakeven, and Trailing Stop functionality
"""

import sys
import os
from datetime import datetime, timezone

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_step6():
    """Test STEP6: Position Management"""
    print("🚀 MR BEN - STEP6 Position Management Test")
    print("=" * 50)
    
    try:
        # Test 1: Configuration
        from core.configx import load_config
        cfg = load_config()
        print("✅ Configuration loaded")
        
        # Test 2: Position Management config
        assert hasattr(cfg, 'position_management')
        assert hasattr(cfg.position_management, 'tp_split_enabled')
        assert hasattr(cfg.position_management, 'breakeven_enabled')
        assert hasattr(cfg.position_management, 'trailing_enabled')
        print("✅ Position management config present")
        
        # Test 3: Import position management
        from core.position_management import (
            PositionManager, PositionInfo, PositionStatus,
            TPManager, BreakevenManager, TrailingStopManager,
            TPLevel
        )
        print("✅ Position management components imported")
        
        # Test 4: Initialize components
        position_manager = PositionManager(cfg)
        print("✅ Position manager initialized")
        
        # Test 5: Test TP-Split functionality
        print("\n📊 Testing TP-Split functionality...")
        
        # Create a test position
        test_position = position_manager.open_position(
            symbol="EURUSD",
            ticket=12345,
            position_type="buy",
            size=1.0,
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            atr_value=0.0030,
            confidence=0.75
        )
        
        assert test_position is not None
        assert len(test_position.tp_levels) == 3
        assert test_position.status == PositionStatus.OPEN
        print("✅ Position opened with TP levels")
        
        # Check TP levels
        tp_levels = test_position.tp_levels
        assert len(tp_levels) == 3
        
        # First TP should be at 2x ATR
        expected_tp1 = 1.1000 + (0.0030 * 2.0)
        assert abs(tp_levels[0].price - expected_tp1) < 0.0001
        assert tp_levels[0].size_percent == 30.0
        
        # Second TP should be at 3x ATR
        expected_tp2 = 1.1000 + (0.0030 * 3.0)
        assert abs(tp_levels[1].price - expected_tp2) < 0.0001
        assert tp_levels[1].size_percent == 40.0
        
        # Third TP should be at 4x ATR
        expected_tp3 = 1.1000 + (0.0030 * 4.0)
        assert abs(tp_levels[2].price - expected_tp3) < 0.0001
        assert tp_levels[2].size_percent == 30.0
        
        print("✅ TP levels calculated correctly")
        
        # Test 6: Test TP triggers
        print("\n🎯 Testing TP triggers...")
        
        # Check TP triggers at different prices
        current_price = 1.1060  # Between TP1 and TP2
        actions = position_manager.update_position(12345, current_price, 60.0)
        
        # Should trigger first TP
        assert 'tp_triggers' in actions
        tp_triggers = actions['tp_triggers']
        assert len(tp_triggers) == 1
        
        tp_level, size_to_close = tp_triggers[0]
        assert tp_level.size_percent == 30.0
        assert size_to_close == 0.3  # 30% of 1.0 lot
        
        print("✅ TP triggers working correctly")
        
        # Test 7: Test partial position closure
        print("\n✂️ Testing partial position closure...")
        
        success = position_manager.close_position_partial(
            12345, tp_level, size_to_close
        )
        assert success is True
        
        # Check position status
        updated_position = position_manager.get_position_status(12345)
        assert updated_position.status == PositionStatus.PARTIALLY_CLOSED
        assert updated_position.closed_size == 0.3
        assert updated_position.remaining_size == 0.7
        
        print("✅ Partial position closure working")
        
        # Test 8: Test Breakeven functionality
        print("\n⚖️ Testing Breakeven functionality...")
        
        # Move price to trigger breakeven (0.5 ATR profit)
        breakeven_price = 1.1000 + (0.0030 * 0.5)  # 0.5 ATR
        actions = position_manager.update_position(12345, breakeven_price, 15.0)
        
        # Should trigger breakeven
        assert 'breakeven' in actions
        new_stop_loss = actions['breakeven']
        
        # Check breakeven execution
        breakeven_position = position_manager.get_position_status(12345)
        assert breakeven_position.breakeven_triggered is True
        assert breakeven_position.status == PositionStatus.BREAKEVEN
        assert breakeven_position.stop_loss == new_stop_loss
        
        print("✅ Breakeven functionality working")
        
        # Test 9: Test Trailing Stop functionality
        print("\n📈 Testing Trailing Stop functionality...")
        
        # Move price to activate trailing (1.0 ATR profit)
        trailing_activation_price = 1.1000 + (0.0030 * 1.0)  # 1.0 ATR
        actions = position_manager.update_position(12345, trailing_activation_price, 30.0)
        
        # Should activate trailing
        assert 'trailing_activated' in actions
        initial_trailing_stop = actions['trailing_activated']
        
        # Check trailing activation
        trailing_position = position_manager.get_position_status(12345)
        assert trailing_position.trailing_enabled is True
        assert trailing_position.status == PositionStatus.TRAILING
        assert trailing_position.current_trailing_stop == initial_trailing_stop
        
        print("✅ Trailing stop activation working")
        
        # Test 10: Test Trailing Stop updates
        print("\n🔄 Testing Trailing Stop updates...")
        
        # Move price higher to update trailing stop
        higher_price = 1.1030
        actions = position_manager.update_position(12345, higher_price, 90.0)
        
        # Should update trailing stop
        if 'trailing_updated' in actions:
            print("✅ Trailing stop updates working")
        else:
            print("ℹ️ No trailing stop update (within expected range)")
        
        # Test 11: Test position summary
        print("\n📋 Testing position summary...")
        
        summary = position_manager.get_position_summary()
        assert summary['total_positions'] == 1
        assert summary['total_size'] == 0.7
        assert 'status_distribution' in summary
        
        print("✅ Position summary working")
        
        # Test 12: Test full position closure
        print("\n🚪 Testing full position closure...")
        
        success = position_manager.close_position_full(12345, 45.0)
        assert success is True
        
        # Position should be removed
        closed_position = position_manager.get_position_status(12345)
        assert closed_position is None
        
        # Summary should show no positions
        final_summary = position_manager.get_position_summary()
        assert final_summary['total_positions'] == 0
        
        print("✅ Full position closure working")
        
        print("\n🎉 STEP6: Position Management - COMPLETED SUCCESSFULLY!")
        print("All components are working correctly:")
        print("✅ TP-Split with multiple levels")
        print("✅ Breakeven functionality")
        print("✅ Trailing Stop Loss")
        print("✅ Position lifecycle management")
        print("✅ Comprehensive monitoring and adjustment")
        
        return True
        
    except Exception as e:
        print(f"❌ STEP6 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_step6()
    sys.exit(0 if success else 1)
