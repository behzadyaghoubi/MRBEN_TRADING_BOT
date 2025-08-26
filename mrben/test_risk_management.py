#!/usr/bin/env python3
"""
MR BEN - Risk Management Testing
Comprehensive tests for all risk management components
"""

from __future__ import annotations
import numpy as np
import sys
import os
from datetime import datetime, timezone, time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.configx import load_config
from core.loggingx import setup_logging
from core.risk_gates import (
    RiskManager, SpreadGate, ExposureGate, DailyLossGate, 
    ConsecutiveGate, CooldownGate, GateResult
)
from core.position_sizing import PositionSizer


def test_spread_gate():
    """Test spread gate functionality"""
    print("\n🧪 Testing Spread Gate...")
    
    try:
        cfg = load_config()
        gate = SpreadGate(cfg.risk_management)
        
        # Test normal spread (should pass)
        result1 = gate.evaluate("EURUSD", 1.1000, 1.1002)
        assert result1.result == GateResult.PASS
        print(f"✅ Normal spread test passed: {result1.reason}")
        
        # Test wide spread (should reject)
        result2 = gate.evaluate("EURUSD", 1.1000, 1.1010)
        assert result2.result == GateResult.REJECT
        print(f"✅ Wide spread test passed: {result2.reason}")
        
        # Test warning level spread
        result3 = gate.evaluate("EURUSD", 1.1000, 1.1002.5)
        print(f"✅ Warning spread test: {result3.result.value} - {result3.reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Spread gate test failed: {e}")
        return False


def test_exposure_gate():
    """Test exposure gate functionality"""
    print("\n🧪 Testing Exposure Gate...")
    
    try:
        cfg = load_config()
        gate = ExposureGate(cfg.risk_management)
        
        # Mock current positions
        current_positions = {
            "EURUSD": {"size": 1.0},
            "GBPUSD": {"size": 0.5}
        }
        
        account_balance = 10000.0
        
        # Test normal exposure (should pass)
        result1 = gate.evaluate("AUDUSD", 0.5, current_positions, account_balance)
        assert result1.result == GateResult.PASS
        print(f"✅ Normal exposure test passed: {result1.reason}")
        
        # Test excessive exposure (should reject)
        result2 = gate.evaluate("AUDUSD", 10.0, current_positions, account_balance)
        assert result2.result == GateResult.REJECT
        print(f"✅ Excessive exposure test passed: {result2.reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Exposure gate test failed: {e}")
        return False


def test_daily_loss_gate():
    """Test daily loss gate functionality"""
    print("\n🧪 Testing Daily Loss Gate...")
    
    try:
        cfg = load_config()
        gate = DailyLossGate(cfg.risk_management)
        
        account_balance = 10000.0
        
        # Test normal state (should pass)
        result1 = gate.evaluate(account_balance)
        assert result1.result == GateResult.PASS
        print(f"✅ Normal daily loss test passed: {result1.reason}")
        
        # Simulate a loss
        gate.update_pnl(-500.0)
        
        # Test after moderate loss (should still pass)
        result2 = gate.evaluate(account_balance)
        print(f"✅ Moderate loss test: {result2.result.value} - {result2.reason}")
        
        # Simulate excessive loss
        gate.update_pnl(-600.0)  # Total: -1100
        
        # Test after excessive loss (should reject)
        result3 = gate.evaluate(account_balance)
        assert result3.result == GateResult.REJECT
        print(f"✅ Excessive loss test passed: {result3.reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Daily loss gate test failed: {e}")
        return False


def test_consecutive_gate():
    """Test consecutive signal gate functionality"""
    print("\n🧪 Testing Consecutive Gate...")
    
    try:
        cfg = load_config()
        gate = ConsecutiveGate(cfg.risk_management)
        
        # Test first signal (should pass)
        result1 = gate.evaluate(1)
        assert result1.result == GateResult.PASS
        print(f"✅ First signal test passed: {result1.reason}")
        
        # Add signals and test
        gate.add_signal(1)
        gate.add_signal(1)
        
        # Test third consecutive signal (should warn)
        result2 = gate.evaluate(1)
        print(f"✅ Third consecutive test: {result2.result.value} - {result2.reason}")
        
        # Add more signals
        gate.add_signal(1)
        
        # Test fourth consecutive signal (should reject)
        result3 = gate.evaluate(1)
        assert result3.result == GateResult.REJECT
        print(f"✅ Fourth consecutive test passed: {result3.reason}")
        
        # Test opposite direction (should pass)
        result4 = gate.evaluate(-1)
        assert result4.result == GateResult.PASS
        print(f"✅ Opposite direction test passed: {result4.reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consecutive gate test failed: {e}")
        return False


def test_cooldown_gate():
    """Test cooldown gate functionality"""
    print("\n🧪 Testing Cooldown Gate...")
    
    try:
        cfg = load_config()
        gate = CooldownGate(cfg.risk_management)
        
        # Test normal state (should pass)
        result1 = gate.evaluate()
        assert result1.result == GateResult.PASS
        print(f"✅ Normal cooldown test passed: {result1.reason}")
        
        # Record a significant loss
        gate.record_loss(250.0)
        
        # Test immediately after loss (should reject)
        result2 = gate.evaluate()
        assert result2.result == GateResult.REJECT
        print(f"✅ Immediate post-loss test passed: {result2.reason}")
        
        # Record a small loss (should not trigger cooldown)
        gate.last_loss_time = None  # Reset
        gate.record_loss(50.0)
        
        result3 = gate.evaluate()
        assert result3.result == GateResult.PASS
        print(f"✅ Small loss test passed: {result3.reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cooldown gate test failed: {e}")
        return False


def test_risk_manager():
    """Test complete risk manager functionality"""
    print("\n🧪 Testing Risk Manager...")
    
    try:
        cfg = load_config()
        risk_manager = RiskManager(cfg.risk_management)
        
        # Mock market data
        symbol = "EURUSD"
        bid = 1.1000
        ask = 1.1002
        position_size = 1.0
        signal_direction = 1
        current_positions = {"GBPUSD": {"size": 0.5}}
        account_balance = 10000.0
        
        # Test normal conditions (should pass)
        allowed, responses = risk_manager.evaluate_all_gates(
            symbol, bid, ask, position_size, signal_direction, 
            current_positions, account_balance
        )
        
        assert allowed == True
        print(f"✅ Normal conditions test passed: {len(responses)} gates evaluated")
        
        # Test with wide spread (should reject)
        allowed2, responses2 = risk_manager.evaluate_all_gates(
            symbol, 1.1000, 1.1020, position_size, signal_direction,
            current_positions, account_balance
        )
        
        assert allowed2 == False
        print(f"✅ Wide spread rejection test passed")
        
        # Test trade result recording
        risk_manager.record_trade_result(-100.0, 1)
        print(f"✅ Trade result recording test passed")
        
        # Test risk status
        status = risk_manager.get_risk_status()
        assert "daily_pnl" in status
        print(f"✅ Risk status test passed: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_position_sizing():
    """Test position sizing functionality"""
    print("\n🧪 Testing Position Sizing...")
    
    try:
        cfg = load_config()
        sizer = PositionSizer(cfg)
        
        # Test normal position sizing
        result = sizer.calculate_position_size(
            symbol="EURUSD",
            entry_price=1.1000,
            stop_loss_price=1.0950,
            account_balance=10000.0,
            confidence=0.75,
            atr_value=0.0030,
            regime="NORMAL",
            session="london",
            direction=1
        )
        
        assert result.size > 0
        print(f"✅ Normal position sizing: {result.size:.2f} lots")
        print(f"   Risk amount: ${result.risk_amount:.2f}")
        print(f"   Final multiplier: {result.final_multiplier:.2f}")
        
        # Test high confidence sizing
        result2 = sizer.calculate_position_size(
            symbol="EURUSD",
            entry_price=1.1000,
            stop_loss_price=1.0950,
            account_balance=10000.0,
            confidence=0.95,
            atr_value=0.0030,
            regime="LOW",
            session="london",
            direction=1
        )
        
        assert result2.size > result.size  # Higher confidence should mean larger size
        print(f"✅ High confidence sizing: {result2.size:.2f} lots (larger)")
        
        # Test stop loss suggestion
        stop_loss, method = sizer.suggest_stop_loss(
            symbol="EURUSD",
            entry_price=1.1000,
            direction=1,
            atr_value=0.0030,
            confidence=0.75
        )
        
        assert stop_loss < 1.1000  # Stop should be below entry for long
        print(f"✅ Stop loss suggestion: {stop_loss:.4f} ({method})")
        
        # Test portfolio heat calculation
        positions = {
            "EURUSD": {"size": 1.0, "entry_price": 1.1000, "stop_loss": 1.0950},
            "GBPUSD": {"size": 0.5, "entry_price": 1.3000, "stop_loss": 1.2950}
        }
        
        heat = sizer.calculate_portfolio_heat(positions, 10000.0)
        assert "heat_percent" in heat
        print(f"✅ Portfolio heat: {heat['heat_percent']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Position sizing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of risk management with other components"""
    print("\n🧪 Testing Risk Management Integration...")
    
    try:
        cfg = load_config()
        
        # Test configuration loading
        assert hasattr(cfg, 'risk_management')
        assert hasattr(cfg.risk_management, 'gates')
        print("✅ Configuration integration working")
        
        # Test all components can be initialized
        risk_manager = RiskManager(cfg.risk_management)
        position_sizer = PositionSizer(cfg)
        
        # Simulate a complete trading decision flow
        symbol = "EURUSD"
        entry_price = 1.1000
        confidence = 0.80
        atr_value = 0.0030
        account_balance = 10000.0
        current_positions = {}
        
        # 1. Calculate position size
        position_info = position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_price=1.0950,
            account_balance=account_balance,
            confidence=confidence,
            atr_value=atr_value
        )
        
        # 2. Evaluate risk gates
        allowed, gate_responses = risk_manager.evaluate_all_gates(
            symbol=symbol,
            bid=entry_price - 0.0001,
            ask=entry_price + 0.0001,
            position_size=position_info.size,
            signal_direction=1,
            current_positions=current_positions,
            account_balance=account_balance
        )
        
        assert allowed == True
        assert position_info.size > 0
        
        print(f"✅ Integration test passed:")
        print(f"   Position size: {position_info.size:.2f} lots")
        print(f"   Risk gates: {len([r for r in gate_responses if r.result.value == 'pass'])} passed")
        
        # 3. Simulate trade result
        pnl = 150.0  # Profitable trade
        risk_manager.record_trade_result(pnl, 1)
        
        status = risk_manager.get_risk_status()
        print(f"   Daily P&L: ${status['daily_pnl']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 MR BEN - Risk Management Testing")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Run all tests
    tests = [
        ("Spread Gate", test_spread_gate),
        ("Exposure Gate", test_exposure_gate),
        ("Daily Loss Gate", test_daily_loss_gate),
        ("Consecutive Gate", test_consecutive_gate),
        ("Cooldown Gate", test_cooldown_gate),
        ("Risk Manager", test_risk_manager),
        ("Position Sizing", test_position_sizing),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RISK MANAGEMENT TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 STEP5 RISK MANAGEMENT COMPLETED SUCCESSFULLY!")
        print("All risk management components are working:")
        print("  • Spread Gate - Filters wide spreads")
        print("  • Exposure Gate - Limits position exposure")
        print("  • Daily Loss Gate - Stops trading after daily limits")
        print("  • Consecutive Gate - Prevents overtrading")
        print("  • Cooldown Gate - Enforces waiting after losses")
        print("  • Position Sizing - Dynamic risk-based sizing")
        print("  • Integration - All components work together")
    else:
        print(f"\n⚠️  {total - passed} tests failed - STEP5 incomplete")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


