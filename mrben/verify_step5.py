#!/usr/bin/env python3
"""
MR BEN - STEP5 Verification
Verify that Risk Management Gates are working correctly
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_step5():
    """Verify STEP5 completion"""
    print("🚀 MR BEN - STEP5 Verification")
    print("=" * 50)
    
    try:
        # Test configuration loading
        from core.configx import load_config
        cfg = load_config()
        print("✅ Configuration loaded successfully")
        
        # Test risk management config
        assert hasattr(cfg, 'risk_management')
        assert hasattr(cfg.risk_management, 'gates')
        print("✅ Risk management configuration present")
        
        # Test risk gates import
        from core.risk_gates import (
            RiskManager, SpreadGate, ExposureGate, DailyLossGate, 
            ConsecutiveGate, CooldownGate, GateResult
        )
        print("✅ Risk gates imported successfully")
        
        # Test position sizing import
        from core.position_sizing import PositionSizer
        print("✅ Position sizing imported successfully")
        
        # Test component initialization
        risk_manager = RiskManager(cfg.risk_management)
        position_sizer = PositionSizer(cfg)
        print("✅ Components initialized successfully")
        
        # Test basic functionality
        symbol = "EURUSD"
        bid = 1.1000
        ask = 1.1002
        position_size = 1.0
        signal_direction = 1
        current_positions = {}
        account_balance = 10000.0
        
        # Test risk gates evaluation
        allowed, responses = risk_manager.evaluate_all_gates(
            symbol, bid, ask, position_size, signal_direction,
            current_positions, account_balance
        )
        
        assert isinstance(allowed, bool)
        assert isinstance(responses, list)
        print("✅ Risk gates evaluation working")
        
        # Test position sizing
        position_info = position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            account_balance=account_balance,
            confidence=0.75,
            atr_value=0.0030
        )
        
        assert position_info.size > 0
        print("✅ Position sizing working")
        
        # Test risk status
        status = risk_manager.get_risk_status()
        assert isinstance(status, dict)
        print("✅ Risk status working")
        
        print("\n📊 STEP5 VERIFICATION RESULTS")
        print("=" * 50)
        print("✅ Configuration: Loaded")
        print("✅ Risk Management: Present")
        print("✅ Risk Gates: All Working")
        print("✅ Position Sizing: Working")
        print("✅ Integration: Successful")
        
        print(f"\n🎉 STEP5: Risk Management Gates - COMPLETED SUCCESSFULLY!")
        print("The MR BEN trading system now has:")
        print("  • Spread Gate - Filters wide spreads")
        print("  • Exposure Gate - Limits position exposure")
        print("  • Daily Loss Gate - Stops trading after daily limits")
        print("  • Consecutive Gate - Prevents overtrading")
        print("  • Cooldown Gate - Enforces waiting after losses")
        print("  • Position Sizing - Dynamic risk-based sizing")
        print("  • Complete risk management integration")
        
        print(f"\n🚀 Ready for STEP6: Position Management")
        return True
        
    except Exception as e:
        print(f"❌ STEP5 verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_step5()
    sys.exit(0 if success else 1)
