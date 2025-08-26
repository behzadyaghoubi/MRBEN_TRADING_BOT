#!/usr/bin/env python3
"""
Quick test script to verify MR BEN Agent System components
"""

import sys
import os

def test_imports():
    """Test that all agent components can be imported"""
    try:
        from src.agent import (
            maybe_start_agent, DecisionCard, HealthEvent, AgentAction,
            AdvancedRiskGate, AdvancedPlaybooks, MLIntegration,
            PredictiveMaintenance, AdvancedAlerting, DashboardIntegration
        )
        print("✅ All agent components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['agent', 'dashboard', 'advanced']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing config key: {key}")
                return False
        
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_agent_bridge():
    """Test agent bridge functionality"""
    try:
        from src.agent.bridge import AgentBridge, DecisionCard
        
        # Create a test decision card
        dc = DecisionCard(
            ts="2025-08-18T19:10:00",
            symbol="XAUUSD.PRO",
            cycle=1,
            price=3338.84,
            sma20=3334.84,
            sma50=3337.54,
            raw_conf=0.700,
            adj_conf=0.720,
            threshold=0.650,
            allow_trade=True,
            regime_label="TRENDING_UP",
            regime_scores={"TRENDING_UP": 0.75},
            spread_pts=45.2,
            consecutive=3,
            open_positions=1,
            risk={},
            signal_src="SMA",
            mode="paper",
            agent_mode="guard"
        )
        
        print("✅ DecisionCard creation successful")
        print(f"   - Symbol: {dc.symbol}")
        print(f"   - Price: {dc.price}")
        print(f"   - Spread: {dc.spread_pts} pts")
        print(f"   - Agent Mode: {dc.agent_mode}")
        
        return True
    except Exception as e:
        print(f"❌ Agent bridge test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 MR BEN Agent System - Component Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Test", test_config),
        ("Agent Bridge Test", test_agent_bridge),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
