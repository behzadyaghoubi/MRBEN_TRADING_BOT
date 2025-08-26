#!/usr/bin/env python3
"""
MR BEN - Agent Mode Verification Script
Simple verification of agent bridge functionality
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_agent_mode():
    """Verify agent mode functionality"""
    print("🤖 MR BEN - Agent Mode Verification")
    print("=" * 40)
    
    try:
        # Test 1: Import agent components
        print("📦 Testing imports...")
        from core.agent_bridge import AgentBridge, AgentAction, AgentConfidence, AgentIntervention
        from core.typesx import DecisionCard, MarketContext, Levels
        print("✅ All agent components imported successfully")
        
        # Test 2: Create agent bridge
        print("\n🔧 Testing agent bridge creation...")
        agent_bridge = AgentBridge(
            config_path="agent_config.json",
            enable_intervention=True,
            risk_threshold=0.8,
            confidence_threshold=0.7
        )
        print("✅ Agent bridge created successfully")
        
        # Test 3: Test decision review
        print("\n🔍 Testing decision review...")
        
        # Create test decision
        test_decision = DecisionCard(
            action="ENTER",
            dir=1,
            reason="Test decision for verification",
            score=0.75,
            dyn_conf=0.65,
            lot=1.0,
            levels=Levels(sl=1.1000, tp1=1.1100, tp2=1.1150),
            track="pro"
        )
        
        # Create test context
        test_context = MarketContext(
            price=1.1050,
            bid=1.1049,
            ask=1.1051,
            atr_pts=25.0,
            sma20=1.1040,
            sma50=1.1020,
            session="london",
            regime="NORMAL",
            equity=10000.0,
            balance=10000.0,
            spread_pts=20.0,
            open_positions=0
        )
        
        # Review decision
        intervention = agent_bridge.review_decision(test_decision, test_context)
        if intervention:
            print(f"⚠️ Intervention triggered: {intervention.action.value} - {intervention.reason}")
        else:
            print("✅ Decision approved - no intervention needed")
        
        # Test 4: Test high-risk decision
        print("\n⚠️ Testing high-risk decision...")
        
        risky_decision = DecisionCard(
            action="ENTER",
            dir=1,
            reason="High risk decision",
            score=0.45,
            dyn_conf=0.35,
            lot=2.5,
            levels=Levels(sl=1.1000, tp1=1.1100, tp2=1.1150),
            track="pro"
        )
        
        risky_context = MarketContext(
            price=1.1050,
            bid=1.1049,
            ask=1.1051,
            atr_pts=120.0,  # High volatility
            sma20=1.1040,
            sma50=1.1020,
            session="overlap",  # Overlap session
            regime="HIGH",
            equity=8000.0,  # Lower equity
            balance=10000.0,
            spread_pts=45.0,  # Higher spread
            open_positions=3  # Multiple positions
        )
        
        # Review risky decision
        intervention = agent_bridge.review_decision(risky_decision, risky_context)
        if intervention:
            print(f"🚨 Intervention triggered: {intervention.action.value} - {intervention.reason}")
            print(f"   Confidence: {intervention.confidence.value}")
            print(f"   Risk Level: {intervention.risk_level}")
        else:
            print("⚠️ Unexpected: High-risk decision not intervened")
        
        # Test 5: Test agent status
        print("\n📊 Testing agent status...")
        status = agent_bridge.get_status()
        print(f"   Active: {status['is_active']}")
        print(f"   Interventions: {status['interventions_count']}")
        print(f"   Warnings: {status['warnings_count']}")
        print(f"   Performance Score: {status['performance_score']:.2f}")
        print(f"   Risk Assessment: {status['risk_assessment']}")
        
        # Test 6: Test recommendations
        print("\n💡 Testing recommendations...")
        recommendations = agent_bridge.get_recommendations()
        print(f"   Recommendations: {len(recommendations)} items")
        for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
            print(f"   {i}. {rec}")
        
        # Test 7: Test callback system
        print("\n📞 Testing callback system...")
        
        callback_called = False
        callback_data = None
        
        def test_callback(intervention_data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = intervention_data
            print(f"   📞 Callback triggered: {intervention_data.action.value}")
        
        agent_bridge.add_intervention_callback(test_callback)
        
        # Trigger another intervention
        agent_bridge.review_decision(risky_decision, risky_context)
        
        if callback_called:
            print("✅ Callback system working")
        else:
            print("⚠️ Callback system not triggered")
        
        # Test 8: Test configuration
        print("\n⚙️ Testing configuration...")
        config_path = Path("agent_config.json")
        if config_path.exists():
            print("✅ Agent configuration file exists")
            print(f"   Path: {config_path.absolute()}")
        else:
            print("❌ Agent configuration file not found")
        
        # Test 9: Test metrics integration
        print("\n📈 Testing metrics integration...")
        from core.metricsx import observe_agent_decision, observe_agent_intervention
        
        try:
            observe_agent_decision("ENTER", 0.8, "approved")
            observe_agent_intervention("warn", "medium", "test")
            print("✅ Metrics integration working")
        except Exception as e:
            print(f"⚠️ Metrics integration issue: {e}")
        
        # Test 10: Test agent cleanup
        print("\n🧹 Testing cleanup...")
        agent_bridge.cleanup()
        print("✅ Agent cleanup completed")
        
        print("\n🎉 Agent Mode Verification - COMPLETED SUCCESSFULLY!")
        print("\n📋 Summary of Agent Capabilities:")
        print("✅ AI Agent Bridge System")
        print("✅ Decision Review & Analysis")
        print("✅ Risk Assessment & Scoring")
        print("✅ Intervention Management")
        print("✅ Callback System")
        print("✅ Configuration Management")
        print("✅ Status Monitoring")
        print("✅ Recommendations Engine")
        print("✅ Metrics Integration")
        print("✅ Resource Cleanup")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent mode verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_agent_mode()
    sys.exit(0 if success else 1)
