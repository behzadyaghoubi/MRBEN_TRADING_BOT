#!/usr/bin/env python3
"""
MR BEN - STEP11 Agent Supervision Test
Test AI agent bridge and supervision functionality
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_step11():
    """Test STEP11: AI Agent Supervision System"""
    print("🤖 MR BEN - STEP11 Agent Supervision Test")
    print("=" * 50)
    
    try:
        # Test 1: Configuration
        from core.configx import load_config
        cfg = load_config()
        print("✅ Configuration loaded")
        
        # Test 2: Import agent components
        from core.agent_bridge import AgentBridge, AgentAction, AgentConfidence, AgentIntervention
        from core.typesx import DecisionCard, MarketContext, Levels
        from core.ab import ABRunner
        from core.context_factory import ContextFactory
        print("✅ Agent components imported")
        
        # Test 3: Test Agent Bridge with temporary config
        print("\n🤖 Testing Agent Bridge System...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_agent_config.json"
            
            # Create test configuration
            test_config = {
                "enable_intervention": True,
                "risk_threshold": 0.7,
                "confidence_threshold": 0.6,
                "monitoring_interval": 5.0
            }
            
            with open(config_file, 'w') as f:
                import json
                json.dump(test_config, f, indent=2)
            
            # Create agent bridge
            agent_bridge = AgentBridge(
                config_path=str(config_file),
                enable_intervention=True,
                risk_threshold=0.7,
                confidence_threshold=0.6
            )
            
            # Test initial state
            initial_status = agent_bridge.get_status()
            assert not initial_status['is_active']
            assert initial_status['interventions_count'] == 0
            print("✅ Initial agent state correct")
            
            # Test monitoring start
            agent_bridge.start_monitoring()
            time.sleep(0.2)  # Wait for monitoring to start
            
            active_status = agent_bridge.get_status()
            assert active_status['is_active']
            print("✅ Agent monitoring started")
            
            # Test 4: Test Decision Review
            print("\n🔍 Testing Decision Review...")
            
            # Create test decision and context
            test_decision = DecisionCard(
                action="ENTER",
                direction=1,
                reason="Test decision",
                score=0.75,
                confidence=0.65,
                lot=1.0,
                levels=Levels(sl=1.1000, tp1=1.1100, tp2=1.1150),
                track="pro"
            )
            
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
            
            # Review normal decision
            intervention = agent_bridge.review_decision(test_decision, test_context)
            assert intervention is None  # Should not intervene on normal decision
            print("✅ Normal decision review working")
            
            # Test 5: Test High-Risk Decision
            print("\n⚠️ Testing High-Risk Decision...")
            
            risky_decision = DecisionCard(
                action="ENTER",
                direction=1,
                reason="High risk decision",
                score=0.45,
                confidence=0.35,
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
            assert intervention is not None  # Should intervene on risky decision
            assert intervention.action in [AgentAction.WARN, AgentAction.INTERVENE]
            print("✅ High-risk decision intervention working")
            
            # Test 6: Test Callbacks
            print("\n📞 Testing Callback System...")
            
            callback_called = False
            callback_data = None
            
            def test_callback(intervention_data):
                nonlocal callback_called, callback_data
                callback_called = True
                callback_data = intervention_data
            
            agent_bridge.add_intervention_callback(test_callback)
            
            # Trigger another intervention
            agent_bridge.review_decision(risky_decision, risky_context)
            time.sleep(0.1)  # Wait for callback
            
            assert callback_called
            assert callback_data is not None
            print("✅ Callback system working")
            
            # Test 7: Test A/B Testing Integration
            print("\n🔄 Testing A/B Testing Integration...")
            
            ctx_factory = ContextFactory()
            ab_runner = ABRunner(
                ctx_factory=ctx_factory.create_from_bar,
                symbol="EURUSD",
                agent_bridge=agent_bridge
            )
            
            # Test bar processing with agent supervision
            test_bar = {
                'timestamp': '2024-01-01T10:00:00Z',
                'close': 1.1050,
                'bid': 1.1049,
                'ask': 1.1051,
                'atr_pts': 25.0,
                'sma20': 1.1040,
                'sma50': 1.1020,
                'equity': 10000.0,
                'balance': 10000.0,
                'spread_pts': 20.0,
                'open_positions': 0
            }
            
            ab_runner.on_bar(test_bar)
            time.sleep(0.1)
            
            # Check statistics include agent status
            stats = ab_runner.get_statistics()
            assert 'agent_status' in stats
            assert 'agent_recommendations' in stats
            print("✅ A/B testing integration working")
            
            # Test 8: Test Recommendations
            print("\n💡 Testing Recommendations...")
            
            recommendations = agent_bridge.get_recommendations()
            assert isinstance(recommendations, list)
            print(f"✅ Agent recommendations: {len(recommendations)} items")
            
            # Test 9: Test Status and Statistics
            print("\n📊 Testing Status and Statistics...")
            
            status = agent_bridge.get_status()
            assert 'is_active' in status
            assert 'interventions_count' in status
            assert 'warnings_count' in status
            assert 'performance_score' in status
            assert 'risk_assessment' in status
            print("✅ Status reporting working")
            
            # Test 10: Test Statistics Reset
            print("\n🔄 Testing Statistics Reset...")
            
            agent_bridge.reset_statistics()
            reset_status = agent_bridge.get_status()
            assert reset_status['interventions_count'] == 0
            assert reset_status['warnings_count'] == 0
            print("✅ Statistics reset working")
            
            # Cleanup
            ab_runner.cleanup()
            agent_bridge.cleanup()
        
        # Test 11: Test Configuration Integration
        print("\n⚙️ Testing Configuration Integration...")
        
        # Check if agent config file exists
        agent_config_path = Path("agent_config.json")
        assert agent_config_path.exists()
        print("✅ Agent configuration file exists")
        
        # Test 12: Test Metrics Integration
        print("\n📈 Testing Metrics Integration...")
        
        from core.metricsx import observe_agent_decision, observe_agent_intervention
        
        # Test metrics functions (they should not raise errors)
        observe_agent_decision("ENTER", 0.8, "approved")
        observe_agent_intervention("warn", "medium", "test")
        print("✅ Metrics integration working")
        
        print("\n🎉 STEP11: AI Agent Supervision System - COMPLETED SUCCESSFULLY!")
        print("All components are working correctly:")
        print("✅ Agent Bridge System")
        print("✅ Decision Review & Analysis")
        print("✅ Risk Assessment")
        print("✅ Intervention Management")
        print("✅ Callback System")
        print("✅ A/B Testing Integration")
        print("✅ Configuration Management")
        print("✅ Status Monitoring")
        print("✅ Recommendations Engine")
        print("✅ Metrics Integration")
        
        return True
        
    except Exception as e:
        print(f"❌ STEP11 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_step11()
    sys.exit(0 if success else 1)
