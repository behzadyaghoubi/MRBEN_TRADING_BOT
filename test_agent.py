#!/usr/bin/env python3
"""
Simple test script for MR BEN AI Agent system.
Tests basic functionality without requiring external dependencies.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_agent_imports():
    """Test that all agent components can be imported."""
    print("🧪 Testing agent imports...")
    
    try:
        from agent.schemas import (
            TradingMode, ToolPermission, DecisionStatus,
            ToolSchema, ReadOnlyToolSchema, WriteToolSchema
        )
        print("✅ Schemas imported successfully")
        
        from agent.prompts import (
            SUPERVISOR_SYSTEM_PROMPT, RISK_OFFICER_SYSTEM_PROMPT
        )
        print("✅ Prompts imported successfully")
        
        from agent.evaluators import SupervisorEvaluator, RiskOfficerEvaluator
        print("✅ Evaluators imported successfully")
        
        from agent.risk_gate import RiskGate
        print("✅ Risk gate imported successfully")
        
        from agent.decision_store import DecisionStore
        print("✅ Decision store imported successfully")
        
        from agent.bridge import MRBENAgentBridge
        print("✅ Agent bridge imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_schema_creation():
    """Test that schemas can be created and validated."""
    print("\n🧪 Testing schema creation...")
    
    try:
        from agent.schemas import (
            TradingMode, ToolPermission, DecisionStatus,
            ToolSchema, ReadOnlyToolSchema, WriteToolSchema
        )
        
        # Test enum values
        assert TradingMode.OBSERVE == "observe"
        assert TradingMode.PAPER == "paper"
        assert TradingMode.LIVE == "live"
        print("✅ TradingMode enum works")
        
        assert ToolPermission.READ_ONLY == "read_only"
        assert ToolPermission.WRITE_RESTRICTED == "write_restricted"
        assert ToolPermission.WRITE_FULL == "write_full"
        print("✅ ToolPermission enum works")
        
        # Test tool schema creation
        read_tool = ReadOnlyToolSchema(
            name="test_tool",
            description="Test tool",
            permission=ToolPermission.READ_ONLY
        )
        assert read_tool.name == "test_tool"
        assert read_tool.permission == ToolPermission.READ_ONLY
        print("✅ ReadOnlyToolSchema creation works")
        
        write_tool = WriteToolSchema(
            name="write_tool",
            description="Write tool",
            permission=ToolPermission.WRITE_RESTRICTED,
            risk_level="medium",
            max_impact_usd=1000.0
        )
        assert write_tool.name == "write_tool"
        assert write_tool.risk_level == "medium"
        print("✅ WriteToolSchema creation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema creation failed: {e}")
        return False

def test_decision_store():
    """Test decision store functionality."""
    print("\n🧪 Testing decision store...")
    
    try:
        from agent.decision_store import DecisionStore
        
        # Create decision store
        store = DecisionStore(
            storage_dir="test_decisions",
            max_memory_decisions=100,
            enable_jsonl=True,
            enable_parquet=False  # Disable to avoid pandas dependency
        )
        print("✅ Decision store created")
        
        # Test statistics
        stats = store.get_decision_statistics()
        assert stats["total_decisions"] == 0
        print("✅ Decision store statistics work")
        
        # Test storage info
        storage_info = store.get_storage_info()
        assert "storage_directory" in storage_info
        print("✅ Storage info works")
        
        # Cleanup
        store.clear_all()
        print("✅ Decision store cleanup works")
        
        return True
        
    except Exception as e:
        print(f"❌ Decision store test failed: {e}")
        return False

def test_risk_gate():
    """Test risk gate functionality."""
    print("\n🧪 Testing risk gate...")
    
    try:
        from agent.risk_gate import RiskGate
        from agent.evaluators import SupervisorEvaluator, RiskOfficerEvaluator
        from agent.schemas import TradingMode
        
        # Create evaluators
        supervisor = SupervisorEvaluator()
        risk_officer = RiskOfficerEvaluator()
        print("✅ Evaluators created")
        
        # Create risk gate
        config = {
            "max_daily_loss_percent": 2.0,
            "max_open_trades": 3,
            "max_position_size_usd": 10000.0,
            "max_risk_per_trade_percent": 1.0,
            "cooldown_after_loss_minutes": 30,
            "emergency_threshold_percent": 5.0
        }
        
        risk_gate = RiskGate(supervisor, risk_officer, config)
        print("✅ Risk gate created")
        
        # Test status
        status = risk_gate.get_status()
        assert "is_halted" in status
        assert "risk_level" in status
        print("✅ Risk gate status works")
        
        # Test halt/resume
        risk_gate.halt_trading("Test halt", emergency=False)
        assert risk_gate.is_halted == True
        print("✅ Risk gate halt works")
        
        risk_gate.resume_trading("Test resume")
        assert risk_gate.is_halted == False
        print("✅ Risk gate resume works")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk gate test failed: {e}")
        return False

def test_agent_bridge():
    """Test agent bridge functionality."""
    print("\n🧪 Testing agent bridge...")
    
    try:
        from agent.bridge import MRBENAgentBridge
        from agent.schemas import TradingMode
        
        # Agent configuration
        config = {
            "model_name": "gpt-5",
            "temperature": 0.1,
            "max_tokens": 4000,
            "structured_output": True,
            "risk_gate_enabled": True,
            "approval_required": True,
            "max_concurrent_decisions": 5,
            "decision_timeout_seconds": 300,
            "audit_logging": True,
            "performance_monitoring": True,
            "risk_gate": {
                "max_daily_loss_percent": 2.0,
                "max_open_trades": 3,
                "max_position_size_usd": 10000.0,
                "max_risk_per_trade_percent": 1.0,
                "cooldown_after_loss_minutes": 30,
                "emergency_threshold_percent": 5.0
            },
            "decision_storage_dir": "test_decisions",
            "max_memory_decisions": 100
        }
        
        # Create agent bridge
        agent = MRBENAgentBridge(config, TradingMode.OBSERVE)
        print("✅ Agent bridge created")
        
        # Test status
        status = agent.get_agent_status()
        assert "agent_id" in status
        assert "trading_mode" in status
        print("✅ Agent status works")
        
        # Test available tools
        tools = agent.get_available_tools()
        assert len(tools) > 0
        print(f"✅ Available tools: {len(tools)}")
        
        # Test tool execution (read-only)
        result = agent.execute_tool(
            tool_name="get_market_snapshot",
            input_data={"symbol": "XAUUSD.PRO", "timeframe_minutes": 15, "bars": 100},
            reasoning="Test execution",
            risk_assessment="Low risk",
            expected_outcome="Market data",
            urgency="normal",
            confidence=0.8
        )
        assert "success" in result
        print("✅ Tool execution works")
        
        # Test mode change
        success = agent.change_trading_mode(TradingMode.PAPER)
        assert success == True
        print("✅ Mode change works")
        
        # Cleanup
        agent.shutdown()
        print("✅ Agent shutdown works")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 MR BEN AI Agent System - Test Suite")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    tests = [
        ("Agent Imports", test_agent_imports),
        ("Schema Creation", test_schema_creation),
        ("Decision Store", test_decision_store),
        ("Risk Gate", test_risk_gate),
        ("Agent Bridge", test_agent_bridge)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI Agent system is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
