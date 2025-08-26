#!/usr/bin/env python3
"""
Simple test script for the refactored MR BEN Trading System.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports."""
    print("🧪 Testing basic imports...")
    
    try:
        # Test core exceptions
        from core.exceptions import TradingSystemError, MT5ConnectionError, DataError, RiskError
        print("✅ Core exceptions imported successfully")
        
        # Test metrics
        from core.metrics import PerformanceMetrics
        metrics = PerformanceMetrics()
        print("✅ Performance metrics imported and instantiated successfully")
        
        # Test config
        from config.settings import MT5Config
        print("✅ Config settings imported successfully")
        
        # Test risk manager
        from risk.manager import EnhancedRiskManager
        risk_manager = EnhancedRiskManager()
        print("✅ Risk manager imported and instantiated successfully")
        
        # Test AI system
        from ai.system import MRBENAdvancedAISystem
        ai_system = MRBENAdvancedAISystem()
        print("✅ AI system imported and instantiated successfully")
        
        # Test trade executor
        from execution.executor import EnhancedTradeExecutor
        executor = EnhancedTradeExecutor(risk_manager)
        print("✅ Trade executor imported and instantiated successfully")
        
        # Test data manager
        from data.manager import MT5DataManager
        data_manager = MT5DataManager("XAUUSD.PRO", 15)
        print("✅ Data manager imported and instantiated successfully")
        
        # Test utilities
        from utils.helpers import round_price
        rounded = round_price("XAUUSD.PRO", 2000.12345)
        print(f"✅ Utility functions: rounded price {rounded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test performance metrics
        from core.metrics import PerformanceMetrics
        metrics = PerformanceMetrics()
        metrics.record_cycle(0.1)
        metrics.record_trade()
        stats = metrics.get_stats()
        print(f"✅ Performance metrics: {stats['cycle_count']} cycles, {stats['total_trades']} trades")
        
        # Test risk manager
        from risk.manager import EnhancedRiskManager
        risk_manager = EnhancedRiskManager()
        print(f"✅ Risk manager: base risk {risk_manager.base_risk}, max lot {risk_manager.max_lot}")
        
        # Test AI system
        from ai.system import MRBENAdvancedAISystem
        ai_system = MRBENAdvancedAISystem()
        print(f"✅ AI system: {len(ai_system.models)} models, weights {ai_system.ensemble_weights}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 MR BEN Trading System - Simple Test")
    print("=" * 50)
    
    # Test imports
    if not test_basic_imports():
        return False
    
    # Test functionality
    if not test_basic_functionality():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! The refactored system is working correctly.")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 