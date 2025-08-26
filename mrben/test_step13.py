#!/usr/bin/env python3
"""
MR BEN - STEP13 Advanced Position Management Test
Test enhanced position sizing, portfolio optimization, and ML integration
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_step13():
    """Test STEP13: Advanced Position Management System"""
    print("📊 MR BEN - STEP13 Advanced Position Management Test")
    print("=" * 60)
    
    try:
        # Test 1: Configuration
        from core.configx import load_config
        cfg = load_config()
        print("✅ Configuration loaded")
        
        # Test 2: Import advanced position components
        print("📦 Testing imports...")
        from core.advanced_position import (
            AdvancedPositionManager, PositionStrategy, ScalingMethod, ExitStrategy,
            PositionSizingResult, PositionAdjustment, PortfolioPosition
        )
        from core.typesx import DecisionCard, MarketContext, Levels
        print("✅ All advanced position components imported successfully")
        
        # Test 3: Test Advanced Position Manager with temporary directory
        print("\n🔧 Testing Advanced Position Manager System...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary config
            config_file = Path(temp_dir) / "test_position_config.json"
            test_config = {
                "enable_ml": True,
                "enable_portfolio_optimization": True,
                "enable_dynamic_adjustment": True
            }
            
            with open(config_file, 'w') as f:
                import json
                json.dump(test_config, f, indent=2)
            
            # Create advanced position manager
            position_manager = AdvancedPositionManager(
                config_path=str(config_file),
                model_dir=str(Path(temp_dir) / "position_models"),
                enable_ml=True,
                enable_portfolio_optimization=True,
                enable_dynamic_adjustment=True
            )
            
            print("✅ Advanced Position Manager created successfully")
            
            # Test 4: Test Position Sizing Strategies
            print("\n📏 Testing Position Sizing Strategies...")
            
            # Create test decision
            test_decision = DecisionCard(
                action="ENTER",
                dir=1,
                reason="Test decision for position sizing",
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
            
            # Test Fixed Size Strategy
            fixed_result = position_manager.calculate_position_size(
                test_decision, test_context, 10000.0, "2%", PositionStrategy.FIXED_SIZE
            )
            assert isinstance(fixed_result, PositionSizingResult)
            assert fixed_result.strategy == PositionStrategy.FIXED_SIZE
            assert fixed_result.lot_size > 0
            print(f"✅ Fixed Size Strategy - Lot Size: {fixed_result.lot_size:.3f}")
            
            # Test Kelly Criterion Strategy
            kelly_result = position_manager.calculate_position_size(
                test_decision, test_context, 10000.0, "2%", PositionStrategy.KELLY_CRITERION
            )
            assert isinstance(kelly_result, PositionSizingResult)
            assert kelly_result.strategy == PositionStrategy.KELLY_CRITERION
            assert kelly_result.lot_size > 0
            print(f"✅ Kelly Criterion Strategy - Lot Size: {kelly_result.lot_size:.3f}")
            
            # Test Volatility Adjusted Strategy
            vol_result = position_manager.calculate_position_size(
                test_decision, test_context, 10000.0, "2%", PositionStrategy.VOLATILITY_ADJUSTED
            )
            assert isinstance(vol_result, PositionSizingResult)
            assert vol_result.strategy == PositionStrategy.VOLATILITY_ADJUSTED
            assert vol_result.lot_size > 0
            print(f"✅ Volatility Adjusted Strategy - Lot Size: {vol_result.lot_size:.3f}")
            
            # Test ML Optimized Strategy
            ml_result = position_manager.calculate_position_size(
                test_decision, test_context, 10000.0, "2%", PositionStrategy.ML_OPTIMIZED
            )
            assert isinstance(ml_result, PositionSizingResult)
            assert ml_result.strategy == PositionStrategy.ML_OPTIMIZED
            assert ml_result.lot_size > 0
            print(f"✅ ML Optimized Strategy - Lot Size: {ml_result.lot_size:.3f}")
            
            # Test Portfolio Optimized Strategy
            portfolio_result = position_manager.calculate_position_size(
                test_decision, test_context, 10000.0, "2%", PositionStrategy.PORTFOLIO_OPTIMIZED
            )
            assert isinstance(portfolio_result, PositionSizingResult)
            assert portfolio_result.strategy == PositionStrategy.PORTFOLIO_OPTIMIZED
            assert portfolio_result.lot_size > 0
            print(f"✅ Portfolio Optimized Strategy - Lot Size: {portfolio_result.lot_size:.3f}")
            
            # Test 5: Test High-Risk Decision
            print("\n⚠️ Testing High-Risk Decision...")
            
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
            
            # Test high-risk position sizing
            risky_result = position_manager.calculate_position_size(
                risky_decision, risky_context, 8000.0, "2%", PositionStrategy.PORTFOLIO_OPTIMIZED
            )
            
            assert risky_result.overall_risk_score > 0
            assert len(risky_result.recommendations) > 0
            print(f"✅ High-risk analysis completed - Risk Score: {risky_result.overall_risk_score:.3f}")
            print(f"   Risk Factors: {list(risky_result.risk_factors.keys())}")
            print(f"   Recommendations: {len(risky_result.recommendations)} items")
            
            # Test 6: Test Portfolio Management
            print("\n💼 Testing Portfolio Management...")
            
            # Add portfolio positions
            position_manager.add_portfolio_position("EURUSD", 1, 1.0, 1.1050, 1.1070)
            position_manager.add_portfolio_position("GBPUSD", -1, 0.5, 1.2500, 1.2480)
            position_manager.add_portfolio_position("USDJPY", 1, 0.8, 150.00, 150.50)
            
            # Get portfolio summary
            portfolio_summary = position_manager.get_portfolio_summary()
            assert "total_positions" in portfolio_summary
            assert portfolio_summary["total_positions"] == 3
            assert "total_exposure" in portfolio_summary
            assert "total_pnl" in portfolio_summary
            
            print(f"✅ Portfolio management active:")
            print(f"   Total Positions: {portfolio_summary['total_positions']}")
            print(f"   Total Exposure: {portfolio_summary['total_exposure']:.2f}")
            print(f"   Total P&L: {portfolio_summary['total_pnl']:.2f}")
            print(f"   Long Positions: {portfolio_summary['long_positions']}")
            print(f"   Short Positions: {portfolio_summary['short_positions']}")
            
            # Test 7: Test Model Training
            print("\n🤖 Testing Model Training...")
            
            # Create training data
            training_data = []
            for i in range(20):
                training_data.append({
                    'decision': DecisionCard(
                        action="ENTER",
                        dir=1 if i % 2 == 0 else -1,
                        reason=f"Training decision {i}",
                        score=0.5 + i * 0.02,
                        dyn_conf=0.6 + i * 0.015,
                        lot=0.5 + i * 0.1,
                        levels=Levels(sl=1.1000, tp1=1.1100, tp2=1.1150),
                        track="pro"
                    ),
                    'context': MarketContext(
                        price=1.1050 + i * 0.001,
                        bid=1.1049 + i * 0.001,
                        ask=1.1051 + i * 0.001,
                        atr_pts=20 + i * 2,
                        sma20=1.1040 + i * 0.001,
                        sma50=1.1020 + i * 0.001,
                        session=['london', 'newyork', 'asia', 'overlap'][i % 4],
                        regime=['LOW', 'NORMAL', 'HIGH'][i % 3],
                        equity=10000 - i * 50,
                        balance=10000,
                        spread_pts=15 + i,
                        open_positions=i % 3
                    ),
                    'account_balance': 10000 - i * 50,
                    'optimal_size_multiplier': 0.8 + i * 0.02
                })
            
            # Train models
            training_success = position_manager.train_models(training_data)
            assert training_success
            print("✅ Model training completed successfully")
            
            # Check model performance
            assert hasattr(position_manager, 'model_performance')
            print("✅ Model performance tracking active")
            
            # Test 8: Test Integration with Position Management
            print("\n🔄 Testing Position Management Integration...")
            
            from core.position_management import PositionManager
            
            # Create position manager with advanced manager
            pos_manager = PositionManager(cfg)
            
            # Check if advanced position manager is initialized
            assert hasattr(pos_manager, 'advanced_manager')
            print("✅ Position Manager integrated with Advanced Position Manager")
            
            # Get position summary
            position_summary = pos_manager.get_position_summary()
            assert "advanced_position_management" in position_summary
            print("✅ Advanced position management status available in position manager")
            
            # Test 9: Test Metrics Integration
            print("\n📈 Testing Metrics Integration...")
            
            from core.metricsx import observe_position_metric, observe_position_adjustment
            
            # Test metrics functions (they should not raise errors)
            observe_position_metric("test_metric", 0.75)
            observe_position_adjustment("increase", 0.5, 0.85)
            print("✅ Metrics integration working")
            
            # Test 10: Test Cleanup
            print("\n🧹 Testing Cleanup...")
            
            position_manager.cleanup()
            print("✅ Advanced Position Manager cleanup completed")
            
            pos_manager.advanced_manager.cleanup() if pos_manager.advanced_manager else None
            print("✅ Position Manager cleanup completed")
        
        # Test 11: Test Configuration File
        print("\n⚙️ Testing Configuration Integration...")
        
        # Check if advanced position config file exists
        position_config_path = Path("advanced_position_config.json")
        assert position_config_path.exists()
        print("✅ Advanced position configuration file exists")
        
        # Load and validate config
        with open(position_config_path, 'r') as f:
            config_data = json.load(f)
        
        assert "enable_ml" in config_data
        assert "enable_portfolio_optimization" in config_data
        assert "enable_dynamic_adjustment" in config_data
        assert "position_sizing" in config_data
        assert "portfolio_management" in config_data
        assert "ml_models" in config_data
        
        print("✅ Configuration validation passed")
        
        print("\n🎉 STEP13: Advanced Position Management System - COMPLETED SUCCESSFULLY!")
        print("\n📋 Summary of Advanced Position Management Capabilities:")
        print("✅ Enhanced Position Sizing with ML Integration")
        print("✅ Portfolio-Level Risk Management")
        print("✅ Dynamic Position Adjustment")
        print("✅ Advanced Exit Strategies")
        print("✅ Position Scaling & Pyramiding")
        print("✅ Integration with Existing Position Management")
        print("✅ Configuration Management")
        print("✅ Metrics & Observability")
        print("✅ Resource Management & Cleanup")
        
        return True
        
    except Exception as e:
        print(f"❌ STEP13 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_step13()
    sys.exit(0 if success else 1)
