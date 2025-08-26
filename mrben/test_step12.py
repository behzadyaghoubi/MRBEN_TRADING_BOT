#!/usr/bin/env python3
"""
MR BEN - STEP12 Advanced Risk Analytics Test
Test enhanced risk modeling, predictive assessment, and dynamic threshold adjustment
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_step12():
    """Test STEP12: Advanced Risk Analytics System"""
    print("📊 MR BEN - STEP12 Advanced Risk Analytics Test")
    print("=" * 55)
    
    try:
        # Test 1: Configuration
        from core.configx import load_config
        cfg = load_config()
        print("✅ Configuration loaded")
        
        # Test 2: Import advanced risk components
        print("📦 Testing imports...")
        from core.advanced_risk import (
            AdvancedRiskAnalytics, RiskProfile, RiskCorrelation, 
            DynamicThreshold, RiskModelType, RiskMetricType
        )
        from core.typesx import DecisionCard, MarketContext, Levels
        print("✅ All advanced risk components imported successfully")
        
        # Test 3: Test Advanced Risk Analytics with temporary directory
        print("\n🔧 Testing Advanced Risk Analytics System...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary config
            config_file = Path(temp_dir) / "test_risk_config.json"
            test_config = {
                "enable_ml": True,
                "enable_dynamic_thresholds": True,
                "enable_correlation_analysis": True
            }
            
            with open(config_file, 'w') as f:
                import json
                json.dump(test_config, f, indent=2)
            
            # Create advanced risk analytics
            risk_analytics = AdvancedRiskAnalytics(
                config_path=str(config_file),
                model_dir=str(Path(temp_dir) / "risk_models"),
                enable_ml=True,
                enable_dynamic_thresholds=True,
                enable_correlation_analysis=True
            )
            
            print("✅ Advanced Risk Analytics created successfully")
            
            # Test 4: Test Risk Analysis
            print("\n🔍 Testing Risk Analysis...")
            
            # Create test decision
            test_decision = DecisionCard(
                action="ENTER",
                dir=1,
                reason="Test decision for risk analysis",
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
            
            # Perform risk analysis
            risk_profile = risk_analytics.analyze_risk(test_decision, test_context)
            
            assert isinstance(risk_profile, RiskProfile)
            assert 0.0 <= risk_profile.overall_risk_score <= 1.0
            assert risk_profile.volatility_risk > 0
            assert risk_profile.correlation_risk > 0
            assert risk_profile.concentration_risk > 0
            assert risk_profile.liquidity_risk > 0
            assert risk_profile.regime_risk > 0
            assert len(risk_profile.recommendations) > 0
            assert 0.0 <= risk_profile.model_confidence <= 1.0
            
            print(f"✅ Risk analysis completed - Overall Risk: {risk_profile.overall_risk_score:.3f}")
            print(f"   Volatility Risk: {risk_profile.volatility_risk:.3f}")
            print(f"   Correlation Risk: {risk_profile.correlation_risk:.3f}")
            print(f"   Concentration Risk: {risk_profile.concentration_risk:.3f}")
            print(f"   Liquidity Risk: {risk_profile.liquidity_risk:.3f}")
            print(f"   Regime Risk: {risk_profile.regime_risk:.3f}")
            print(f"   Model Confidence: {risk_profile.model_confidence:.3f}")
            
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
            
            # Perform risk analysis on risky decision
            risky_profile = risk_analytics.analyze_risk(risky_decision, risky_context)
            
            assert risky_profile.overall_risk_score > risk_profile.overall_risk_score
            print(f"✅ High-risk analysis completed - Risk Score: {risky_profile.overall_risk_score:.3f}")
            print(f"   Risk Factors: {list(risky_profile.risk_factors.keys())}")
            print(f"   Recommendations: {len(risky_profile.recommendations)} items")
            
            # Test 6: Test Dynamic Thresholds
            print("\n⚙️ Testing Dynamic Thresholds...")
            
            # Check if dynamic thresholds were updated
            risk_summary = risk_analytics.get_risk_summary()
            assert "dynamic_thresholds" in risk_summary
            
            dynamic_thresholds = risk_summary["dynamic_thresholds"]
            print(f"✅ Dynamic thresholds initialized: {len(dynamic_thresholds)} metrics")
            
            for metric, threshold_info in dynamic_thresholds.items():
                print(f"   {metric}: {threshold_info['current']:.3f} (base: {threshold_info['base']:.3f})")
            
            # Test 7: Test Correlation Analysis
            print("\n🔗 Testing Correlation Analysis...")
            
            # Check correlation matrix
            assert hasattr(risk_analytics, 'correlation_matrix')
            print(f"✅ Correlation analysis active - Matrix size: {len(risk_analytics.correlation_matrix)}")
            
            # Test 8: Test Model Training
            print("\n🤖 Testing Model Training...")
            
            # Create training data
            training_data = []
            for i in range(20):
                training_data.append({
                    'atr_pts': 20 + i * 2,
                    'spread_pts': 15 + i,
                    'equity': 10000 - i * 50,
                    'balance': 10000,
                    'open_positions': i % 3,
                    'session': ['london', 'newyork', 'asia', 'overlap'][i % 4],
                    'regime': ['LOW', 'NORMAL', 'HIGH'][i % 3],
                    'decision': {
                        'score': 0.5 + i * 0.02,
                        'dyn_conf': 0.6 + i * 0.015,
                        'lot': 0.5 + i * 0.1
                    },
                    'risk_score': 0.3 + i * 0.02
                })
            
            # Train models
            training_success = risk_analytics.train_models(training_data)
            assert training_success
            print("✅ Model training completed successfully")
            
            # Check model performance
            updated_summary = risk_analytics.get_risk_summary()
            assert "model_performance" in updated_summary
            print("✅ Model performance tracking active")
            
            # Test 9: Test Risk Summary
            print("\n📊 Testing Risk Summary...")
            
            summary = risk_analytics.get_risk_summary()
            assert "total_analyses" in summary
            assert "average_risk_score" in summary
            assert "risk_distribution" in summary
            
            print(f"✅ Risk summary generated:")
            print(f"   Total Analyses: {summary['total_analyses']}")
            print(f"   Average Risk Score: {summary['average_risk_score']:.3f}")
            print(f"   Risk Distribution: {summary['risk_distribution']}")
            
            # Test 10: Test Integration with Risk Gates
            print("\n🔄 Testing Risk Gates Integration...")
            
            from core.risk_gates import RiskManager
            
            # Create risk manager with advanced analytics
            risk_manager = RiskManager(cfg)
            
            # Check if advanced risk analytics is initialized
            assert hasattr(risk_manager, 'advanced_risk')
            print("✅ Risk Manager integrated with Advanced Risk Analytics")
            
            # Get risk status
            risk_status = risk_manager.get_risk_status()
            assert "advanced_risk" in risk_status
            print("✅ Advanced risk status available in risk manager")
            
            # Test 11: Test Metrics Integration
            print("\n📈 Testing Metrics Integration...")
            
            from core.metricsx import observe_risk_metric, observe_risk_prediction
            
            # Test metrics functions (they should not raise errors)
            observe_risk_metric("test_metric", 0.75)
            observe_risk_prediction(0.6, 0.65, 0.92)
            print("✅ Metrics integration working")
            
            # Test 12: Test Cleanup
            print("\n🧹 Testing Cleanup...")
            
            risk_analytics.cleanup()
            print("✅ Advanced Risk Analytics cleanup completed")
            
            risk_manager.advanced_risk.cleanup() if risk_manager.advanced_risk else None
            print("✅ Risk Manager cleanup completed")
        
        # Test 13: Test Configuration File
        print("\n⚙️ Testing Configuration Integration...")
        
        # Check if risk analytics config file exists
        risk_config_path = Path("risk_analytics_config.json")
        assert risk_config_path.exists()
        print("✅ Risk analytics configuration file exists")
        
        # Load and validate config
        with open(risk_config_path, 'r') as f:
            config_data = json.load(f)
        
        assert "enable_ml" in config_data
        assert "enable_dynamic_thresholds" in config_data
        assert "enable_correlation_analysis" in config_data
        assert "risk_models" in config_data
        assert "dynamic_thresholds" in config_data
        
        print("✅ Configuration validation passed")
        
        print("\n🎉 STEP12: Advanced Risk Analytics System - COMPLETED SUCCESSFULLY!")
        print("\n📋 Summary of Advanced Risk Capabilities:")
        print("✅ Enhanced Risk Modeling with ML Integration")
        print("✅ Dynamic Risk Threshold Adjustment")
        print("✅ Risk Correlation Analysis")
        print("✅ Predictive Risk Assessment")
        print("✅ Advanced Risk Metrics & Visualization")
        print("✅ Integration with Existing Risk Gates")
        print("✅ Model Training & Performance Tracking")
        print("✅ Configuration Management")
        print("✅ Metrics & Observability")
        print("✅ Resource Management & Cleanup")
        
        return True
        
    except Exception as e:
        print(f"❌ STEP12 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_step12()
    sys.exit(0 if success else 1)
