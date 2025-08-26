#!/usr/bin/env python3
"""
MR BEN - STEP14 Advanced Market Analysis Test
Test enhanced market regime detection, multi-timeframe analysis, and pattern recognition
"""

import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_step14():
    """Test STEP14: Advanced Market Analysis System"""
    print("📊 MR BEN - STEP14 Advanced Market Analysis Test")
    print("=" * 60)

    try:
        # Test 1: Configuration
        from core.configx import load_config

        cfg = load_config()
        print("✅ Configuration loaded")

        # Test 2: Import advanced market components
        print("📦 Testing imports...")
        from core.advanced_market import (
            AdvancedMarketAnalyzer,
            MarketRegimeAnalysis,
            MarketRegimeType,
        )
        from core.typesx import MarketContext

        print("✅ All advanced market components imported successfully")

        # Test 3: Test Advanced Market Analyzer with temporary directory
        print("\n🔧 Testing Advanced Market Analyzer System...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary config
            config_file = Path(temp_dir) / "test_market_config.json"
            test_config = {
                "enable_ml": True,
                "enable_multi_timeframe": True,
                "enable_pattern_recognition": True,
            }

            with open(config_file, 'w') as f:
                import json

                json.dump(test_config, f, indent=2)

            # Create advanced market analyzer
            market_analyzer = AdvancedMarketAnalyzer(
                config_path=str(config_file),
                model_dir=str(Path(temp_dir) / "market_models"),
                enable_ml=True,
                enable_multi_timeframe=True,
                enable_pattern_recognition=True,
            )

            print("✅ Advanced Market Analyzer created successfully")

            # Test 4: Test Market Regime Analysis
            print("\n📈 Testing Market Regime Analysis...")

            # Create test market context
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
                open_positions=0,
            )

            # Test basic regime detection
            basic_regime = market_analyzer._detect_basic_regime(test_context)
            assert isinstance(basic_regime, MarketRegimeType)
            print(f"✅ Basic Regime Detection - Regime: {basic_regime.value}")

            # Test enhanced regime analysis
            regime_analysis = market_analyzer.analyze_market_regime(test_context)
            assert isinstance(regime_analysis, MarketRegimeAnalysis)
            assert regime_analysis.regime in MarketRegimeType
            assert 0.0 <= regime_analysis.confidence <= 1.0
            assert 0.0 <= regime_analysis.trend_strength <= 1.0
            assert 0.0 <= regime_analysis.volatility_level <= 1.0
            assert 0.0 <= regime_analysis.momentum_score <= 1.0

            print(f"✅ Enhanced Regime Analysis - Regime: {regime_analysis.regime.value}")
            print(f"   Confidence: {regime_analysis.confidence:.3f}")
            print(f"   Trend Strength: {regime_analysis.trend_strength:.3f}")
            print(f"   Volatility Level: {regime_analysis.volatility_level:.3f}")
            print(f"   Momentum Score: {regime_analysis.momentum_score:.3f}")

            # Test 5: Test Volatility Analysis
            print("\n📊 Testing Volatility Analysis...")

            # Test different volatility levels
            low_vol_context = MarketContext(
                price=1.1050,
                bid=1.1049,
                ask=1.1051,
                atr_pts=15.0,
                sma20=1.1040,
                sma50=1.1020,
                session="london",
                regime="LOW",
                equity=10000.0,
                balance=10000.0,
                spread_pts=15.0,
                open_positions=0,
            )

            high_vol_context = MarketContext(
                price=1.1050,
                bid=1.1049,
                ask=1.1051,
                atr_pts=120.0,
                sma20=1.1040,
                sma50=1.1020,
                session="overlap",
                regime="HIGH",
                equity=10000.0,
                balance=10000.0,
                spread_pts=45.0,
                open_positions=2,
            )

            low_vol = market_analyzer._analyze_volatility(low_vol_context, None)
            high_vol = market_analyzer._analyze_volatility(high_vol_context, None)

            assert low_vol < high_vol
            assert 0.0 <= low_vol <= 1.0
            assert 0.0 <= high_vol <= 1.0

            print(f"✅ Volatility Analysis - Low: {low_vol:.3f}, High: {high_vol:.3f}")

            # Test 6: Test Trend Strength Analysis
            print("\n📈 Testing Trend Strength Analysis...")

            # Test strong uptrend
            strong_uptrend_context = MarketContext(
                price=1.1100,
                bid=1.1099,
                ask=1.1101,
                atr_pts=30.0,
                sma20=1.1080,
                sma50=1.1050,
                session="london",
                regime="NORMAL",
                equity=10000.0,
                balance=10000.0,
                spread_pts=20.0,
                open_positions=0,
            )

            # Test weak trend
            weak_trend_context = MarketContext(
                price=1.1050,
                bid=1.1049,
                ask=1.1051,
                atr_pts=20.0,
                sma20=1.1045,
                sma50=1.1040,
                session="asia",
                regime="LOW",
                equity=10000.0,
                balance=10000.0,
                spread_pts=15.0,
                open_positions=0,
            )

            strong_trend = market_analyzer._analyze_trend_strength(strong_uptrend_context, None)
            weak_trend = market_analyzer._analyze_trend_strength(weak_trend_context, None)

            assert strong_trend > weak_trend
            assert 0.0 <= strong_trend <= 1.0
            assert 0.0 <= weak_trend <= 1.0

            print(
                f"✅ Trend Strength Analysis - Strong: {strong_trend:.3f}, Weak: {weak_trend:.3f}"
            )

            # Test 7: Test Support/Resistance Levels
            print("\n🎯 Testing Support/Resistance Levels...")

            support_levels, resistance_levels = market_analyzer._identify_key_levels(
                test_context, None
            )

            assert isinstance(support_levels, list)
            assert isinstance(resistance_levels, list)
            assert len(support_levels) > 0
            assert len(resistance_levels) > 0

            print(
                f"✅ Support/Resistance Levels - Support: {len(support_levels)}, Resistance: {len(resistance_levels)}"
            )
            print(f"   Support Levels: {[f'{level:.4f}' for level in support_levels]}")
            print(f"   Resistance Levels: {[f'{level:.4f}' for level in resistance_levels]}")

            # Test 8: Test Pattern Indicators
            print("\n🔍 Testing Pattern Indicators...")

            pattern_indicators = market_analyzer._identify_pattern_indicators(test_context, None)

            assert isinstance(pattern_indicators, list)
            assert len(pattern_indicators) > 0

            print(f"✅ Pattern Indicators - Count: {len(pattern_indicators)}")
            print(f"   Indicators: {pattern_indicators}")

            # Test 9: Test Multi-Timeframe Analysis
            print("\n⏰ Testing Multi-Timeframe Analysis...")

            # Create historical data for timeframe analysis
            historical_data = []
            for i in range(200):  # 200 data points for multi-timeframe analysis
                historical_data.append(
                    {
                        'close': 1.1050 + i * 0.0001,
                        'volume': 1000 + i * 10,
                        'timestamp': f"2024-01-01T{i:02d}:00:00Z",
                    }
                )

            timeframe_analysis = market_analyzer._analyze_timeframes(test_context, historical_data)

            assert isinstance(timeframe_analysis, dict)
            assert "15m" in timeframe_analysis
            assert "1h" in timeframe_analysis
            assert "4h" in timeframe_analysis

            print(f"✅ Multi-Timeframe Analysis - Timeframes: {list(timeframe_analysis.keys())}")
            for tf, analysis in timeframe_analysis.items():
                print(f"   {tf}: {analysis}")

            # Test 10: Test ML Model Training
            print("\n🤖 Testing ML Model Training...")

            # Create training data
            training_data = []
            for i in range(50):
                training_data.append(
                    {
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
                            open_positions=i % 3,
                        ),
                        'regime_label': i % 8,  # 8 regime types
                        'historical_data': historical_data[:50],
                    }
                )

            # Train models
            training_success = market_analyzer.train_models(training_data)
            assert training_success
            print("✅ Model training completed successfully")

            # Check model performance
            assert hasattr(market_analyzer, 'model_performance')
            print("✅ Model performance tracking active")

            # Test 11: Test Integration with Market Context
            print("\n🔄 Testing Market Context Integration...")

            from core.context import MarketContext as LegacyMarketContext

            # Create legacy market context with advanced analyzer
            legacy_context = LegacyMarketContext(cfg)

            # Check if advanced market analyzer is initialized
            assert hasattr(legacy_context, 'advanced_analyzer')
            print("✅ Legacy Market Context integrated with Advanced Market Analyzer")

            # Test dynamic multipliers with advanced analysis
            multipliers = legacy_context.get_dynamic_multipliers(0.0025)  # 25 pips ATR
            assert "advanced_analysis" in multipliers
            print("✅ Advanced market analysis available in dynamic multipliers")

            # Test 12: Test Metrics Integration
            print("\n📊 Testing Metrics Integration...")

            from core.metricsx import observe_market_metric, observe_market_regime_change

            # Test metrics functions (they should not raise errors)
            observe_market_metric("test_metric", 0.75)
            observe_market_regime_change("low_volatility", "high_volatility", 0.85)
            print("✅ Metrics integration working")

            # Test 13: Test Market Summary
            print("\n📋 Testing Market Summary...")

            market_summary = market_analyzer.get_market_summary()
            assert isinstance(market_summary, dict)
            assert "total_analyses" in market_summary
            assert "regime_distribution" in market_summary

            print("✅ Market Summary Generated:")
            print(f"   Total Analyses: {market_summary['total_analyses']}")
            print(f"   Recent Analyses: {market_summary['recent_analyses']}")
            print(f"   Regime Distribution: {market_summary['regime_distribution']}")

            # Test 14: Test Cleanup
            print("\n🧹 Testing Cleanup...")

            market_analyzer.cleanup()
            print("✅ Advanced Market Analyzer cleanup completed")

            legacy_context.advanced_analyzer.cleanup() if legacy_context.advanced_analyzer else None
            print("✅ Legacy Market Context cleanup completed")

        # Test 15: Test Configuration File
        print("\n⚙️ Testing Configuration Integration...")

        # Check if advanced market config file exists
        market_config_path = Path("advanced_market_config.json")
        assert market_config_path.exists()
        print("✅ Advanced market configuration file exists")

        # Load and validate config
        with open(market_config_path) as f:
            config_data = json.load(f)

        assert "enable_ml" in config_data
        assert "enable_multi_timeframe" in config_data
        assert "enable_pattern_recognition" in config_data
        assert "market_regime" in config_data
        assert "multi_timeframe" in config_data
        assert "pattern_recognition" in config_data
        assert "ml_models" in config_data

        print("✅ Configuration validation passed")

        print("\n🎉 STEP14: Advanced Market Analysis System - COMPLETED SUCCESSFULLY!")
        print("\n📋 Summary of Advanced Market Analysis Capabilities:")
        print("✅ Enhanced Market Regime Detection with ML Integration")
        print("✅ Multi-Timeframe Analysis & Trend Alignment")
        print("✅ Advanced Pattern Recognition & Support/Resistance")
        print("✅ Volatility Analysis & Trend Strength Assessment")
        print("✅ Momentum Analysis & Market Context Integration")
        print("✅ ML Model Training & Performance Tracking")
        print("✅ Configuration Management & Metrics Integration")
        print("✅ Resource Management & Cleanup")

        return True

    except Exception as e:
        print(f"❌ STEP14 test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_step14()
    sys.exit(0 if success else 1)
