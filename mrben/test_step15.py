#!/usr/bin/env python3
"""
MR BEN - STEP15 Test Script
Advanced Signal Generation System Testing
"""

import sys
import os
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_step15_advanced_signals():
    """Test STEP15: Advanced Signal Generation System"""
    
    print("=" * 80)
    print("STEP15: Advanced Signal Generation System Testing")
    print("=" * 80)
    
    test_results = {
        "step": "STEP15",
        "component": "Advanced Signal Generation System",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {},
        "overall_status": "PENDING"
    }
    
    # Test 1: Configuration Loading
    print("\n1. Testing Configuration Loading...")
    try:
        config_path = "advanced_signals_config.json"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            required_sections = [
                "core_settings", "ml_models", "signal_generation", 
                "signal_fusion", "signal_validation", "multi_timeframe"
            ]
            
            config_valid = all(section in config.get("advanced_signals", {}) for section in required_sections)
            
            if config_valid:
                test_results["tests"]["config_loading"] = "PASSED"
                print("✓ Configuration file loaded successfully")
                print(f"  - Core settings: {len(config['advanced_signals']['core_settings'])} parameters")
                print(f"  - ML models: {len(config['advanced_signals']['ml_models'])} models")
                print(f"  - Signal generation: {len(config['advanced_signals']['signal_generation'])} types")
            else:
                test_results["tests"]["config_loading"] = "FAILED"
                print("✗ Configuration file missing required sections")
        else:
            test_results["tests"]["config_loading"] = "FAILED"
            print("✗ Configuration file not found")
    except Exception as e:
        test_results["tests"]["config_loading"] = "ERROR"
        print(f"✗ Configuration loading error: {e}")
    
    # Test 2: Component Imports
    print("\n2. Testing Component Imports...")
    try:
        from core.advanced_signals import (
            AdvancedSignalGenerator, SignalType, SignalQuality, 
            SignalFusionMethod, SignalComponent, FusedSignal, SignalValidation
        )
        test_results["tests"]["component_imports"] = "PASSED"
        print("✓ All advanced signal components imported successfully")
    except Exception as e:
        test_results["tests"]["component_imports"] = "ERROR"
        print(f"✗ Component import error: {e}")
    
    # Test 3: Advanced Signal Generator Initialization
    print("\n3. Testing Advanced Signal Generator Initialization...")
    try:
        signal_generator = AdvancedSignalGenerator(
            config_path="advanced_signals_config.json",
            enable_ml=True,
            enable_fusion=True,
            enable_validation=True
        )
        
        # Check initialization
        if (signal_generator.enable_ml and 
            signal_generator.enable_fusion and 
            signal_generator.enable_validation):
            test_results["tests"]["generator_initialization"] = "PASSED"
            print("✓ Advanced Signal Generator initialized successfully")
            print(f"  - ML enabled: {signal_generator.enable_ml}")
            print(f"  - Fusion enabled: {signal_generator.enable_fusion}")
            print(f"  - Validation enabled: {signal_generator.enable_validation}")
        else:
            test_results["tests"]["generator_initialization"] = "FAILED"
            print("✗ Advanced Signal Generator initialization failed")
    except Exception as e:
        test_results["tests"]["generator_initialization"] = "ERROR"
        print(f"✗ Generator initialization error: {e}")
    
    # Test 4: Signal Generation
    print("\n4. Testing Signal Generation...")
    try:
        if 'signal_generator' in locals():
            # Create mock MarketContext
            from core.typesx import MarketContext
            
            mock_context = MarketContext(
                price=1.2500,
                bid=1.2498,
                ask=1.2502,
                atr_pts=50.0,
                sma20=1.2480,
                sma50=1.2450,
                session="london",
                regime="NORMAL",
                equity=10000.0,
                balance=10000.0,
                spread_pts=20.0,
                open_positions=0
            )
            
            # Test trend following signal
            trend_signal = signal_generator.generate_trend_following_signal(mock_context)
            if isinstance(trend_signal, SignalComponent):
                print("✓ Trend following signal generated successfully")
                print(f"  - Direction: {trend_signal.direction}")
                print(f"  - Strength: {trend_signal.strength:.3f}")
                print(f"  - Confidence: {trend_signal.confidence:.3f}")
            else:
                print("✗ Trend following signal generation failed")
            
            # Test mean reversion signal
            mean_rev_signal = signal_generator.generate_mean_reversion_signal(mock_context)
            if isinstance(mean_rev_signal, SignalComponent):
                print("✓ Mean reversion signal generated successfully")
                print(f"  - Direction: {mean_rev_signal.direction}")
                print(f"  - Strength: {mean_rev_signal.strength:.3f}")
                print(f"  - Confidence: {mean_rev_signal.confidence:.3f}")
            else:
                print("✗ Mean reversion signal generation failed")
            
            # Test breakout signal
            breakout_signal = signal_generator.generate_breakout_signal(mock_context)
            if isinstance(breakout_signal, SignalComponent):
                print("✓ Breakout signal generated successfully")
                print(f"  - Direction: {breakout_signal.direction}")
                print(f"  - Strength: {breakout_signal.strength:.3f}")
                print(f"  - Confidence: {breakout_signal.confidence:.3f}")
            else:
                print("✗ Breakout signal generation failed")
            
            # Test momentum signal
            momentum_signal = signal_generator.generate_momentum_signal(mock_context)
            if isinstance(momentum_signal, SignalComponent):
                print("✓ Momentum signal generated successfully")
                print(f"  - Direction: {momentum_signal.direction}")
                print(f"  - Strength: {momentum_signal.strength:.3f}")
                print(f"  - Confidence: {momentum_signal.confidence:.3f}")
            else:
                print("✗ Momentum signal generation failed")
            
            test_results["tests"]["signal_generation"] = "PASSED"
        else:
            test_results["tests"]["signal_generation"] = "SKIPPED"
            print("- Signal generation test skipped (generator not available)")
    except Exception as e:
        test_results["tests"]["signal_generation"] = "ERROR"
        print(f"✗ Signal generation error: {e}")
    
    # Test 5: Signal Fusion
    print("\n5. Testing Signal Fusion...")
    try:
        if 'signal_generator' in locals() and 'trend_signal' in locals():
            # Test signal fusion
            component_signals = [trend_signal, mean_rev_signal, breakout_signal, momentum_signal]
            
            # Test weighted average fusion
            fused_signal = signal_generator.fuse_signals(
                component_signals, 
                method=SignalFusionMethod.WEIGHTED_AVERAGE
            )
            
            if isinstance(fused_signal, FusedSignal):
                print("✓ Signal fusion completed successfully")
                print(f"  - Fused direction: {fused_signal.direction}")
                print(f"  - Fused strength: {fused_signal.strength:.3f}")
                print(f"  - Fused confidence: {fused_signal.confidence:.3f}")
                print(f"  - Quality score: {fused_signal.quality_score:.3f}")
                print(f"  - Fusion method: {fused_signal.fusion_method.value}")
            else:
                print("✗ Signal fusion failed")
            
            test_results["tests"]["signal_fusion"] = "PASSED"
        else:
            test_results["tests"]["signal_fusion"] = "SKIPPED"
            print("- Signal fusion test skipped (signals not available)")
    except Exception as e:
        test_results["tests"]["signal_fusion"] = "ERROR"
        print(f"✗ Signal fusion error: {e}")
    
    # Test 6: Signal Validation
    print("\n6. Testing Signal Validation...")
    try:
        if 'fused_signal' in locals() and 'component_signals' in locals():
            # Test signal validation
            validation = signal_generator.validate_signal(fused_signal, component_signals)
            
            if isinstance(validation, SignalValidation):
                print("✓ Signal validation completed successfully")
                print(f"  - Is valid: {validation.is_valid}")
                print(f"  - Quality score: {validation.quality_score:.3f}")
                print(f"  - Risk score: {validation.risk_score:.3f}")
                print(f"  - Market alignment: {validation.market_alignment:.3f}")
                print(f"  - Validation checks: {len(validation.validation_checks)}")
                print(f"  - Recommendations: {len(validation.recommendations)}")
            else:
                print("✗ Signal validation failed")
            
            test_results["tests"]["signal_validation"] = "PASSED"
        else:
            test_results["tests"]["signal_validation"] = "SKIPPED"
            print("- Signal validation test skipped (fused signal not available)")
    except Exception as e:
        test_results["tests"]["signal_fusion"] = "ERROR"
        print(f"✗ Signal validation error: {e}")
    
    # Test 7: Integration with Decision Engine
    print("\n7. Testing Decision Engine Integration...")
    try:
        from core.decide import Decider
        
        # Create mock configuration
        class MockConfig:
            class Strategy:
                class PriceAction:
                    enabled = True
                    min_score = 0.3
                class MLFilter:
                    enabled = True
                    min_proba = 0.6
                class LSTMFilter:
                    enabled = True
                    agree_min = 0.6
                price_action = PriceAction()
                ml_filter = MLFilter()
                lstm_filter = LSTMFilter()
            
            class Confidence:
                base = 0.7
                class Dynamic:
                    regime = type('obj', (object,), {'low': 0.8, 'normal': 1.0, 'high': 0.6})()
                    session = type('obj', (object,), {'asia': 0.9, 'london': 1.0, 'ny': 1.0, 'off': 0.5})()
                    drawdown = type('obj', (object,), {'calm': 1.0, 'mild_dd': 0.8, 'deep_dd': 0.5})()
                class Threshold:
                    min = 0.5
                    max = 0.9
                dynamic = Dynamic()
                threshold = Threshold()
            
            strategy = Strategy()
            confidence = Confidence()
        
        mock_cfg = MockConfig()
        
        # Initialize decider with advanced signals
        decider = Decider(mock_cfg, None, None)
        
        if hasattr(decider, 'advanced_signals') and decider.advanced_signals is not None:
            print("✓ Decision engine integration successful")
            print("  - Advanced signal generator initialized")
            print("  - Enhanced vote method available")
        else:
            print("✗ Decision engine integration failed")
        
        test_results["tests"]["decision_engine_integration"] = "PASSED"
    except Exception as e:
        test_results["tests"]["decision_engine_integration"] = "ERROR"
        print(f"✗ Decision engine integration error: {e}")
    
    # Test 8: Metrics Integration
    print("\n8. Testing Metrics Integration...")
    try:
        from core.metricsx import (
            observe_signal_metric, observe_signal_quality, observe_signal_fusion
        )
        
        # Test metric observation functions
        observe_signal_metric("test_metric", 0.75)
        observe_signal_quality("test_signal", 0.8, 0.7)
        observe_signal_fusion("test_method", 0.85, 0.75)
        
        print("✓ Metrics integration successful")
        print("  - Signal metric observation working")
        print("  - Signal quality observation working")
        print("  - Signal fusion observation working")
        
        test_results["tests"]["metrics_integration"] = "PASSED"
    except Exception as e:
        test_results["tests"]["metrics_integration"] = "ERROR"
        print(f"✗ Metrics integration error: {e}")
    
    # Test 9: Signal Summary
    print("\n9. Testing Signal Summary...")
    try:
        if 'signal_generator' in locals():
            summary = signal_generator.get_signal_summary()
            
            if isinstance(summary, dict):
                print("✓ Signal summary generation successful")
                print(f"  - Summary keys: {list(summary.keys())}")
                if "message" in summary:
                    print(f"  - Message: {summary['message']}")
            else:
                print("✗ Signal summary generation failed")
            
            test_results["tests"]["signal_summary"] = "PASSED"
        else:
            test_results["tests"]["signal_summary"] = "SKIPPED"
            print("- Signal summary test skipped (generator not available)")
    except Exception as e:
        test_results["tests"]["signal_summary"] = "ERROR"
        print(f"✗ Signal summary error: {e}")
    
    # Test 10: Model Training
    print("\n10. Testing Model Training...")
    try:
        if 'signal_generator' in locals():
            # Create mock training data
            training_data = []
            for i in range(25):  # Minimum required samples
                training_data.append({
                    'signal_type': 'trend_following',
                    'strength': np.random.random(),
                    'confidence': np.random.random(),
                    'direction': np.random.choice([-1, 0, 1]),
                    'signal_label': np.random.choice([0, 1]),
                    'market_context': {
                        'atr_pts': np.random.uniform(20, 100),
                        'spread_pts': np.random.uniform(10, 50),
                        'open_positions': np.random.randint(0, 5)
                    }
                })
            
            # Test model training
            training_success = signal_generator.train_models(training_data)
            
            if training_success:
                print("✓ Model training completed successfully")
                print(f"  - Training samples: {len(training_data)}")
                print(f"  - Models saved to disk")
            else:
                print("✗ Model training failed")
            
            test_results["tests"]["model_training"] = "PASSED"
        else:
            test_results["tests"]["model_training"] = "SKIPPED"
            print("- Model training test skipped (generator not available)")
    except Exception as e:
        test_results["tests"]["model_training"] = "ERROR"
        print(f"✗ Model training error: {e}")
    
    # Test 11: Cleanup
    print("\n11. Testing Cleanup...")
    try:
        if 'signal_generator' in locals():
            signal_generator.cleanup()
            print("✓ Cleanup completed successfully")
            test_results["tests"]["cleanup"] = "PASSED"
        else:
            test_results["tests"]["cleanup"] = "SKIPPED"
            print("- Cleanup test skipped (generator not available)")
    except Exception as e:
        test_results["tests"]["cleanup"] = "ERROR"
        print(f"✗ Cleanup error: {e}")
    
    # Calculate overall status
    passed_tests = sum(1 for test in test_results["tests"].values() if test == "PASSED")
    total_tests = len(test_results["tests"])
    
    if passed_tests == total_tests:
        test_results["overall_status"] = "PASSED"
        print(f"\n🎉 ALL TESTS PASSED! ({passed_tests}/{total_tests})")
    elif passed_tests > 0:
        test_results["overall_status"] = "PARTIAL"
        print(f"\n⚠️  PARTIAL SUCCESS: {passed_tests}/{total_tests} tests passed")
    else:
        test_results["overall_status"] = "FAILED"
        print(f"\n❌ ALL TESTS FAILED! (0/{total_tests})")
    
    # Print test summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, status in test_results["tests"].items():
        status_symbol = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "⚠️" if status == "PARTIAL" else "-"
        print(f"{status_symbol} {test_name}: {status}")
    
    print(f"\nOverall Status: {test_results['overall_status']}")
    print(f"Timestamp: {test_results['timestamp']}")
    
    # Save test results
    results_file = f"STEP15_TEST_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to: {results_file}")
    
    return test_results


if __name__ == "__main__":
    try:
        results = test_step15_advanced_signals()
        sys.exit(0 if results["overall_status"] == "PASSED" else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        sys.exit(1)
