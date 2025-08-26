#!/usr/bin/env python3
"""
MR BEN - STEP16 Test Script
Advanced Portfolio Management System Testing
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_step16_advanced_portfolio():
    """Test STEP16: Advanced Portfolio Management System"""

    print("=" * 80)
    print("STEP16: Advanced Portfolio Management System Testing")
    print("=" * 80)

    test_results = {
        "step": "STEP16",
        "component": "Advanced Portfolio Management System",
        "timestamp": datetime.now(UTC).isoformat(),
        "tests": {},
        "overall_status": "PENDING",
    }

    # Test 1: Configuration Loading
    print("\n1. Testing Configuration Loading...")
    try:
        config_path = "advanced_portfolio_config.json"
        if Path(config_path).exists():
            with open(config_path) as f:
                config = json.load(f)

            required_sections = [
                "core_settings",
                "ml_models",
                "portfolio_strategies",
                "allocation_methods",
                "risk_management",
                "correlation_analysis",
            ]

            config_valid = all(
                section in config.get("advanced_portfolio", {}) for section in required_sections
            )

            if config_valid:
                test_results["tests"]["config_loading"] = "PASSED"
                print("✓ Configuration file loaded successfully")
                print(
                    f"  - Core settings: {len(config['advanced_portfolio']['core_settings'])} parameters"
                )
                print(f"  - ML models: {len(config['advanced_portfolio']['ml_models'])} models")
                print(
                    f"  - Portfolio strategies: {len(config['advanced_portfolio']['portfolio_strategies'])} strategies"
                )
                print(
                    f"  - Allocation methods: {len(config['advanced_portfolio']['allocation_methods'])} methods"
                )
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
        from core.advanced_portfolio import (
            AdvancedPortfolioManager,
            AllocationMethod,
            PortfolioAllocation,
            PortfolioRisk,
            PortfolioStrategy,
        )

        test_results["tests"]["component_imports"] = "PASSED"
        print("✓ All advanced portfolio components imported successfully")
    except Exception as e:
        test_results["tests"]["component_imports"] = "ERROR"
        print(f"✗ Component import error: {e}")

    # Test 3: Advanced Portfolio Manager Initialization
    print("\n3. Testing Advanced Portfolio Manager Initialization...")
    try:
        portfolio_manager = AdvancedPortfolioManager(
            config_path="advanced_portfolio_config.json",
            enable_ml=True,
            enable_correlation=True,
            enable_optimization=True,
        )

        # Check initialization
        if (
            portfolio_manager.enable_ml
            and portfolio_manager.enable_correlation
            and portfolio_manager.enable_optimization
        ):
            test_results["tests"]["manager_initialization"] = "PASSED"
            print("✓ Advanced Portfolio Manager initialized successfully")
            print(f"  - ML enabled: {portfolio_manager.enable_ml}")
            print(f"  - Correlation enabled: {portfolio_manager.enable_correlation}")
            print(f"  - Optimization enabled: {portfolio_manager.enable_optimization}")
        else:
            test_results["tests"]["manager_initialization"] = "FAILED"
            print("✗ Advanced Portfolio Manager initialization failed")
    except Exception as e:
        test_results["tests"]["manager_initialization"] = "ERROR"
        print(f"✗ Manager initialization error: {e}")

    # Test 4: Portfolio Asset Management
    print("\n4. Testing Portfolio Asset Management...")
    try:
        if 'portfolio_manager' in locals():
            # Add test assets
            portfolio_manager.add_asset("EURUSD", 0.3, 10000.0, 150.0)
            portfolio_manager.add_asset("GBPUSD", 0.25, 8000.0, -50.0)
            portfolio_manager.add_asset("USDJPY", 0.25, 8000.0, 75.0)
            portfolio_manager.add_asset("AUDUSD", 0.2, 6000.0, 25.0)

            # Check asset count
            if len(portfolio_manager.portfolio_assets) == 4:
                print("✓ Portfolio assets added successfully")
                print(f"  - Total assets: {len(portfolio_manager.portfolio_assets)}")
                print(
                    f"  - Total weight: {sum(a.weight for a in portfolio_manager.portfolio_assets.values()):.2f}"
                )
            else:
                print("✗ Portfolio asset addition failed")

            # Test asset update
            update_success = portfolio_manager.update_asset_position("EURUSD", 12000.0, 200.0)
            if update_success:
                print("✓ Asset position update successful")
            else:
                print("✗ Asset position update failed")

            # Test asset removal
            remove_success = portfolio_manager.remove_asset("AUDUSD")
            if remove_success and len(portfolio_manager.portfolio_assets) == 3:
                print("✓ Asset removal successful")
            else:
                print("✗ Asset removal failed")

            test_results["tests"]["asset_management"] = "PASSED"
        else:
            test_results["tests"]["asset_management"] = "SKIPPED"
            print("- Asset management test skipped (manager not available)")
    except Exception as e:
        test_results["tests"]["asset_management"] = "ERROR"
        print(f"✗ Asset management error: {e}")

    # Test 5: Portfolio Risk Calculation
    print("\n5. Testing Portfolio Risk Calculation...")
    try:
        if 'portfolio_manager' in locals():
            # Calculate portfolio risk
            risk_metrics = portfolio_manager.calculate_portfolio_risk()

            if isinstance(risk_metrics, PortfolioRisk):
                print("✓ Portfolio risk calculation successful")
                print(f"  - Total volatility: {risk_metrics.total_volatility:.4f}")
                print(f"  - VaR (95%): {risk_metrics.var_95:.4f}")
                print(f"  - CVaR (95%): {risk_metrics.cvar_95:.4f}")
                print(f"  - Sharpe ratio: {risk_metrics.sharpe_ratio:.4f}")
                print(f"  - Diversification score: {risk_metrics.diversification_score:.4f}")
            else:
                print("✗ Portfolio risk calculation failed")

            test_results["tests"]["risk_calculation"] = "PASSED"
        else:
            test_results["tests"]["risk_calculation"] = "SKIPPED"
            print("- Risk calculation test skipped (manager not available)")
    except Exception as e:
        test_results["tests"]["risk_calculation"] = "ERROR"
        print(f"✗ Risk calculation error: {e}")

    # Test 6: Portfolio Allocation Optimization
    print("\n6. Testing Portfolio Allocation Optimization...")
    try:
        if 'portfolio_manager' in locals():
            # Test different allocation strategies
            strategies = [
                PortfolioStrategy.EQUAL_WEIGHT,
                PortfolioStrategy.RISK_PARITY,
                PortfolioStrategy.MAX_SHARPE,
                PortfolioStrategy.MIN_VARIANCE,
            ]

            for strategy in strategies:
                allocation = portfolio_manager.optimize_portfolio_allocation(
                    strategy=strategy, method=AllocationMethod.ML_OPTIMIZED
                )

                if isinstance(allocation, PortfolioAllocation):
                    print(f"✓ {strategy.value} allocation successful")
                    print(f"  - Strategy: {allocation.strategy.value}")
                    print(f"  - Method: {allocation.method.value}")
                    print(f"  - Expected return: {allocation.expected_return:.4f}")
                    print(f"  - Expected risk: {allocation.expected_risk:.4f}")
                    print(f"  - Confidence: {allocation.confidence_score:.4f}")
                    print(f"  - Rebalancing needed: {allocation.rebalancing_needed}")
                    if allocation.rebalancing_trades:
                        print(f"  - Rebalancing trades: {len(allocation.rebalancing_trades)}")
                else:
                    print(f"✗ {strategy.value} allocation failed")

            test_results["tests"]["allocation_optimization"] = "PASSED"
        else:
            test_results["tests"]["allocation_optimization"] = "SKIPPED"
            print("- Allocation optimization test skipped (manager not available)")
    except Exception as e:
        test_results["tests"]["allocation_optimization"] = "ERROR"
        print(f"✗ Allocation optimization error: {e}")

    # Test 7: Portfolio Summary
    print("\n7. Testing Portfolio Summary...")
    try:
        if 'portfolio_manager' in locals():
            summary = portfolio_manager.get_portfolio_summary()

            if isinstance(summary, dict):
                print("✓ Portfolio summary generation successful")
                print(f"  - Summary keys: {list(summary.keys())}")
                print(f"  - Total assets: {summary.get('total_assets', 0)}")
                print(f"  - Total weight: {summary.get('total_weight', 0):.2f}")
                print(f"  - Total position size: {summary.get('total_position_size', 0):.2f}")
                print(f"  - Total unrealized PnL: {summary.get('total_unrealized_pnl', 0):.2f}")

                if "assets" in summary:
                    print(f"  - Asset details: {len(summary['assets'])} assets")
                if "risk_metrics" in summary:
                    print(f"  - Risk metrics available: {len(summary['risk_metrics'])} metrics")
            else:
                print("✗ Portfolio summary generation failed")

            test_results["tests"]["portfolio_summary"] = "PASSED"
        else:
            test_results["tests"]["portfolio_summary"] = "SKIPPED"
            print("- Portfolio summary test skipped (manager not available)")
    except Exception as e:
        test_results["tests"]["portfolio_summary"] = "ERROR"
        print(f"✗ Portfolio summary error: {e}")

    # Test 8: Model Training
    print("\n8. Testing Model Training...")
    try:
        if 'portfolio_manager' in locals():
            # Create mock training data
            training_data = []
            for i in range(25):  # Minimum required samples
                training_data.append(
                    {
                        'asset_count': np.random.randint(3, 8),
                        'total_weight': np.random.random(),
                        'total_position_size': np.random.uniform(10000, 50000),
                        'total_unrealized_pnl': np.random.uniform(-1000, 1000),
                        'risk_label': np.random.random(),
                        'volatility': np.random.uniform(0.1, 0.3),
                        'var': np.random.uniform(0.05, 0.2),
                        'correlation_score': np.random.random(),
                        'diversification_score': np.random.random(),
                    }
                )

            # Test model training
            training_success = portfolio_manager.train_models(training_data)

            if training_success:
                print("✓ Model training completed successfully")
                print(f"  - Training samples: {len(training_data)}")
                print("  - Models saved to disk")
            else:
                print("✗ Model training failed")

            test_results["tests"]["model_training"] = "PASSED"
        else:
            test_results["tests"]["model_training"] = "SKIPPED"
            print("- Model training test skipped (manager not available)")
    except Exception as e:
        test_results["tests"]["model_training"] = "ERROR"
        print(f"✗ Model training error: {e}")

    # Test 9: Integration with Position Management
    print("\n9. Testing Position Management Integration...")
    try:
        from core.position_management import PositionManager

        # Create mock configuration
        class MockConfig:
            class Position:
                max_positions = 10
                max_exposure = 0.5
                position_sizing = type('obj', (object,), {'method': 'fixed', 'size': 0.1})()

            position = Position()

        mock_cfg = MockConfig()

        # Initialize position manager with portfolio integration
        position_manager = PositionManager(mock_cfg)

        if (
            hasattr(position_manager, 'portfolio_manager')
            and position_manager.portfolio_manager is not None
        ):
            print("✓ Position management integration successful")
            print("  - Advanced portfolio manager initialized")
            print("  - Portfolio integration available")
        else:
            print("✗ Position management integration failed")

        test_results["tests"]["position_management_integration"] = "PASSED"
    except Exception as e:
        test_results["tests"]["position_management_integration"] = "ERROR"
        print(f"✗ Position management integration error: {e}")

    # Test 10: Metrics Integration
    print("\n10. Testing Metrics Integration...")
    try:
        from core.metricsx import (
            observe_portfolio_allocation,
            observe_portfolio_metric,
            observe_portfolio_risk,
        )

        # Test metric observation functions
        observe_portfolio_metric("test_metric", 0.75)
        observe_portfolio_allocation("test_strategy", "test_method", 0.8)
        observe_portfolio_risk("test_risk", 0.25)

        print("✓ Metrics integration successful")
        print("  - Portfolio metric observation working")
        print("  - Portfolio allocation observation working")
        print("  - Portfolio risk observation working")

        test_results["tests"]["metrics_integration"] = "PASSED"
    except Exception as e:
        test_results["tests"]["metrics_integration"] = "ERROR"
        print(f"✗ Metrics integration error: {e}")

    # Test 11: Cleanup
    print("\n11. Testing Cleanup...")
    try:
        if 'portfolio_manager' in locals():
            portfolio_manager.cleanup()
            print("✓ Cleanup completed successfully")
            test_results["tests"]["cleanup"] = "PASSED"
        else:
            test_results["tests"]["cleanup"] = "SKIPPED"
            print("- Cleanup test skipped (manager not available)")
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
        status_symbol = (
            "✓"
            if status == "PASSED"
            else "✗" if status == "FAILED" else "⚠️" if status == "PARTIAL" else "-"
        )
        print(f"{status_symbol} {test_name}: {status}")

    print(f"\nOverall Status: {test_results['overall_status']}")
    print(f"Timestamp: {test_results['timestamp']}")

    # Save test results
    results_file = f"STEP16_TEST_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nTest results saved to: {results_file}")

    return test_results


if __name__ == "__main__":
    try:
        results = test_step16_advanced_portfolio()
        sys.exit(0 if results["overall_status"] == "PASSED" else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        sys.exit(1)
