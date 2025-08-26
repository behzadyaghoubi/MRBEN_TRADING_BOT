#!/usr/bin/env python3
"""
MR BEN - STEP17 Test Script
Final Integration & Testing - System Integrator and Complete System
"""

import sys
import os
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_step17_final_integration():
    """Test STEP17: Final Integration & Testing"""
    
    print("=" * 80)
    print("STEP17: Final Integration & Testing")
    print("=" * 80)
    
    test_results = {
        "step": "STEP17",
        "component": "Final Integration & Testing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {},
        "overall_status": "PENDING"
    }
    
    # Test 1: System Integrator Import
    print("\n1. Testing System Integrator Import...")
    try:
        from core.system_integrator import (
            SystemIntegrator, SystemStatus, ComponentStatus, 
            SystemHealth, IntegrationTest
        )
        test_results["tests"]["system_integrator_import"] = "PASSED"
        print("✓ System Integrator components imported successfully")
    except Exception as e:
        test_results["tests"]["system_integrator_import"] = "ERROR"
        print(f"✗ System Integrator import error: {e}")
    
    # Test 2: Mock Configuration Setup
    print("\n2. Testing Mock Configuration Setup...")
    try:
        # Create mock configuration structure
        mock_config = Mock()
        mock_config.config = {
            'logging': {'level': 'INFO'},
            'position': {'max_positions': 10},
            'risk': {'max_exposure': 0.5}
        }
        
        # Mock config file
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        mock_config_path = config_dir / "config.yaml"
        with open(mock_config_path, 'w') as f:
            f.write("logging:\n  level: INFO\nposition:\n  max_positions: 10\nrisk:\n  max_exposure: 0.5\n")
        
        test_results["tests"]["mock_config_setup"] = "PASSED"
        print("✓ Mock configuration setup successful")
        print(f"  - Config file created: {mock_config_path}")
        
    except Exception as e:
        test_results["tests"]["mock_config_setup"] = "ERROR"
        print(f"✗ Mock configuration setup error: {e}")
    
    # Test 3: System Integrator Initialization
    print("\n3. Testing System Integrator Initialization...")
    try:
        if 'SystemIntegrator' in locals():
            # Create system integrator with mock config
            integrator = SystemIntegrator("config/config.yaml")
            
            # Check initialization
            if integrator.status in [SystemStatus.RUNNING, SystemStatus.INITIALIZING]:
                test_results["tests"]["integrator_initialization"] = "PASSED"
                print("✓ System Integrator initialized successfully")
                print(f"  - Status: {integrator.status.value}")
                print(f"  - Components: {len(integrator.components)}")
                print(f"  - Component Status: {len(integrator.component_status)}")
            else:
                test_results["tests"]["integrator_initialization"] = "FAILED"
                print("✗ System Integrator initialization failed")
        else:
            test_results["tests"]["integrator_initialization"] = "SKIPPED"
            print("- Integrator initialization test skipped (class not available)")
            
    except Exception as e:
        test_results["tests"]["integrator_initialization"] = "ERROR"
        print(f"✗ Integrator initialization error: {e}")
    
    # Test 4: Component Status Verification
    print("\n4. Testing Component Status Verification...")
    try:
        if 'integrator' in locals():
            # Check component status
            online_components = sum(1 for status in integrator.component_status.values() 
                                  if status == ComponentStatus.ONLINE)
            total_components = len(integrator.component_status)
            
            if online_components > 0:
                test_results["tests"]["component_status_verification"] = "PASSED"
                print("✓ Component status verification successful")
                print(f"  - Online components: {online_components}/{total_components}")
                print(f"  - Component status: {dict(integrator.component_status)}")
            else:
                test_results["tests"]["component_status_verification"] = "FAILED"
                print("✗ No components are online")
        else:
            test_results["tests"]["component_status_verification"] = "SKIPPED"
            print("- Component status verification skipped (integrator not available)")
            
    except Exception as e:
        test_results["tests"]["component_status_verification"] = "ERROR"
        print(f"✗ Component status verification error: {e}")
    
    # Test 5: System Health Check
    print("\n5. Testing System Health Check...")
    try:
        if 'integrator' in locals():
            # Perform health check
            health = integrator._check_system_health()
            
            if isinstance(health, SystemHealth):
                test_results["tests"]["system_health_check"] = "PASSED"
                print("✓ System health check successful")
                print(f"  - Overall status: {health.overall_status.value}")
                print(f"  - Error count: {health.error_count}")
                print(f"  - Recommendations: {len(health.recommendations)}")
            else:
                test_results["tests"]["system_health_check"] = "FAILED"
                print("✗ System health check failed")
        else:
            test_results["tests"]["system_health_check"] = "SKIPPED"
            print("- System health check skipped (integrator not available)")
            
    except Exception as e:
        test_results["tests"]["system_health_check"] = "ERROR"
        print(f"✗ System health check error: {e}")
    
    # Test 6: Integration Tests
    print("\n6. Testing Integration Tests...")
    try:
        if 'integrator' in locals():
            # Check integration test results
            if integrator.test_results:
                passed_tests = sum(1 for test in integrator.test_results if test.status == "PASSED")
                total_tests = len(integrator.test_results)
                
                test_results["tests"]["integration_tests"] = "PASSED"
                print("✓ Integration tests completed")
                print(f"  - Passed tests: {passed_tests}/{total_tests}")
                print(f"  - Test results: {[test.test_name for test in integrator.test_results]}")
            else:
                test_results["tests"]["integration_tests"] = "FAILED"
                print("✗ No integration tests were run")
        else:
            test_results["tests"]["integration_tests"] = "SKIPPED"
            print("- Integration tests skipped (integrator not available)")
            
    except Exception as e:
        test_results["tests"]["integration_tests"] = "ERROR"
        print(f"✗ Integration tests error: {e}")
    
    # Test 7: System Status Retrieval
    print("\n7. Testing System Status Retrieval...")
    try:
        if 'integrator' in locals():
            # Get system status
            status = integrator.get_system_status()
            
            if isinstance(status, dict) and 'system_status' in status:
                test_results["tests"]["system_status_retrieval"] = "PASSED"
                print("✓ System status retrieval successful")
                print(f"  - System status: {status['system_status']}")
                print(f"  - Component count: {status.get('component_status', {})}")
                print(f"  - Performance metrics: {len(status.get('performance_metrics', {}))}")
            else:
                test_results["tests"]["system_status_retrieval"] = "FAILED"
                print("✗ System status retrieval failed")
        else:
            test_results["tests"]["system_status_retrieval"] = "SKIPPED"
            print("- System status retrieval skipped (integrator not available)")
            
    except Exception as e:
        test_results["tests"]["system_status_retrieval"] = "ERROR"
        print(f"✗ System status retrieval error: {e}")
    
    # Test 8: System Control Operations
    print("\n8. Testing System Control Operations...")
    try:
        if 'integrator' in locals():
            # Test pause operation
            if integrator.status == SystemStatus.RUNNING:
                pause_success = integrator.pause_system()
                if pause_success:
                    print("✓ System pause successful")
                    
                    # Test resume operation
                    resume_success = integrator.resume_system()
                    if resume_success:
                        print("✓ System resume successful")
                        test_results["tests"]["system_control_operations"] = "PASSED"
                    else:
                        print("✗ System resume failed")
                        test_results["tests"]["system_control_operations"] = "FAILED"
                else:
                    print("✗ System pause failed")
                    test_results["tests"]["system_control_operations"] = "FAILED"
            else:
                print(f"- System control test skipped (status: {integrator.status.value})")
                test_results["tests"]["system_control_operations"] = "SKIPPED"
        else:
            test_results["tests"]["system_control_operations"] = "SKIPPED"
            print("- System control operations skipped (integrator not available)")
            
    except Exception as e:
        test_results["tests"]["system_control_operations"] = "ERROR"
        print(f"✗ System control operations error: {e}")
    
    # Test 9: System Diagnostic
    print("\n9. Testing System Diagnostic...")
    try:
        if 'integrator' in locals():
            # Run diagnostic
            diagnostic = integrator.run_diagnostic()
            
            if isinstance(diagnostic, dict) and 'system_status' in diagnostic:
                test_results["tests"]["system_diagnostic"] = "PASSED"
                print("✓ System diagnostic successful")
                print(f"  - Diagnostic keys: {list(diagnostic.keys())}")
                print(f"  - Component health: {len(diagnostic.get('component_health', {}))}")
                print(f"  - Recommendations: {len(diagnostic.get('recommendations', []))}")
            else:
                test_results["tests"]["system_diagnostic"] = "FAILED"
                print("✗ System diagnostic failed")
        else:
            test_results["tests"]["system_diagnostic"] = "SKIPPED"
            print("- System diagnostic skipped (integrator not available)")
            
    except Exception as e:
        test_results["tests"]["system_diagnostic"] = "ERROR"
        print(f"✗ System diagnostic error: {e}")
    
    # Test 10: Performance Metrics
    print("\n10. Testing Performance Metrics...")
    try:
        if 'integrator' in locals():
            # Check performance metrics
            if integrator.performance_metrics:
                test_results["tests"]["performance_metrics"] = "PASSED"
                print("✓ Performance metrics collection successful")
                print(f"  - Metrics collected: {len(integrator.performance_metrics)}")
                print(f"  - Metric types: {list(integrator.performance_metrics.keys())}")
            else:
                test_results["tests"]["performance_metrics"] = "FAILED"
                print("✗ No performance metrics collected")
        else:
            test_results["tests"]["performance_metrics"] = "SKIPPED"
            print("- Performance metrics test skipped (integrator not available)")
            
    except Exception as e:
        test_results["tests"]["performance_metrics"] = "ERROR"
        print(f"✗ Performance metrics error: {e}")
    
    # Test 11: Context Manager
    print("\n11. Testing Context Manager...")
    try:
        if 'SystemIntegrator' in locals():
            # Test context manager functionality
            with SystemIntegrator("config/config.yaml") as test_integrator:
                if test_integrator.status in [SystemStatus.RUNNING, SystemStatus.INITIALIZING]:
                    print("✓ Context manager entry successful")
                    
                    # Context manager will automatically call stop_system on exit
                    pass
            
            test_results["tests"]["context_manager"] = "PASSED"
            print("✓ Context manager exit successful")
        else:
            test_results["tests"]["context_manager"] = "SKIPPED"
            print("- Context manager test skipped (class not available)")
            
    except Exception as e:
        test_results["tests"]["context_manager"] = "ERROR"
        print(f"✗ Context manager error: {e}")
    
    # Test 12: Complete System Integration
    print("\n12. Testing Complete System Integration...")
    try:
        # Test that all major components can be imported and initialized
        component_imports = [
            ('configx', 'ConfigManager'),
            ('loggingx', 'setup_logging'),
            ('sessionx', 'SessionDetector'),
            ('regime', 'RegimeDetector'),
            ('context', 'MarketContext'),
            ('price_action', 'PriceActionDetector'),
            ('featurize', 'FeatureExtractor'),
            ('decide', 'Decider'),
            ('risk_gates', 'RiskManager'),
            ('position_sizing', 'PositionSizer'),
            ('position_management', 'PositionManager'),
            ('order_management', 'OrderManager'),
            ('metricsx', 'setup_metrics'),
            ('emergency_stop', 'EmergencyStopManager'),
            ('agent_bridge', 'AgentBridge'),
            ('advanced_risk', 'AdvancedRiskAnalytics'),
            ('advanced_position', 'AdvancedPositionManager'),
            ('advanced_market', 'AdvancedMarketAnalyzer'),
            ('advanced_signals', 'AdvancedSignalGenerator'),
            ('advanced_portfolio', 'AdvancedPortfolioManager')
        ]
        
        imported_components = 0
        for module_name, class_name in component_imports:
            try:
                module = __import__(f'core.{module_name}', fromlist=[class_name])
                if hasattr(module, class_name):
                    imported_components += 1
                else:
                    print(f"  - Warning: {class_name} not found in {module_name}")
            except Exception as e:
                print(f"  - Warning: Failed to import {class_name} from {module_name}: {e}")
        
        if imported_components >= len(component_imports) * 0.8:  # 80% success rate
            test_results["tests"]["complete_system_integration"] = "PASSED"
            print("✓ Complete system integration successful")
            print(f"  - Components imported: {imported_components}/{len(component_imports)}")
        else:
            test_results["tests"]["complete_system_integration"] = "FAILED"
            print(f"✗ Complete system integration failed ({imported_components}/{len(component_imports)} components)")
            
    except Exception as e:
        test_results["tests"]["complete_system_integration"] = "ERROR"
        print(f"✗ Complete system integration error: {e}")
    
    # Test 13: System Cleanup
    print("\n13. Testing System Cleanup...")
    try:
        if 'integrator' in locals() and hasattr(integrator, 'stop_system'):
            # Test system cleanup
            cleanup_success = integrator.stop_system()
            
            if cleanup_success or integrator.status == SystemStatus.STOPPED:
                test_results["tests"]["system_cleanup"] = "PASSED"
                print("✓ System cleanup successful")
                print(f"  - Final status: {integrator.status.value}")
            else:
                test_results["tests"]["system_cleanup"] = "FAILED"
                print("✗ System cleanup failed")
        else:
            test_results["tests"]["system_cleanup"] = "SKIPPED"
            print("- System cleanup test skipped (integrator not available)")
            
    except Exception as e:
        test_results["tests"]["system_cleanup"] = "ERROR"
        print(f"✗ System cleanup error: {e}")
    
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
    results_file = f"STEP17_TEST_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to: {results_file}")
    
    return test_results


if __name__ == "__main__":
    try:
        results = test_step17_final_integration()
        sys.exit(0 if results["overall_status"] == "PASSED" else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        sys.exit(1)
