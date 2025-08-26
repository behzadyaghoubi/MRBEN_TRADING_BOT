"""
Comprehensive test runner for MR BEN Trading System.
"""

import os
import sys
import time
import traceback
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def discover_and_run_tests():
    """Discover and run all tests in the project."""
    print("🧪 MR BEN Trading System - Test Suite")
    print("=" * 60)

    # Get the project root directory
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"

    # Discover tests
    loader = unittest.TestLoader()

    # Unit tests
    unit_suite = loader.discover(
        str(tests_dir / "unit"), pattern="test_*.py", top_level_dir=str(project_root)
    )

    # Integration tests
    integration_suite = loader.discover(
        str(tests_dir / "integration"), pattern="test_*.py", top_level_dir=str(project_root)
    )

    # Root level tests
    root_suite = loader.discover(
        str(tests_dir), pattern="test_*.py", top_level_dir=str(project_root)
    )

    # Combine all test suites
    all_tests = unittest.TestSuite()
    all_tests.addTests(unit_suite)
    all_tests.addTests(integration_suite)
    all_tests.addTests(root_suite)

    # Count tests
    test_count = all_tests.countTestCases()
    print(f"📊 Discovered {test_count} test cases")
    print("📁 Test directories:")
    print(f"   - Unit tests: {tests_dir / 'unit'}")
    print(f"   - Integration tests: {tests_dir / 'integration'}")
    print(f"   - Root tests: {tests_dir}")
    print()

    # Run tests
    print("🚀 Starting test execution...")
    print()

    start_time = time.time()

    # Create test runner with verbosity
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)

    # Run tests
    try:
        result = runner.run(all_tests)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Print summary
        print()
        print("=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"✅ Tests run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"⚠️  Errors: {len(result.errors)}")
        print(f"⏭️  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

        # Print detailed results
        if result.failures:
            print()
            print("❌ FAILURES:")
            for test, traceback_str in result.failures:
                print(f"   - {test}: {traceback_str.split('AssertionError:')[-1].strip()}")

        if result.errors:
            print()
            print("⚠️  ERRORS:")
            for test, traceback_str in result.errors:
                print(f"   - {test}: {traceback_str.split('Exception:')[-1].strip()}")

        # Return exit code
        if result.wasSuccessful():
            print()
            print("🎉 All tests passed successfully!")
            return 0
        else:
            print()
            print("💥 Some tests failed. Please review the output above.")
            return 1

    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        traceback.print_exc()
        return 1


def run_specific_test_category(category):
    """Run tests from a specific category."""
    print(f"🎯 Running {category} tests...")
    print()

    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"

    loader = unittest.TestLoader()

    if category == "unit":
        suite = loader.discover(
            str(tests_dir / "unit"), pattern="test_*.py", top_level_dir=str(project_root)
        )
    elif category == "integration":
        suite = loader.discover(
            str(tests_dir / "integration"), pattern="test_*.py", top_level_dir=str(project_root)
        )
    elif category == "smoke":
        # Run only smoke tests
        suite = loader.discover(
            str(tests_dir), pattern="test_smoke.py", top_level_dir=str(project_root)
        )
    else:
        print(f"❌ Unknown test category: {category}")
        return 1

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


def run_quick_tests():
    """Run a quick subset of critical tests."""
    print("⚡ Running quick tests...")
    print()

    # Import and run specific quick tests
    try:
        from tests.test_smoke import TestSmoke
        from tests.unit.test_config import TestMT5Config

        # Create test suites
        smoke_suite = unittest.TestLoader().loadTestsFromTestCase(TestSmoke)
        config_suite = unittest.TestLoader().loadTestsFromTestCase(TestMT5Config)

        # Combine suites
        quick_suite = unittest.TestSuite()
        quick_suite.addTests(smoke_suite)
        quick_suite.addTests(config_suite)

        # Run tests
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(quick_suite)

        return 0 if result.wasSuccessful() else 1

    except Exception as e:
        print(f"💥 Quick tests failed: {e}")
        return 1


def main():
    """Main entry point for test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="MR BEN Trading System Test Runner")
    parser.add_argument(
        "--category",
        "-c",
        choices=["all", "unit", "integration", "smoke", "quick"],
        default="all",
        help="Test category to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TEST_VERBOSITY'] = '2'

    # Run tests based on category
    if args.category == "all":
        return discover_and_run_tests()
    elif args.category == "quick":
        return run_quick_tests()
    else:
        return run_specific_test_category(args.category)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
