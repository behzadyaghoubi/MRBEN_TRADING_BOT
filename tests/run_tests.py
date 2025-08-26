#!/usr/bin/env python3
"""
Test runner for MR BEN Trading Bot.
Runs BookStrategy tests with sample data.
"""

import sys
from pathlib import Path

# Add tests to path
sys.path.insert(0, str(Path(__file__).parent / "tests"))


def main():
    """Run the BookStrategy tests."""
    print("🧪 MR BEN Trading Bot - BookStrategy Test Runner")
    print("=" * 60)

    try:
        # Import and run tests
        from tests.test_book_strategy import run_comprehensive_test

        # Run the comprehensive test
        success = run_comprehensive_test()

        if success:
            print("\n🎉 All tests passed! BookStrategy is working correctly.")
            return 0
        else:
            print("\n❌ Some tests failed. Please check the output above.")
            return 1

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
