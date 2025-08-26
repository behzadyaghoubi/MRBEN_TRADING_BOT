"""
Test smoke regime detection functionality.
"""

import subprocess
import sys
import os
import time
import logging

# Setup logging for tests
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def test_smoke_regime_runs():
    """Test that smoke command with regime detection runs successfully"""
    logger.info("🧪 Testing smoke regime detection...")
    
    cmd = [
        sys.executable, 
        "live_trader_clean.py", 
        "smoke", 
        "--minutes", "1", 
        "--symbol", "XAUUSD", 
        "--agent", 
        "--regime"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120  # 2 minute timeout
        )
        
        # Check return code
        assert result.returncode == 0, f"Smoke test failed with return code {result.returncode}"
        
        # Check output for regime detection
        output = result.stdout + result.stderr
        
        # Verify advanced regime detection is working
        assert "Advanced smoke test module loaded" in output, "Advanced smoke module not loaded"
        assert "Advanced regime detected:" in output, "No regime detection found in output"
        
        # Check for regime labels
        regime_labels = ["TREND", "RANGE", "HIGH_VOL", "ILLIQUID"]
        found_regimes = [label for label in regime_labels if label in output]
        assert len(found_regimes) > 0, f"No regime labels found. Expected one of {regime_labels}"
        
        # Check for confidence adjustments
        assert "adj_conf" in output or "adj_conf" in output, "No confidence adjustments found"
        
        logger.info("✅ Smoke regime test passed!")
        logger.info(f"Found regimes: {found_regimes}")
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Smoke test timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Smoke test failed: {e}")
        return False


def test_smoke_without_regime():
    """Test that smoke command works without regime detection"""
    logger.info("🧪 Testing smoke without regime detection...")
    
    cmd = [
        sys.executable, 
        "live_trader_clean.py", 
        "smoke", 
        "--minutes", "1", 
        "--symbol", "XAUUSD"
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        
        assert result.returncode == 0, f"Smoke test without regime failed"
        logger.info("✅ Smoke test without regime passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Smoke test without regime failed: {e}")
        return False


def test_smoke_agent_flag():
    """Test that smoke command works with agent flag"""
    logger.info("🧪 Testing smoke with agent flag...")
    
    cmd = [
        sys.executable, 
        "live_trader_clean.py", 
        "smoke", 
        "--minutes", "1", 
        "--symbol", "XAUUSD", 
        "--agent"
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        
        assert result.returncode == 0, f"Smoke test with agent failed"
        logger.info("✅ Smoke test with agent passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Smoke test with agent failed: {e}")
        return False


def main():
    """Run all smoke tests"""
    logger.info("🚀 Starting smoke regime detection tests...")
    
    tests = [
        test_smoke_without_regime,
        test_smoke_agent_flag,
        test_smoke_regime_runs,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            time.sleep(1)  # Brief pause between tests
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
    
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All smoke tests passed!")
        return 0
    else:
        logger.error("❌ Some smoke tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
