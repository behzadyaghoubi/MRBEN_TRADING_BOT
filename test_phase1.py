#!/usr/bin/env python3
"""
Phase 1 Test Script for MRBEN System
This script tests the system components and generates a PHASE1_REPORT
"""

import sys
from pathlib import Path

import requests

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_system_components():
    """Test system components and generate report"""
    report = []
    report.append("PHASE1_REPORT")
    report.append("=" * 50)

    # Test 1: Import SystemIntegrator
    try:
        from core.system_integrator import SystemIntegrator

        report.append("System Status: ✅ SystemIntegrator imported successfully")
        integrator_available = True
    except Exception as e:
        report.append(f"System Status: ❌ Import failed - {e}")
        integrator_available = False

    # Test 2: Check config loading
    try:
        from core.configx import load_config

        config = load_config("config/config.yaml")
        report.append("Config Status: ✅ Configuration loaded successfully")
        config_available = True
    except Exception as e:
        report.append(f"Config Status: ❌ Config failed - {e}")
        config_available = False

    # Test 3: Check metrics endpoint
    try:
        response = requests.get("http://127.0.0.1:8765/metrics", timeout=5)
        if response.status_code == 200:
            metrics_content = response.text
            report.append("Metrics Status: ✅ Port 8765 accessible")

            # Check for A/B tracks
            has_control = 'track="control"' in metrics_content
            has_pro = 'track="pro"' in metrics_content
            report.append(
                f"A/B Tracks seen?: {'Yes' if has_control and has_pro else 'No'} (control & pro)"
            )

            # Extract key metrics
            mrben_lines = [
                line for line in metrics_content.split('\n') if line.startswith('mrben_')
            ]
            report.append("Metrics (key lines):")
            for line in mrben_lines[:6]:
                report.append(f"  {line}")
        else:
            report.append(f"Metrics Status: ❌ HTTP {response.status_code}")
    except Exception as e:
        report.append(f"Metrics Status: ❌ Unreachable - {e}")

    # Test 4: Check logs for ensemble labels
    log_file = Path("logs/mrben.log")
    if log_file.exists():
        try:
            with open(log_file, encoding='utf-8') as f:
                log_content = f.read()

            # Check for ensemble labels
            ensemble_labels = ['[PA]', '[ML]', '[LSTM]', '[CONF]', '[VOTE]']
            found_labels = [label for label in ensemble_labels if label in log_content]

            if found_labels:
                report.append(f"Ensemble labels seen?: Yes ({', '.join(found_labels)})")
            else:
                report.append("Ensemble labels seen?: No ([PA/ML/LSTM/CONF/VOTE])")

            # Check for legacy mode
            if 'legacy' in log_content or 'SMA_Only' in log_content:
                report.append("Legacy Mode: ❌ Found legacy/SMA_Only references")
            else:
                report.append("Legacy Mode: ✅ No legacy references found")

            # Get recent log lines
            lines = log_content.split('\n')
            recent_lines = [
                line for line in lines[-20:] if any(label in line for label in ensemble_labels)
            ]
            if recent_lines:
                report.append("Logs (recent ensemble lines):")
                for line in recent_lines[-5:]:
                    report.append(f"  {line}")
            else:
                report.append("Logs: No recent ensemble labels found")

        except Exception as e:
            report.append(f"Log Analysis: ❌ Error reading logs - {e}")
    else:
        report.append("Logs: ❌ mrben.log not found")

    # Test 5: Check for errors
    if log_file.exists():
        try:
            with open(log_file, encoding='utf-8') as f:
                log_content = f.read()

            error_patterns = ['10030', '10018', 'error', 'ERROR']
            found_errors = []
            for pattern in error_patterns:
                if pattern in log_content:
                    found_errors.append(pattern)

            if found_errors:
                report.append(f"Errors: Found error patterns: {', '.join(found_errors)}")
            else:
                report.append("Errors: None found")

        except Exception as e:
            report.append(f"Error Check: ❌ Error analyzing logs - {e}")

    # Overall assessment
    report.append("\n" + "=" * 50)
    report.append("OVERALL ASSESSMENT:")

    if integrator_available and config_available:
        report.append("✅ System components available")
        report.append("✅ Configuration loaded")

        # Check if we can actually start the system
        try:
            integrator = SystemIntegrator("config/config.yaml")
            report.append("✅ SystemIntegrator initialized")
            report.append("System Status: Ready for Phase 1 testing")
        except Exception as e:
            report.append(f"❌ SystemIntegrator failed to initialize: {e}")
            report.append("System Status: Error - needs fixing")
    else:
        report.append("❌ Critical components missing")
        report.append("System Status: Not ready")

    return "\n".join(report)


if __name__ == "__main__":
    print("Testing MRBEN System for Phase 1...")
    print("=" * 50)

    report = test_system_components()

    # Save report
    with open("PHASE1_REPORT.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print("\n" + "=" * 50)
    print("PHASE1_REPORT saved to PHASE1_REPORT.txt")
