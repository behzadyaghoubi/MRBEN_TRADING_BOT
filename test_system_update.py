#!/usr/bin/env python3
"""
Simple test script to verify MR BEN AI System update functionality
"""

import json
import os
from datetime import datetime

import pandas as pd


def test_system_components():
    """Test system components and generate a simple report"""
    print("🎯 MR BEN AI System - Component Test")
    print("=" * 50)

    results = {'data_files': {}, 'model_files': {}, 'system_status': {}}

    # Test data files
    data_files = [
        'data/XAUUSD_PRO_M5_live.csv',
        'data/XAUUSD_PRO_M5_enhanced.csv',
        'data/XAUUSD_PRO_M5_data.csv',
        'data/ohlc_data.csv',
    ]

    print("📊 Testing data files...")
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                results['data_files'][file_path] = {
                    'exists': True,
                    'rows': len(df),
                    'columns': list(df.columns),
                }
                print(f"✅ {file_path}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                results['data_files'][file_path] = {'exists': True, 'error': str(e)}
                print(f"❌ {file_path}: Error - {e}")
        else:
            results['data_files'][file_path] = {'exists': False}
            print(f"⚠️ {file_path}: Not found")

    # Test model files
    model_files = [
        'models/mrben_simple_model.joblib',
        'models/mrben_ai_signal_filter_xgb_balanced.joblib',
        'models/mrben_lstm_real_data.h5',
    ]

    print("\n🤖 Testing model files...")
    for file_path in model_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            results['model_files'][file_path] = {'exists': True, 'size_mb': round(file_size, 2)}
            print(f"✅ {file_path}: {round(file_size, 2)} MB")
        else:
            results['model_files'][file_path] = {'exists': False}
            print(f"⚠️ {file_path}: Not found")

    # Test system status
    print("\n🔧 Testing system status...")

    # Check Python packages
    try:
        import tensorflow as tf

        results['system_status']['tensorflow'] = f"✅ Available - {tf.__version__}"
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        results['system_status']['tensorflow'] = "❌ Not available"
        print("❌ TensorFlow: Not available")

    try:
        import sklearn

        results['system_status']['sklearn'] = f"✅ Available - {sklearn.__version__}"
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        results['system_status']['sklearn'] = "❌ Not available"
        print("❌ Scikit-learn: Not available")

    try:
        import joblib

        results['system_status']['joblib'] = "✅ Available"
        print("✅ Joblib: Available")
    except ImportError:
        results['system_status']['joblib'] = "❌ Not available"
        print("❌ Joblib: Not available")

    # Generate simple report
    print("\n📋 Generating test report...")

    report = f"""
============================================================
MR BEN AI SYSTEM - COMPONENT TEST REPORT
============================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: 4.1 - Test

DATA FILES STATUS:
==================
"""

    for file_path, status in results['data_files'].items():
        if status.get('exists'):
            if 'rows' in status:
                report += (
                    f"- {file_path}: ✅ {status['rows']} rows, {len(status['columns'])} columns\n"
                )
            else:
                report += f"- {file_path}: ❌ Error - {status.get('error', 'Unknown')}\n"
        else:
            report += f"- {file_path}: ⚠️ Not found\n"

    report += """
MODEL FILES STATUS:
===================
"""

    for file_path, status in results['model_files'].items():
        if status.get('exists'):
            report += f"- {file_path}: ✅ {status['size_mb']} MB\n"
        else:
            report += f"- {file_path}: ⚠️ Not found\n"

    report += """
SYSTEM STATUS:
==============
"""

    for component, status in results['system_status'].items():
        report += f"- {component}: {status}\n"

    report += """
RECOMMENDATIONS:
===============
"""

    # Count available components
    available_data = sum(
        1 for status in results['data_files'].values() if status.get('exists') and 'rows' in status
    )
    available_models = sum(1 for status in results['model_files'].values() if status.get('exists'))

    if available_data >= 2:
        report += "✅ Sufficient data files available for retraining\n"
    else:
        report += "❌ Insufficient data files for retraining\n"

    if available_models >= 2:
        report += "✅ Sufficient model files available for enhancement\n"
    else:
        report += "❌ Insufficient model files for enhancement\n"

    if all('✅' in status for status in results['system_status'].values()):
        report += "✅ All required packages are available\n"
    else:
        report += "❌ Some required packages are missing\n"

    report += """
NEXT STEPS:
===========
1. If all components are available, run the comprehensive update
2. If components are missing, install required packages or obtain missing files
3. Monitor system performance after update
4. Schedule regular maintenance and retraining

============================================================
Report generated by MR BEN AI System v4.1 - Test
============================================================
"""

    # Save report
    os.makedirs('logs', exist_ok=True)
    report_path = f'logs/system_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ Test report saved: {report_path}")

    # Save results as JSON
    results_path = f'logs/system_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Test results saved: {results_path}")

    return results


if __name__ == "__main__":
    test_system_components()
    print("\n✅ System test completed!")
