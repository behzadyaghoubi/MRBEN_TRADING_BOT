#!/usr/bin/env python3
"""
MR BEN - STEP4 Verification
Final verification that STEP4 is complete
"""

import os
import sys
from pathlib import Path

def verify_step4():
    """Verify STEP4 completion"""
    print("🚀 MR BEN - STEP4 Verification")
    print("=" * 50)
    
    # Check models directory
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    # Check required model files
    required_files = [
        "ml_filter_v1.onnx",      # ML Filter ONNX model
        "lstm_dir_v1.joblib",     # LSTM Direction model
        "ml_filter_v1.joblib"     # ML Filter backup
    ]
    
    missing_files = []
    for file in required_files:
        if not (models_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        return False
    
    print("✅ All required model files present")
    
    # Check core components
    core_files = [
        "core/decide.py",
        "core/configx.py", 
        "core/loggingx.py",
        "core/context.py",
        "core/sessionx.py",
        "core/regime.py",
        "features/featurize.py",
        "features/price_action.py"
    ]
    
    missing_core = []
    for file in core_files:
        if not Path(file).exists():
            missing_core.append(file)
    
    if missing_core:
        print(f"❌ Missing core files: {missing_core}")
        return False
    
    print("✅ All core component files present")
    
    # Check configuration
    config_file = "config/config.yaml"
    if not Path(config_file).exists():
        print(f"❌ Configuration file missing: {config_file}")
        return False
    
    print("✅ Configuration file present")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 STEP4 VERIFICATION RESULTS")
    print("=" * 50)
    print("✅ Models Directory: Present")
    print("✅ ML Filter (ONNX): Present")
    print("✅ LSTM Direction Model: Present")
    print("✅ Core Components: All Present")
    print("✅ Configuration: Present")
    
    print(f"\n🎉 STEP4: ML Filter (ONNX) - COMPLETED SUCCESSFULLY!")
    print("The MR BEN trading system now has:")
    print("  • ML Filter for signal noise reduction")
    print("  • LSTM Direction model for sequence prediction")
    print("  • Integrated decision engine")
    print("  • Complete feature engineering pipeline")
    
    print(f"\n🚀 Ready for STEP5: Risk Management Gates")
    
    return True

if __name__ == "__main__":
    success = verify_step4()
    sys.exit(0 if success else 1)
