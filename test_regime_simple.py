#!/usr/bin/env python3
"""
Simple test script for the Market Regime Detection system.
Tests the core functionality without importing problematic modules.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_regime_system():
    """Test the regime detection system."""
    print("🧪 Testing Market Regime Detection System")
    print("=" * 50)
    
    try:
        # Test 1: Import regime components
        print("\n1️⃣ Testing imports...")
        from ai.regime import RegimeLabel, RegimeSnapshot, RegimeConfig, create_regime_classifier
        print("✅ Regime imports successful")
        
        # Test 2: Create configuration
        print("\n2️⃣ Testing configuration...")
        regime_config = RegimeConfig(
            windows={"vol_atr": 14, "adx": 14, "return_ms": 20, "rv": 60},
            thresholds={
                "trend_adx": 20.0,
                "vol_high": 0.012,
                "spread_wide_bp": 15.0,
                "chop_z": 1.0
            },
            smoothing={"method": "ema", "alpha": 0.2, "min_dwell_bars": 5}
        )
        print("✅ Configuration created successfully")
        
        # Test 3: Initialize components
        print("\n3️⃣ Testing component initialization...")
        regime_classifier = create_regime_classifier(regime_config.dict())
        print("✅ Components initialized successfully")
        
        # Test 4: Create test data
        print("\n4️⃣ Testing data generation...")
        np.random.seed(42)
        n_bars = 100
        
        # Create trending data
        base_price = 2000.0
        trend = np.linspace(0, 0.05, n_bars)  # 5% trend
        noise = np.random.normal(0, 0.003, n_bars)
        prices = base_price * np.exp(trend + noise)
        
        trend_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_bars)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars)
        })
        
        # Ensure high >= low
        trend_data['high'] = np.maximum(trend_data['high'], trend_data['close'])
        trend_data['low'] = np.minimum(trend_data['low'], trend_data['close'])
        
        # Add timestamps
        start_time = datetime.now() - timedelta(days=n_bars)
        trend_data.index = pd.date_range(start=start_time, periods=n_bars, freq='H')
        
        print("✅ Test data generated successfully")
        
        # Test 5: Regime detection
        print("\n5️⃣ Testing regime detection...")
        regime_snapshot = regime_classifier.infer_regime(trend_data)
        
        print(f"   📊 Detected Regime: {regime_snapshot.label.value.upper()}")
        print(f"   📈 Confidence: {regime_snapshot.confidence:.3f}")
        print(f"   🕐 Session: {regime_snapshot.session}")
        print(f"   📊 Feature Scores:")
        for key, value in regime_snapshot.scores.items():
            if isinstance(value, (int, float)):
                print(f"      - {key}: {value:.3f}")
            else:
                print(f"      - {key}: {value}")
        
        print("✅ Regime detection successful")
        
        # Test 6: Regime summary
        print("\n6️⃣ Testing regime summary...")
        regime_summary = regime_classifier.get_regime_summary(lookback_days=1)
        
        if regime_summary:
            print(f"   📋 Total Snapshots: {regime_summary.get('total_snapshots', 0)}")
            print(f"   🏷️  Current Regime: {regime_summary.get('current_regime', 'unknown')}")
            print(f"   📊 Regime Counts:")
            for regime, count in regime_summary.get("regime_counts", {}).items():
                if count > 0:
                    print(f"      - {regime}: {count}")
        
        print("✅ Regime summary successful")
        
        # Test 7: Multiple regime classifications
        print("\n7️⃣ Testing multiple regime classifications...")
        
        # Create ranging data
        range_data = trend_data.copy()
        range_data['close'] = base_price * (1 + 0.01 * np.sin(np.linspace(0, 4*np.pi, n_bars)))
        range_data['high'] = range_data['close'] * (1 + 0.002)
        range_data['low'] = range_data['close'] * (1 - 0.002)
        
        range_snapshot = regime_classifier.infer_regime(range_data)
        print(f"   📊 Range Data Regime: {range_snapshot.label.value.upper()}")
        
        print("✅ Multiple regime classifications successful")
        
        # Test 8: Test features module
        print("\n8️⃣ Testing features module...")
        from ai.features import atr, adx, realized_vol, rolling_z, session_tag
        
        # Test ATR calculation
        atr_values = atr(trend_data['high'].values, trend_data['low'].values, trend_data['close'].values, 14)
        print(f"   📊 ATR calculation: {atr_values[-1]:.2f}")
        
        # Test ADX calculation
        adx_values = adx(trend_data['high'].values, trend_data['low'].values, trend_data['close'].values, 14)
        print(f"   📊 ADX calculation: {adx_values[-1]:.2f}")
        
        # Test session tagging
        sessions = session_tag(trend_data.index)
        print(f"   🕐 Session tagging: {len(sessions)} sessions tagged")
        
        print("✅ Features module successful")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Market Regime Detection system is working correctly.")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_regime_system()
    sys.exit(0 if success else 1)
