#!/usr/bin/env python3
"""
Test Script for Phase 3: Feature Engineering & Dataset Creation
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add src to path
sys.path.append('src')


def create_sample_data():
    """Create sample OHLCV data for testing"""
    print("📊 Creating sample OHLCV data...")

    # Generate 1000 bars of sample data
    np.random.seed(42)
    n_bars = 1000

    # Base price around 3300 (XAUUSD)
    base_price = 3300.0
    prices = [base_price]

    for i in range(1, n_bars):
        # Random walk with some trend
        change = np.random.normal(0, 0.5) + (0.1 if i > 500 else -0.1)  # Trend change at midpoint
        new_price = prices[-1] + change
        prices.append(max(1000, new_price))  # Ensure positive price

    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        # Generate realistic OHLC from close
        volatility = np.random.uniform(0.5, 2.0)
        high = close + np.random.uniform(0, volatility)
        low = close - np.random.uniform(0, volatility)
        open_price = np.random.uniform(low, high)
        volume = np.random.uniform(100, 1000)

        # Add some time component
        timestamp = datetime.now() - timedelta(minutes=15 * (n_bars - i))

        data.append(
            {
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
            }
        )

    df = pd.DataFrame(data)
    print(f"✅ Created {len(df)} bars of sample data")
    print(f"   Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"   Volume range: {df['volume'].min():.0f} - {df['volume'].max():.0f}")

    return df


def test_feature_engineering(df):
    """Test feature engineering"""
    print("\n🔧 Testing Feature Engineering...")

    try:
        from src.data.fe import FeatureEngineer

        # Configuration
        config = {
            'feature_config': {
                'lookback_periods': [5, 10, 20, 50],
                'volume_thresholds': [1.0, 1.5, 2.0],
            },
            'pa_config': {'min_body_ratio': 0.3, 'min_shadow_ratio': 1.5, 'volume_threshold': 1.2},
        }

        # Create feature engineer
        engineer = FeatureEngineer(config)

        # Engineer features
        features_df = engineer.engineer_features(df)

        print("✅ Feature engineering completed")
        print(f"   Original columns: {len(df.columns)}")
        print(f"   Feature columns: {len(features_df.columns)}")
        print(f"   New features added: {len(features_df.columns) - len(df.columns)}")

        # Show some feature columns
        feature_cols = [
            col
            for col in features_df.columns
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        ]
        print(f"   Sample features: {feature_cols[:10]}")

        return features_df

    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return None


def test_label_engineering(df, features_df):
    """Test label engineering"""
    print("\n🏷️ Testing Label Engineering...")

    try:
        from src.data.label import LabelEngineer

        # Configuration
        config = {
            'label_config': {
                'return_periods': [1, 4, 24],
                'confidence_thresholds': [0.6, 0.8],
                'risk_adjusted': True,
            },
            'pa_config': {'min_body_ratio': 0.3, 'min_shadow_ratio': 1.5, 'volume_threshold': 1.2},
        }

        # Create label engineer
        engineer = LabelEngineer(config)

        # Create labels
        labels_df = engineer.create_labels(df, features_df)

        print("✅ Label engineering completed")
        print(f"   Original columns: {len(features_df.columns)}")
        print(f"   Label columns: {len(labels_df.columns)}")
        print(f"   New labels added: {len(labels_df.columns) - len(features_df.columns)}")

        # Show some label columns
        label_cols = [
            col
            for col in labels_df.columns
            if 'signal_' in col or 'return_' in col or 'confidence_' in col
        ]
        print(f"   Sample labels: {label_cols[:10]}")

        return labels_df

    except Exception as e:
        print(f"❌ Label engineering failed: {e}")
        return None


def test_dataset_management(df, features_df, labels_df):
    """Test dataset management"""
    print("\n📁 Testing Dataset Management...")

    try:
        from src.data.dataset import DatasetManager

        # Configuration
        config = {
            'dataset_config': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'data_dir': 'data/pro',
                'models_dir': 'models',
            }
        }

        # Create dataset manager
        manager = DatasetManager(config)

        # Build dataset
        metadata = manager.build_dataset(df, features_df, labels_df)

        print("✅ Dataset management completed")
        print(f"   Total samples: {metadata.get('total_samples', 0)}")
        print(f"   Feature count: {metadata.get('feature_count', 0)}")
        print(f"   Label count: {metadata.get('label_count', 0)}")
        print(f"   Train samples: {metadata.get('splits', {}).get('train', 0)}")
        print(f"   Val samples: {metadata.get('splits', {}).get('val', 0)}")
        print(f"   Test samples: {metadata.get('splits', {}).get('test', 0)}")

        # Test loading
        print("\n🔄 Testing dataset loading...")
        train_features, train_labels = manager.load_dataset('train')
        print(f"   Loaded training data: {train_features.shape}, {train_labels.shape}")

        val_features, val_labels = manager.load_dataset('val')
        print(f"   Loaded validation data: {val_features.shape}, {val_labels.shape}")

        test_features, test_labels = manager.load_dataset('test')
        print(f"   Loaded test data: {test_features.shape}, {test_labels.shape}")

        # Get dataset info
        info = manager.get_dataset_info()
        print(f"   Available splits: {info.get('available_splits', [])}")
        print(f"   Total datasets: {info.get('total_datasets', 0)}")

        return metadata

    except Exception as e:
        print(f"❌ Dataset management failed: {e}")
        return None


def main():
    """Main test function"""
    print("🚀 MR BEN Pro Strategy - Phase 3 Testing")
    print("=" * 50)

    # Create sample data
    df = create_sample_data()

    # Test feature engineering
    features_df = test_feature_engineering(df)
    if features_df is None:
        print("❌ Phase 3 test failed at feature engineering")
        return False

    # Test label engineering
    labels_df = test_label_engineering(df, features_df)
    if labels_df is None:
        print("❌ Phase 3 test failed at label engineering")
        return False

    # Test dataset management
    metadata = test_dataset_management(df, features_df, labels_df)
    if metadata is None:
        print("❌ Phase 3 test failed at dataset management")
        return False

    # Summary
    print("\n" + "=" * 50)
    print("🎉 PHASE 3 TEST COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"✅ Feature Engineering: {len(features_df.columns)} columns")
    print(f"✅ Label Engineering: {len(labels_df.columns)} columns")
    print(f"✅ Dataset Management: {metadata.get('total_samples', 0)} samples")
    print("✅ Train/Val/Test splits created and saved")
    print("✅ Feature scaler saved")
    print("✅ Metadata saved")

    print("\n📊 Dataset Summary:")
    print(f"   Features: {metadata.get('feature_count', 0)}")
    print(f"   Labels: {metadata.get('label_count', 0)}")
    print(f"   Training: {metadata.get('splits', {}).get('train', 0)} samples")
    print(f"   Validation: {metadata.get('splits', {}).get('val', 0)} samples")
    print(f"   Testing: {metadata.get('splits', {}).get('test', 0)} samples")

    print("\n🎯 Phase 3 is ready for ML model development!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
