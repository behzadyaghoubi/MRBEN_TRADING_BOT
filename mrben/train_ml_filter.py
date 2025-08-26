#!/usr/bin/env python3
"""
MR BEN - ML Filter Training Script
Trains a machine learning model to filter trading signals
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.utils import save_model
import os
from pathlib import Path

from features.featurize import build_features
from core.loggingx import setup_logging

def generate_synthetic_data(n_samples: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for ML filter"""
    np.random.seed(42)
    
    # Generate random OHLCV data
    base_price = 100.0
    prices = []
    volumes = []
    
    for i in range(n_samples):
        # Random price movement
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        base_price *= (1 + change)
        
        # Generate OHLC from base price
        high = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low = base_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = base_price * (1 + np.random.normal(0, 0.005))
        close = base_price
        
        prices.append([open_price, high, low, close])
        volumes.append(np.random.randint(1000, 10000))
    
    # Create DataFrame
    df = pd.DataFrame(prices, columns=['O', 'H', 'L', 'C'])
    df['V'] = volumes
    
    # Generate labels (1 for good signal, 0 for bad signal)
    # Simple rule: if price increased and volume is above average, it's a good signal
    price_changes = df['C'].pct_change()
    volume_avg = df['V'].rolling(20).mean()
    
    labels = ((price_changes > 0.01) & (df['V'] > volume_avg)).astype(int)
    labels = labels.fillna(0)
    
    # Build features
    features = build_features(df)
    
    print(f"Debug: features shape: {features.shape}, labels length: {len(labels)}")
    
    # Remove rows with NaN features
    valid_mask = ~np.isnan(features).any(axis=1)
    print(f"Debug: valid_mask length: {len(valid_mask)}")
    
    # Ensure labels and features have matching lengths
    if len(features) != len(labels):
        print(f"Warning: Length mismatch. Truncating labels to match features.")
        labels = labels[:len(features)]
    
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    return features, labels

def train_ml_filter(features: np.ndarray, labels: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42) -> RandomForestClassifier:
    """Train the ML filter model"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print("\n=== Model Performance ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature count: {features.shape[1]}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== Feature Importance (Top 10) ===")
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(features.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))
    
    return model

def export_to_onnx(model: RandomForestClassifier, output_path: str):
    """Export scikit-learn model to ONNX format"""
    
    # Create initial type - use the correct format for skl2onnx
    initial_type = [('float_input', FloatTensorType([None, model.n_features_in_]))]
    
    # Convert to ONNX
    onx = onnxmltools.convert_sklearn(model, initial_types=initial_type)
    
    # Save model
    save_model(onx, output_path)
    print(f"\n✅ Model exported to ONNX: {output_path}")

def main():
    """Main training function"""
    logger = setup_logging("INFO")
    logger.info("Starting ML Filter training...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data
    logger.info("Generating synthetic training data...")
    features, labels = generate_synthetic_data(n_samples=15000)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {len(features)}")
    print(f"Features per sample: {features.shape[1]}")
    print(f"Positive labels: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    print(f"Negative labels: {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
    
    # Train model
    logger.info("Training ML filter model...")
    model = train_ml_filter(features, labels)
    
    # Export to ONNX
    output_path = models_dir / "ml_filter_v1.onnx"
    logger.info(f"Exporting model to ONNX: {output_path}")
    export_to_onnx(model, str(output_path))
    
    # Also save as joblib for comparison
    joblib_path = models_dir / "ml_filter_v1.joblib"
    joblib.dump(model, joblib_path)
    logger.info(f"Model also saved as joblib: {joblib_path}")
    
    logger.info("ML Filter training completed successfully!")
    
    # Test the exported model
    logger.info("Testing exported ONNX model...")
    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(str(output_path))
        input_name = sess.get_inputs()[0].name
        
        # Test with a single sample
        test_input = features[:1].astype(np.float32)
        prediction = sess.run(None, {input_name: test_input})
        
        print(f"\n✅ ONNX model test successful!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {prediction[0].shape}")
        
    except Exception as e:
        logger.error(f"ONNX model test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
