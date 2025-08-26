#!/usr/bin/env python3
"""
MR BEN - Fix XGBoost BUY Bias Script
====================================
This script fixes the BUY bias issue in XGBoost model only.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting XGBoost BUY Bias Fix Process...")
    print("=" * 50)
    
    # Step 1: Load and analyze data
    print("🔍 Loading and analyzing current dataset...")
    data = pd.read_csv('data/mrben_ai_signal_dataset.csv')
    
    # Analyze signal distribution
    signal_counts = data['signal'].value_counts()
    print(f"\n📊 Current Signal Distribution:")
    print(signal_counts)
    print(f"\nTotal signals: {len(data)}")
    
    # Calculate imbalance
    total = len(data)
    hold_pct = signal_counts.get('HOLD', 0) / total * 100
    buy_pct = signal_counts.get('BUY', 0) / total * 100
    sell_pct = signal_counts.get('SELL', 0) / total * 100
    
    print(f"\n📈 Distribution Percentages:")
    print(f"HOLD: {hold_pct:.1f}%")
    print(f"BUY: {buy_pct:.1f}%")
    print(f"SELL: {sell_pct:.1f}%")
    
    # Step 2: Create balanced dataset
    print(f"\n⚖️ Creating balanced dataset...")
    
    # Separate signals
    hold_data = data[data['signal'] == 'HOLD']
    buy_data = data[data['signal'] == 'BUY']
    sell_data = data[data['signal'] == 'SELL']
    
    print(f"Original counts - HOLD: {len(hold_data)}, BUY: {len(buy_data)}, SELL: {len(sell_data)}")
    
    # Calculate target counts (30% each for BUY/SELL)
    target_count = int(total * 0.3)
    
    # Balance BUY and SELL
    if len(buy_data) < target_count:
        buy_data_balanced = buy_data.sample(n=target_count, replace=True, random_state=42)
    else:
        buy_data_balanced = buy_data.sample(n=target_count, random_state=42)
        
    if len(sell_data) < target_count:
        sell_data_balanced = sell_data.sample(n=target_count, replace=True, random_state=42)
    else:
        sell_data_balanced = sell_data.sample(n=target_count, random_state=42)
    
    # Adjust HOLD data
    hold_target = total - len(buy_data_balanced) - len(sell_data_balanced)
    if len(hold_data) > hold_target:
        hold_data_balanced = hold_data.sample(n=hold_target, random_state=42)
    else:
        hold_data_balanced = hold_data.sample(n=hold_target, replace=True, random_state=42)
    
    # Combine balanced data
    balanced_data = pd.concat([hold_data_balanced, buy_data_balanced, sell_data_balanced])
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify balance
    balanced_counts = balanced_data['signal'].value_counts()
    print(f"\n✅ Balanced dataset created:")
    print(balanced_counts)
    
    # Step 3: Prepare XGBoost data
    print("\n🤖 Preparing XGBoost training data...")
    
    # Select features
    feature_columns = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist']
    available_features = [col for col in feature_columns if col in balanced_data.columns]
    
    if not available_features:
        print("❌ No suitable features found!")
        return
    
    # Prepare features and labels
    X = balanced_data[available_features].copy()
    
    # Create numeric labels
    label_map = {'HOLD': 0, 'SELL': 1, 'BUY': 2}
    y = balanced_data['signal'].map(label_map)
    
    print(f"Features: {available_features}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Step 4: Train balanced XGBoost
    print("\n🚀 Training balanced XGBoost model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print(f"Class weights: {weight_dict}")
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    # Fit with sample weights
    sample_weights = np.array([weight_dict[label] for label in y_train])
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print("\n📊 XGBoost Model Performance:")
    print(classification_report(y_test, y_pred, target_names=['HOLD', 'SELL', 'BUY']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📈 Confusion Matrix:")
    print(cm)
    
    # Test prediction distribution
    pred_counts = pd.Series(y_pred).value_counts()
    print(f"\n🎯 Prediction Distribution:")
    print(f"HOLD: {pred_counts.get(0, 0)}")
    print(f"SELL: {pred_counts.get(1, 0)}")
    print(f"BUY: {pred_counts.get(2, 0)}")
    
    # Step 5: Test with sample data
    print("\n🧪 Testing model with sample data...")
    
    # Create sample features
    sample_features = np.array([
        [50, 0.5, 0.3, 0.2],   # Neutral
        [20, -1.0, -0.8, -0.2], # Bearish
        [80, 1.0, 0.8, 0.2]    # Bullish
    ])
    
    predictions = model.predict(sample_features)
    probabilities = model.predict_proba(sample_features)
    
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        signal_name = ['HOLD', 'SELL', 'BUY'][pred]
        print(f"Sample {i+1}: {signal_name} (prob: {prob[pred]:.3f})")
    
    # Step 6: Save model
    print("\n💾 Saving balanced XGBoost model...")
    joblib.dump(model, 'models/mrben_ai_signal_filter_xgb_balanced.joblib')
    print("✅ XGBoost model saved as 'mrben_ai_signal_filter_xgb_balanced.joblib'")
    
    print("\n🎉 XGBoost BUY Bias Fix Process Completed!")
    print("=" * 50)

if __name__ == "__main__":
    main() 