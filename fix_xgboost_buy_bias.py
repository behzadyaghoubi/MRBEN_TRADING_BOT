import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


def load_and_analyze_data():
    """Load and analyze the current dataset."""
    print("📊 Loading and analyzing current dataset...")

    # Load dataset
    df = pd.read_csv('data/mrben_ai_signal_dataset.csv')

    # Analyze current distribution
    signal_counts = df['signal'].value_counts()
    total_signals = len(df)

    print("Current signal distribution:")
    for signal in signal_counts.index:
        count = signal_counts[signal]
        percentage = (count / total_signals) * 100
        print(f"  {signal}: {count} ({percentage:.1f}%)")

    return df


def create_balanced_dataset(df, target_ratio=0.3):
    """Create a balanced dataset with target BUY/SELL ratio."""
    print(f"\n🔄 Creating balanced dataset with {target_ratio*100:.0f}% BUY/SELL ratio...")

    # Separate signals
    hold_data = df[df['signal'] == 'HOLD'].copy()
    buy_data = df[df['signal'] == 'BUY'].copy()
    sell_data = df[df['signal'] == 'SELL'].copy()

    print(f"Original counts: HOLD={len(hold_data)}, BUY={len(buy_data)}, SELL={len(sell_data)}")

    # Calculate target counts
    min_buy_sell = min(len(buy_data), len(sell_data))
    target_buy_sell = max(min_buy_sell, 50)  # At least 50 samples each

    # Balance BUY and SELL
    if len(buy_data) > target_buy_sell:
        buy_data = buy_data.sample(n=target_buy_sell, random_state=42)
    elif len(buy_data) < target_buy_sell:
        # Oversample BUY data
        buy_data = buy_data.sample(n=target_buy_sell, replace=True, random_state=42)

    if len(sell_data) > target_buy_sell:
        sell_data = sell_data.sample(n=target_buy_sell, random_state=42)
    elif len(sell_data) < target_buy_sell:
        # Oversample SELL data
        sell_data = sell_data.sample(n=target_buy_sell, replace=True, random_state=42)

    # Reduce HOLD data to balance the dataset
    target_hold = int(target_buy_sell * (1 - 2 * target_ratio) / target_ratio)
    if len(hold_data) > target_hold:
        hold_data = hold_data.sample(n=target_hold, random_state=42)

    # Combine balanced data
    balanced_df = pd.concat([hold_data, buy_data, sell_data], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Analyze balanced distribution
    balanced_counts = balanced_df['signal'].value_counts()
    total_balanced = len(balanced_df)

    print("Balanced signal distribution:")
    for signal in balanced_counts.index:
        count = balanced_counts[signal]
        percentage = (count / total_balanced) * 100
        print(f"  {signal}: {count} ({percentage:.1f}%)")

    return balanced_df


def prepare_xgboost_data(df):
    """Prepare data for XGBoost training."""
    print("\n🔧 Preparing data for XGBoost...")

    # Select features (excluding time and signal)
    feature_columns = [
        'open',
        'high',
        'low',
        'close',
        'SMA20',
        'SMA50',
        'RSI',
        'MACD',
        'MACD_signal',
        'MACD_hist',
    ]
    available_features = [col for col in feature_columns if col in df.columns]

    print(f"Using features: {available_features}")

    # Prepare X and y
    X = df[available_features].copy()
    y = df['signal'].copy()

    # Handle missing values
    X = X.fillna(X.mean())

    # Encode labels
    label_mapping = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
    y_encoded = y.map(label_mapping)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_features


def train_balanced_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with balanced data and class weights."""
    print("\n🚀 Training balanced XGBoost model...")

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(zip(np.unique(y_train), class_weights, strict=False))
    sample_weights = np.array([weight_dict[label] for label in y_train])

    print(f"Class weights: {weight_dict}")

    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
    )

    # Fit with sample weights
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    print("\n📊 Model Evaluation:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['HOLD', 'BUY', 'SELL']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Analyze prediction distribution
    pred_counts = pd.Series(y_pred).value_counts()
    print("\nPrediction distribution on test set:")
    for i, count in pred_counts.items():
        percentage = (count / len(y_pred)) * 100
        signal_name = ['HOLD', 'BUY', 'SELL'][i]
        print(f"  {signal_name}: {count} ({percentage:.1f}%)")

    return model, y_pred_proba


def save_model_and_test(model, scaler, feature_names):
    """Save the trained model and create test script."""
    print("\n💾 Saving balanced model...")

    # Save model
    model_path = 'models/mrben_ai_signal_filter_xgb_balanced.joblib'
    joblib.dump(model, model_path)

    # Save scaler
    scaler_path = 'models/mrben_ai_signal_filter_xgb_balanced_scaler.joblib'
    joblib.dump(scaler, scaler_path)

    # Save feature names
    feature_path = 'models/mrben_ai_signal_filter_xgb_balanced_features.txt'
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_names))

    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    print(f"✅ Features saved to: {feature_path}")

    # Create test script
    create_test_script(feature_names)


def create_test_script(feature_names):
    """Create a test script to verify the balanced model."""
    test_script = '''import joblib
import numpy as np
import pandas as pd

def test_balanced_model():
    """Test the balanced XGBoost model."""
    print("Testing balanced XGBoost model...")

    # Load model and scaler
    model = joblib.load('models/mrben_ai_signal_filter_xgb_balanced.joblib')
    scaler = joblib.load('models/mrben_ai_signal_filter_xgb_balanced_scaler.joblib')

    # Create test data with different scenarios
    test_scenarios = [
        # Bullish scenario
        {'open': 3300, 'high': 3310, 'low': 3295, 'close': 3308,
         'SMA20': 3290, 'SMA50': 3280, 'RSI': 65, 'MACD': 0.5, 'MACD_signal': 0.3, 'MACD_hist': 0.2},
        # Bearish scenario
        {'open': 3300, 'high': 3305, 'low': 3280, 'close': 3285,
         'SMA20': 3310, 'SMA50': 3320, 'RSI': 35, 'MACD': -0.5, 'MACD_signal': -0.3, 'MACD_hist': -0.2},
        # Neutral scenario
        {'open': 3300, 'high': 3302, 'low': 3298, 'close': 3300,
         'SMA20': 3300, 'SMA50': 3300, 'RSI': 50, 'MACD': 0.0, 'MACD_signal': 0.0, 'MACD_hist': 0.0}
    ]

    feature_names = {feature_names}

    for i, scenario in enumerate(test_scenarios):
        # Prepare features
        features = [scenario[col] for col in feature_names]
        features_scaled = scaler.transform([features])

        # Get prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        scenario_name = ['Bullish', 'Bearish', 'Neutral'][i]
        signal_name = ['HOLD', 'BUY', 'SELL'][prediction]

        print(f"\\n{scenario_name} Scenario:")
        print(f"  Prediction: {signal_name} (class {prediction})")
        print(f"  Probabilities: HOLD={probabilities[0]:.3f}, BUY={probabilities[1]:.3f}, SELL={probabilities[2]:.3f}")

if __name__ == "__main__":
    test_balanced_model()
'''

    # Replace placeholder with actual feature names
    test_script = test_script.replace('{feature_names}', str(feature_names))

    with open('test_balanced_model.py', 'w', encoding='utf-8') as f:
        f.write(test_script)

    print("Test script created: test_balanced_model.py")


def main():
    """Main function to fix XGBoost BUY bias."""
    print("🚀 Starting XGBoost BUY Bias Fix Process...")
    print("=" * 60)

    try:
        # Step 1: Load and analyze data
        df = load_and_analyze_data()

        # Step 2: Create balanced dataset
        balanced_df = create_balanced_dataset(df, target_ratio=0.3)

        # Step 3: Prepare data for training
        X_train, X_test, y_train, y_test, scaler, feature_names = prepare_xgboost_data(balanced_df)

        # Step 4: Train balanced model
        model, y_pred_proba = train_balanced_xgboost(X_train, y_train, X_test, y_test)

        # Step 5: Save model and create test script
        save_model_and_test(model, scaler, feature_names)

        print("\n🎉 XGBoost BUY Bias Fix Completed Successfully!")
        print("=" * 60)
        print("Next steps:")
        print("1. Run: python test_balanced_model.py")
        print("2. Update live_trader_clean.py to use the new balanced model")
        print("3. Test the system with the balanced model")

    except Exception as e:
        print(f"❌ Error during XGBoost bias fix: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
