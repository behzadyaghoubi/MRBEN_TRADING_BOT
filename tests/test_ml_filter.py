import joblib
import numpy as np

# Load the model
print("Loading ML filter model...")
model = joblib.load('mrben_ai_signal_filter_xgb.joblib')

print(f"Model type: {type(model)}")
print(f"Expected features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown'}")

# Test with 6 features
buy_proba = 0.6
sell_proba = 0.3
signal = 1
hold_proba = 1 - buy_proba - sell_proba

# Try different combinations
test_cases = [
    ([buy_proba, hold_proba, sell_proba, signal, 0, 0], "6 features with zeros"),
    ([buy_proba, hold_proba, sell_proba, signal, 1000, 0], "6 features with price"),
    ([buy_proba, hold_proba, sell_proba, signal, 1000, 1], "6 features with additional"),
]

for features, desc in test_cases:
    try:
        features_array = np.array([features])
        pred = model.predict(features_array)
        print(f"✅ {desc}: Success - Prediction: {pred[0]}")
    except Exception as e:
        print(f"❌ {desc}: Failed - {str(e)[:100]}")

print("\nTesting with actual data from CSV...")
import pandas as pd
df = pd.read_csv('lstm_filtered_signals.csv')
if len(df) > 0:
    row = df.iloc[0]
    features = [
        row['lstm_buy_proba'],
        row['lstm_hold_proba'], 
        row['lstm_sell_proba'],
        row['lstm_signal'],
        row['close'],
        0  # additional feature
    ]
    try:
        features_array = np.array([features])
        pred = model.predict(features_array)
        print(f"✅ Real data test: Success - Prediction: {pred[0]}")
    except Exception as e:
        print(f"❌ Real data test: Failed - {str(e)[:100]}") 