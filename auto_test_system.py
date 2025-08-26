import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


class AutoTestSystem:
    def __init__(self):
        self.test_results = {}
        self.test_log = []

    def test_dataset_balance(self):
        """Test dataset balance"""
        print("\n🔍 Testing Dataset Balance...")

        try:
            df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')

            buy_count = len(df[df['signal'] == 'BUY'])
            sell_count = len(df[df['signal'] == 'SELL'])
            hold_count = len(df[df['signal'] == 'HOLD'])
            total_count = len(df)

            buy_pct = (buy_count / total_count) * 100
            sell_pct = (sell_count / total_count) * 100
            hold_pct = (hold_count / total_count) * 100

            print(f"Total samples: {total_count}")
            print(f"BUY: {buy_pct:.1f}% ({buy_count})")
            print(f"SELL: {sell_pct:.1f}% ({sell_count})")
            print(f"HOLD: {hold_pct:.1f}% ({hold_count})")

            # Check balance
            if abs(buy_pct - sell_pct) <= 5:
                print("✅ Dataset is balanced")
                return True
            else:
                print("❌ Dataset is not balanced")
                return False

        except Exception as e:
            print(f"❌ Error testing dataset: {e}")
            return False

    def test_lstm_model(self):
        """Test LSTM model predictions"""
        print("\n🔍 Testing LSTM Model...")

        try:
            # Load model
            model = load_model('models/lstm_balanced_model.h5')
            scaler = joblib.load('models/lstm_balanced_scaler.joblib')

            # Load test data
            df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')

            # Prepare features
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
            X = df[feature_columns].values[-100:]  # Last 100 samples for testing

            # Normalize and reshape
            X_scaled = scaler.transform(X)
            timesteps = 10
            X_reshaped = []

            for i in range(timesteps, len(X_scaled)):
                X_reshaped.append(X_scaled[i - timesteps : i])

            X_reshaped = np.array(X_reshaped)

            # Get predictions
            predictions = model.predict(X_reshaped)
            predicted_classes = np.argmax(predictions, axis=1)

            # Analyze distribution
            unique, counts = np.unique(predicted_classes, return_counts=True)
            total = len(predicted_classes)

            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

            print("Predicted Signal Distribution:")
            for class_id, count in zip(unique, counts, strict=False):
                percentage = (count / total) * 100
                signal_name = signal_map[class_id]
                print(f"{signal_name}: {percentage:.1f}% ({count})")

            # Check for bias
            if len(unique) >= 2:
                buy_count = counts[unique == 2][0] if 2 in unique else 0
                sell_count = counts[unique == 0][0] if 0 in unique else 0

                if buy_count > 0 and sell_count > 0:
                    ratio = buy_count / sell_count
                    print(f"BUY/SELL ratio: {ratio:.2f}")

                    if 0.7 <= ratio <= 1.3:
                        print("✅ LSTM predictions are balanced")
                        return True
                    else:
                        print("❌ LSTM predictions show bias")
                        return False

        except Exception as e:
            print(f"❌ Error testing LSTM model: {e}")
            return False

    def test_system_files(self):
        """Test if all required files exist"""
        print("\n🔍 Testing System Files...")

        required_files = [
            'live_trader_clean.py',
            'models/lstm_balanced_model.h5',
            'models/lstm_balanced_scaler.joblib',
            'data/mrben_ai_signal_dataset_synthetic_balanced.csv',
            'advanced_monitoring.py',
        ]

        all_exist = True
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path}")
                all_exist = False

        return all_exist

    def run_comprehensive_test(self):
        """Run all tests"""
        print("🚀 Running Comprehensive Auto Test System")
        print("=" * 60)

        # Test 1: Dataset balance
        dataset_ok = self.test_dataset_balance()
        self.test_results['dataset_balance'] = dataset_ok

        # Test 2: LSTM model
        lstm_ok = self.test_lstm_model()
        self.test_results['lstm_model'] = lstm_ok

        # Test 3: System files
        files_ok = self.test_system_files()
        self.test_results['system_files'] = files_ok

        # Overall result
        all_tests_passed = all([dataset_ok, lstm_ok, files_ok])

        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY:")
        print("=" * 60)
        print(f"Dataset Balance: {'✅ PASS' if dataset_ok else '❌ FAIL'}")
        print(f"LSTM Model: {'✅ PASS' if lstm_ok else '❌ FAIL'}")
        print(f"System Files: {'✅ PASS' if files_ok else '❌ FAIL'}")
        print(
            f"Overall Result: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}"
        )

        # Save results
        self.save_test_results()

        return all_tests_passed

    def save_test_results(self):
        """Save test results to file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'test_log': self.test_log,
        }

        with open('auto_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("\n📄 Test results saved to: auto_test_results.json")


if __name__ == "__main__":
    tester = AutoTestSystem()
    success = tester.run_comprehensive_test()

    if success:
        print("\n🎉 System is ready for live trading!")
    else:
        print("\n⚠️ System needs attention before live trading!")
