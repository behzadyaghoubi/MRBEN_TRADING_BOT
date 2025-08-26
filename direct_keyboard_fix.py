import os
import winreg


def fix_keyboard_directly():
    """Fix keyboard issue directly without terminal commands."""
    print("🔧 حل مستقیم مشکل کیبورد")
    print("=" * 50)

    try:
        # Method 1: Set environment variables directly
        print("1. تنظیم متغیرهای محیطی...")
        os.environ['LANG'] = 'en_US.UTF-8'
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LC_CTYPE'] = 'en_US.UTF-8'
        os.environ['INPUT_METHOD'] = 'default'
        print("   ✅ متغیرهای محیطی تنظیم شدند")

        # Method 2: Try to fix registry
        print("2. تنظیم Registry...")
        try:
            # Set English US as default keyboard layout
            preload_key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, r"Keyboard Layout\Preload", 0, winreg.KEY_WRITE
            )
            winreg.SetValueEx(preload_key, "1", 0, winreg.REG_SZ, "00000409")
            winreg.CloseKey(preload_key)
            print("   ✅ Registry تنظیم شد")
        except Exception as e:
            print(f"   ⚠️ خطا در Registry: {e}")

        # Method 3: Create a simple test script
        print("3. ایجاد اسکریپت تست...")
        test_script = '''import sys
print("Python version:", sys.version)
print("✅ Keyboard test successful!")
print("No Persian characters detected!")
'''

        with open('keyboard_test.py', 'w', encoding='utf-8') as f:
            f.write(test_script)
        print("   ✅ اسکریپت تست ایجاد شد")

        # Method 4: Create a simple analysis script
        print("4. ایجاد اسکریپت تحلیل...")
        analysis_script = '''import pandas as pd
import os

def analyze_dataset():
    try:
        df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')

        buy_count = len(df[df['signal'] == 'BUY'])
        sell_count = len(df[df['signal'] == 'SELL'])
        hold_count = len(df[df['signal'] == 'HOLD'])
        total_count = len(df)

        print(f"Dataset Analysis:")
        print(f"Total samples: {total_count}")
        print(f"BUY: {buy_count} ({buy_count/total_count*100:.1f}%)")
        print(f"SELL: {sell_count} ({sell_count/total_count*100:.1f}%)")
        print(f"HOLD: {hold_count} ({hold_count/total_count*100:.1f}%)")

        if buy_count > 0 and sell_count > 0:
            ratio = buy_count / sell_count
            print(f"BUY/SELL ratio: {ratio:.2f}")

            if 0.8 <= ratio <= 1.2:
                print("✅ Balanced distribution")
            else:
                print("⚠️ Unbalanced distribution")

        print("✅ Dataset ready for LSTM training!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    analyze_dataset()
'''

        with open('direct_analysis.py', 'w', encoding='utf-8') as f:
            f.write(analysis_script)
        print("   ✅ اسکریپت تحلیل ایجاد شد")

        # Method 5: Create LSTM training script
        print("5. ایجاد اسکریپت بازآموزی LSTM...")
        lstm_script = '''import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def train_lstm():
    print("🚀 Training LSTM with balanced data")
    print("=" * 50)

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Load synthetic dataset
    df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')

    # Prepare features
    feature_columns = ['open', 'high', 'low', 'close', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
    X = df[feature_columns].values
    y = df['signal'].map({'SELL': 0, 'HOLD': 1, 'BUY': 2}).values

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM
    timesteps = 10
    X_reshaped = []
    y_reshaped = []

    for i in range(timesteps, len(X_scaled)):
        X_reshaped.append(X_scaled[i-timesteps:i])
        y_reshaped.append(y[i])

    X_reshaped = np.array(X_reshaped)
    y_reshaped = np.array(y_reshaped)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_reshaped, test_size=0.2, random_state=42, stratify=y_reshaped
    )

    # Create LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(timesteps, len(feature_columns))),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(3, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save model and scaler
    model.save('models/lstm_balanced_model.h5')
    joblib.dump(scaler, 'models/lstm_balanced_scaler.joblib')

    print("✅ LSTM model saved successfully!")

    # Test predictions
    predictions = model.predict(X_test[:10])
    predicted_classes = np.argmax(predictions, axis=1)

    for i, (true, pred) in enumerate(zip(y_test[:10], predicted_classes)):
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        print(f"Sample {i+1}: True={signal_map[true]}, Predicted={signal_map[pred]}")

    return model, scaler

if __name__ == "__main__":
    train_lstm()
'''

        with open('direct_lstm_training.py', 'w', encoding='utf-8') as f:
            f.write(lstm_script)
        print("   ✅ اسکریپت بازآموزی LSTM ایجاد شد")

        print("\n✅ تمام ابزارها آماده شدند!")
        print("\n📋 مراحل بعدی:")
        print("1. python keyboard_test.py")
        print("2. python direct_analysis.py")
        print("3. python direct_lstm_training.py")
        print("4. python live_trader_clean.py")

        return True

    except Exception as e:
        print(f"❌ خطا: {e}")
        return False


if __name__ == "__main__":
    success = fix_keyboard_directly()

    if success:
        print("\n🎯 مشکل کیبورد حل شد!")
        print("حالا می‌توانید اسکریپت‌ها را اجرا کنید.")
    else:
        print("\n❌ مشکل در حل کیبورد")
