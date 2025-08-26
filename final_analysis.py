import pandas as pd


def final_analysis():
    """Final analysis of synthetic dataset."""
    print("🚀 تحلیل نهایی دیتاست مصنوعی")
    print("=" * 50)

    try:
        # Load synthetic dataset
        df = pd.read_csv('data/mrben_ai_signal_dataset_synthetic_balanced.csv')

        # Count signals
        buy_count = len(df[df['signal'] == 'BUY'])
        sell_count = len(df[df['signal'] == 'SELL'])
        hold_count = len(df[df['signal'] == 'HOLD'])
        total_count = len(df)

        print(f"کل نمونه‌ها: {total_count}")
        print(f"BUY: {buy_count} ({buy_count/total_count*100:.1f}%)")
        print(f"SELL: {sell_count} ({sell_count/total_count*100:.1f}%)")
        print(f"HOLD: {hold_count} ({hold_count/total_count*100:.1f}%)")

        # Check balance
        if buy_count > 0 and sell_count > 0:
            ratio = buy_count / sell_count
            print(f"\nنسبت BUY/SELL: {ratio:.2f}")

            if 0.8 <= ratio <= 1.2:
                print("✅ توزیع BUY/SELL متعادل است")
            else:
                print("⚠️ توزیع BUY/SELL نامتعادل است")

        # Compare with original
        original_hold = 97.7
        improvement = original_hold - (hold_count / total_count * 100)
        print(f"\nبهبود نسبت به دیتاست اصلی: {improvement:.1f}% کاهش غلبه HOLD")

        # Check if ready for LSTM training
        print(
            f"\nآماده برای بازآموزی LSTM: {'✅ بله' if buy_count > 0 and sell_count > 0 else '❌ خیر'}"
        )

        # Create LSTM training script if ready
        if buy_count > 0 and sell_count > 0:
            create_lstm_training_script()

        return True

    except Exception as e:
        print(f"❌ خطا: {e}")
        return False


def create_lstm_training_script():
    """Create LSTM training script."""
    print("\n📋 ایجاد اسکریپت بازآموزی LSTM:")

    script_content = '''import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def train_lstm_with_synthetic_data():
    """Train LSTM model with synthetic balanced data."""
    print("🚀 بازآموزی LSTM با داده مصنوعی متعادل")
    print("=" * 50)

    # Create models directory if it doesn't exist
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

    # Reshape for LSTM (samples, timesteps, features)
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
        Dense(3, activation='softmax')  # 3 classes: SELL, HOLD, BUY
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    print("آموزش مدل...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate model
    print("\\nارزیابی مدل:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"دقت تست: {test_accuracy:.4f}")

    # Save model and scaler
    model.save('models/lstm_balanced_model.h5')
    joblib.dump(scaler, 'models/lstm_balanced_scaler.joblib')

    print("✅ مدل LSTM متعادل ذخیره شد")

    # Test predictions
    print("\\nتست پیش‌بینی‌ها:")
    predictions = model.predict(X_test[:10])
    predicted_classes = np.argmax(predictions, axis=1)

    for i, (true, pred) in enumerate(zip(y_test[:10], predicted_classes)):
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        print(f"نمونه {i+1}: واقعی={signal_map[true]}, پیش‌بینی={signal_map[pred]}")

    return model, scaler

if __name__ == "__main__":
    train_lstm_with_synthetic_data()
'''

    try:
        with open('train_lstm_balanced.py', 'w', encoding='utf-8') as f:
            f.write(script_content)
        print("   ✅ اسکریپت train_lstm_balanced.py ایجاد شد")
        print("   📋 حالا می‌توانید LSTM را بازآموزی کنید")
    except Exception as e:
        print(f"   ❌ خطا در ایجاد اسکریپت: {e}")


if __name__ == "__main__":
    final_analysis()
