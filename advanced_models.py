"""
Advanced LSTM Models with Attention Mechanism
Enhanced models for our Gold Trading Bot
"""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Attention, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger(__name__)


def build_advanced_lstm(input_shape, num_classes=3):
    """
    Build advanced LSTM model with attention mechanism

    Args:
        input_shape: Tuple of (timesteps, features)
        num_classes: Number of output classes (default: 3 for BUY/HOLD/SELL)

    Returns:
        Compiled Keras model
    """
    try:
        inputs = Input(shape=input_shape)

        # First Bidirectional LSTM Layer
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)

        # Second Bidirectional LSTM Layer
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)

        # Attention Layer
        attention_output = Attention()([x, x])

        # Global Average Pooling to reduce sequence dimension
        x = tf.keras.layers.GlobalAveragePooling1D()(attention_output)

        # Dense Layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)

        # Output Layer
        if num_classes == 1:
            outputs = Dense(1, activation='linear')(x)  # For regression
        else:
            outputs = Dense(num_classes, activation='softmax')(x)  # For classification

        # Build model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        if num_classes == 1:
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        else:
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )

        logger.info(f"Advanced LSTM model built successfully with input shape: {input_shape}")
        return model

    except Exception as e:
        logger.error(f"Error building advanced LSTM model: {e}")
        return None


def build_enhanced_lstm_classifier(input_shape, num_classes=3):
    """
    Enhanced LSTM classifier specifically for trading signals

    Args:
        input_shape: Tuple of (timesteps, features)
        num_classes: Number of output classes (default: 3 for BUY/HOLD/SELL)

    Returns:
        Compiled Keras model
    """
    try:
        inputs = Input(shape=input_shape)

        # Multiple LSTM layers with different configurations
        x = LSTM(256, return_sequences=True, activation='tanh')(inputs)
        x = Dropout(0.4)(x)

        x = LSTM(128, return_sequences=True, activation='tanh')(x)
        x = Dropout(0.3)(x)

        x = LSTM(64, return_sequences=False, activation='tanh')(x)
        x = Dropout(0.2)(x)

        # Dense layers for feature extraction
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)

        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)

        # Build and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy'],
        )

        logger.info("Enhanced LSTM classifier built successfully")
        return model

    except Exception as e:
        logger.error(f"Error building enhanced LSTM classifier: {e}")
        return None


def get_callbacks(patience=10, min_lr=1e-7):
    """
    Get training callbacks for the models

    Args:
        patience: Number of epochs to wait before early stopping
        min_lr: Minimum learning rate for ReduceLROnPlateau

    Returns:
        List of callbacks
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=min_lr, verbose=1),
    ]
    return callbacks


def prepare_sequences_for_advanced_lstm(data, timesteps=50, features=None):
    """
    Prepare sequences for advanced LSTM model

    Args:
        data: DataFrame with features
        timesteps: Number of time steps for sequences
        features: List of feature columns to use

    Returns:
        X: Sequences array, y: Labels array
    """
    try:
        if features is None:
            # Default features for trading
            features = ['open', 'high', 'low', 'close', 'tick_volume']

        # Select features
        feature_data = data[features].values

        # Create sequences
        X, y = [], []
        for i in range(timesteps, len(feature_data)):
            X.append(feature_data[i - timesteps : i])
            y.append(feature_data[i, 3])  # Use close price as target

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Prepared sequences: X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    except Exception as e:
        logger.error(f"Error preparing sequences: {e}")
        return None, None


# Example usage function
def test_advanced_lstm():
    """
    Test function to verify the advanced LSTM model works correctly
    """
    try:
        # Test parameters
        timesteps = 50
        features = 5
        input_shape = (timesteps, features)

        # Build model
        model = build_advanced_lstm(input_shape, num_classes=3)

        if model is not None:
            print("✅ Advanced LSTM model built successfully!")
            print("Model summary:")
            model.summary()

            # Test with dummy data
            dummy_X = np.random.random((10, timesteps, features))
            dummy_y = np.random.randint(0, 3, (10,))

            # Test prediction
            predictions = model.predict(dummy_X)
            print(f"✅ Predictions shape: {predictions.shape}")
            print(f"✅ Sample predictions: {predictions[0]}")

            return True
        else:
            print("❌ Failed to build model")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the advanced LSTM model
    print("🧪 Testing Advanced LSTM Model...")
    test_advanced_lstm()
