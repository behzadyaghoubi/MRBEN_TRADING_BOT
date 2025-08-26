import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ADAUSDLSTMTrainer:
    """
    Professional ADAUSD LSTM Trainer
    """

    def __init__(self, sequence_length=60, n_features=100, n_classes=3):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None

    def load_data(self, data_dir='outputs'):
        """
        Load preprocessed data
        """
        logger.info("Loading ADAUSD training data...")

        try:
            X_train = np.load(f'{data_dir}/X_train_adausd.npy')
            X_val = np.load(f'{data_dir}/X_val_adausd.npy')
            X_test = np.load(f'{data_dir}/X_test_adausd.npy')
            y_train = np.load(f'{data_dir}/y_train_adausd.npy')
            y_val = np.load(f'{data_dir}/y_val_adausd.npy')
            y_test = np.load(f'{data_dir}/y_test_adausd.npy')

            logger.info("Loaded data shapes:")
            logger.info(f"X_train: {X_train.shape}")
            logger.info(f"X_val: {X_val.shape}")
            logger.info(f"X_test: {X_test.shape}")
            logger.info(f"y_train: {y_train.shape}")
            logger.info(f"y_val: {y_val.shape}")
            logger.info(f"y_test: {y_test.shape}")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except FileNotFoundError as e:
            logger.error(f"Data files not found: {e}")
            logger.error("Please run build_adausd_lstm_dataset.py first!")
            return None

    def build_model(self, learning_rate=0.001, dropout_rate=0.3):
        """
        Build advanced LSTM model
        """
        logger.info("Building advanced ADAUSD LSTM model...")

        model = Sequential(
            [
                # First LSTM layer with return sequences
                Bidirectional(
                    LSTM(
                        units=128,
                        return_sequences=True,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                        recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    ),
                    input_shape=(self.sequence_length, self.n_features),
                ),
                BatchNormalization(),
                Dropout(dropout_rate),
                # Second LSTM layer
                Bidirectional(
                    LSTM(
                        units=64,
                        return_sequences=True,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                        recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    )
                ),
                BatchNormalization(),
                Dropout(dropout_rate),
                # Third LSTM layer
                Bidirectional(
                    LSTM(
                        units=32,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                        recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    )
                ),
                BatchNormalization(),
                Dropout(dropout_rate),
                # Dense layers
                Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.5),
                Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.5),
                # Output layer
                Dense(self.n_classes, activation='softmax'),
            ]
        )

        # Compile model
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy'],
        )

        logger.info("Model architecture:")
        model.summary()

        self.model = model
        return model

    def create_callbacks(self, model_dir='outputs'):
        """
        Create training callbacks
        """
        os.makedirs(model_dir, exist_ok=True)

        callbacks = [
            # Early stopping
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            # Learning rate reduction
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
            # Model checkpoint
            ModelCheckpoint(
                filepath=f'{model_dir}/adausd_lstm_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
            ),
        ]

        return callbacks

    def train_model(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
        """
        Train the model
        """
        logger.info("Starting ADAUSD LSTM model training...")

        # Create callbacks
        callbacks = self.create_callbacks()

        # Train model
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )

        logger.info("Training completed!")
        return self.history

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        logger.info("Evaluating model performance...")

        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        test_loss, test_accuracy, test_categorical_accuracy = self.model.evaluate(
            X_test, y_test, verbose=0
        )

        # Classification report
        class_names = ['SELL', 'HOLD', 'BUY']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")

        print("\n" + "=" * 60)
        print("ADAUSD LSTM MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        print("=" * 60)

        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': report,
            'confusion_matrix': cm,
        }

    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            logger.error("No training history available!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot learning rate
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        # Plot confusion matrix
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('outputs/adausd_lstm_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, model_dir='outputs'):
        """
        Save the trained model
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save final model
        final_model_path = f'{model_dir}/adausd_lstm_final_model.h5'
        self.model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        # Save model info
        model_info = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'model_architecture': self.model.get_config(),
        }

        joblib.dump(model_info, f'{model_dir}/adausd_lstm_model_info.joblib')
        logger.info(f"Model info saved to {model_dir}/adausd_lstm_model_info.joblib")


def main():
    """
    Main function to train ADAUSD LSTM model
    """
    logger.info("Starting ADAUSD LSTM model training...")

    # Initialize trainer
    trainer = ADAUSDLSTMTrainer(
        sequence_length=60, n_features=100, n_classes=3  # Will be updated based on actual data
    )

    # Load data
    data = trainer.load_data()
    if data is None:
        return

    X_train, X_val, X_test, y_train, y_val, y_test = data

    # Update n_features based on actual data
    trainer.n_features = X_train.shape[2]
    logger.info(f"Updated n_features to {trainer.n_features}")

    # Build model
    model = trainer.build_model(learning_rate=0.001, dropout_rate=0.3)

    # Train model
    history = trainer.train_model(X_train, y_train, X_val, y_val, epochs=200, batch_size=32)

    # Evaluate model
    results = trainer.evaluate_model(X_test, y_test)

    # Plot training history
    trainer.plot_training_history()

    # Save model
    trainer.save_model()

    logger.info("ADAUSD LSTM model training completed successfully!")


if __name__ == "__main__":
    main()
