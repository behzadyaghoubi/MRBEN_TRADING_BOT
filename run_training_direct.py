#!/usr/bin/env python3
"""
Run Training Direct
Direct execution without command line issues
"""

print("🚀 Starting LSTM Training - Direct Execution")
print("=" * 60)

# Import and run training directly
try:
    from final_training_solution import main
    print("✅ Imported training module successfully")
    
    # Run training
    success = main()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("✅ Model saved: models/mrben_lstm_real_data.h5")
        print("✅ Scaler saved: models/mrben_lstm_real_data_scaler.save")
    else:
        print("\n❌ Training failed!")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative method...")
    
    # Alternative: direct execution
    try:
        import os
        import sys
        import numpy as np
        import joblib
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        from tensorflow.keras.utils import to_categorical
        
        print("✅ All modules imported successfully")
        print("🎯 Starting training process...")
        
        # Check data files
        sequences_path = "data/real_market_sequences.npy"
        labels_path = "data/real_market_labels.npy"
        
        if not os.path.exists(sequences_path):
            print(f"❌ Sequences file not found: {sequences_path}")
            sys.exit(1)
        
        if not os.path.exists(labels_path):
            print(f"❌ Labels file not found: {labels_path}")
            sys.exit(1)
        
        print("✅ Data files found")
        
        # Load data
        print("📊 Loading real market data...")
        sequences = np.load(sequences_path)
        labels = np.load(labels_path)
        print(f"✅ Data loaded: {sequences.shape}")
        
        # Prepare data
        print("🔧 Preparing data for training...")
        X = sequences
        y = to_categorical(labels, num_classes=3)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"✅ Data prepared: {len(X_train)} training, {len(X_test)} test")
        
        # Create model
        print("🏗️ Creating LSTM model...")
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model created with {model.count_params()} parameters")
        
        # Ensure models directory
        if not os.path.exists("models"):
            os.makedirs("models")
        
        # Train model
        print("🎯 Training LSTM model...")
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'models/mrben_lstm_real_data_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Evaluate
        print("📊 Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        accuracy = np.mean(y_pred_classes == y_test_classes)
        print(f"✅ Test Accuracy: {accuracy:.4f}")
        
        # Save model and scaler
        print("💾 Saving model and scaler...")
        model.save('models/mrben_lstm_real_data.h5')
        
        scaler = MinMaxScaler()
        n_samples, n_timesteps, n_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, n_features)
        scaler.fit(sequences_reshaped)
        joblib.dump(scaler, 'models/mrben_lstm_real_data_scaler.save')
        
        print("✅ Model and scaler saved successfully")
        
        print(f"\n🎉 LSTM Retraining Completed Successfully!")
        print(f"   Final Test Accuracy: {accuracy:.4f}")
        print(f"   Training Epochs: {len(history.history['loss'])}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

print("\nPress Enter to exit...")
input() 