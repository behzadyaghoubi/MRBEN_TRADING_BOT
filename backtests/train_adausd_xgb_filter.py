import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ADAUSDXGBFilterTrainer:
    """
    Professional ADAUSD XGBoost Signal Filter Trainer
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.model = None
        self.feature_importance = None

    def load_and_prepare_data(self, signal_file='adausd_signals.csv'):
        """
        Load and prepare ADAUSD data for XGBoost training
        """
        logger.info("Loading and preparing ADAUSD data...")

        try:
            df = pd.read_csv(signal_file)
            logger.info(f"Loaded {len(df)} rows from {signal_file}")
        except FileNotFoundError:
            logger.error(f"Signal file {signal_file} not found!")
            return None

        # Remove rows with NaN values
        df = df.dropna()
        logger.info(f"After removing NaN: {len(df)} rows")

        # Prepare features and target
        feature_columns = [
            col
            for col in df.columns
            if col not in ['time', 'signal', 'random_signal', 'Dividends', 'Stock Splits']
        ]

        X = df[feature_columns]
        y = df['signal']

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        logger.info(f"Feature columns: {len(feature_columns)}")
        logger.info(f"Target distribution: {np.bincount(y_encoded)}")

        return X, y_encoded, feature_columns

    def feature_engineering(self, X):
        """
        Advanced feature engineering for ADAUSD
        """
        logger.info("Performing advanced feature engineering...")

        # Create interaction features
        X_engineered = X.copy()

        # Price momentum interactions
        if 'price_change' in X.columns and 'momentum_5' in X.columns:
            X_engineered['price_momentum_interaction'] = X['price_change'] * X['momentum_5']

        if 'price_change' in X.columns and 'momentum_10' in X.columns:
            X_engineered['price_momentum_10_interaction'] = X['price_change'] * X['momentum_10']

        # RSI and MACD interactions
        if 'RSI' in X.columns and 'MACD' in X.columns:
            X_engineered['rsi_macd_interaction'] = X['RSI'] * X['MACD']

        # Volatility and volume interactions
        if 'volatility_5' in X.columns and 'volume_ratio' in X.columns:
            X_engineered['vol_vol_interaction'] = X['volatility_5'] * X['volume_ratio']

        # Bollinger Bands interactions
        if 'bb_pos' in X.columns and 'bb_width' in X.columns:
            X_engineered['bb_interaction'] = X['bb_pos'] * X['bb_width']

        # Moving average crossovers
        if 'price_vs_sma_5' in X.columns and 'price_vs_sma_20' in X.columns:
            X_engineered['ma_crossover'] = X['price_vs_sma_5'] - X['price_vs_sma_20']

        if 'price_vs_ema_12' in X.columns and 'price_vs_ema_26' in X.columns:
            X_engineered['ema_crossover'] = X['price_vs_ema_12'] - X['price_vs_ema_26']

        # Trend strength features
        if 'trend_5' in X.columns and 'trend_10' in X.columns and 'trend_20' in X.columns:
            X_engineered['trend_strength'] = X['trend_5'] + X['trend_10'] + X['trend_20']

        # Support/Resistance features
        if 'dist_from_support_5' in X.columns and 'dist_from_resistance_5' in X.columns:
            X_engineered['support_resistance_ratio'] = X['dist_from_support_5'] / (
                X['dist_from_resistance_5'] + 1e-8
            )

        # Price action features
        if 'body_size' in X.columns and 'upper_shadow' in X.columns and 'lower_shadow' in X.columns:
            X_engineered['candle_strength'] = X['body_size'] / (
                X['upper_shadow'] + X['lower_shadow'] + 1e-8
            )

        # Technical indicator combinations
        if 'RSI' in X.columns and 'stoch_k' in X.columns:
            X_engineered['rsi_stoch_divergence'] = X['RSI'] - X['stoch_k']

        if 'MACD' in X.columns and 'MACD_signal' in X.columns:
            X_engineered['macd_divergence'] = X['MACD'] - X['MACD_signal']

        # Fill any remaining NaN values
        X_engineered = X_engineered.fillna(0)

        logger.info(f"Engineered features: {X_engineered.shape[1]}")

        return X_engineered

    def feature_selection(self, X, y, n_features=50):
        """
        Select best features using multiple methods
        """
        logger.info(f"Performing feature selection (selecting {n_features} features)...")

        # Method 1: Statistical tests
        selector1 = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        X_selected1 = selector1.fit_transform(X, y)
        selected_features1 = X.columns[selector1.get_support()].tolist()

        # Method 2: Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        feature_importance = rf.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-n_features:]
        selected_features2 = X.columns[top_features_idx].tolist()

        # Combine both methods
        all_selected = list(set(selected_features1 + selected_features2))

        # Use RFE for final selection
        if len(all_selected) > n_features:
            X_temp = X[all_selected]
            rfe = RFE(
                estimator=RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                n_features_to_select=n_features,
            )
            X_rfe = rfe.fit_transform(X_temp, y)
            final_features = X_temp.columns[rfe.get_support()].tolist()
        else:
            final_features = all_selected

        logger.info(f"Selected {len(final_features)} features")

        return X[final_features], final_features

    def optimize_hyperparameters(self, X, y):
        """
        Optimize XGBoost hyperparameters using GridSearchCV
        """
        logger.info("Optimizing XGBoost hyperparameters...")

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
        }

        # Initialize XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            random_state=self.random_state,
            eval_metric='mlogloss',
            use_label_encoder=False,
        )

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X, y)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def train_model(self, X, y):
        """
        Train the final XGBoost model
        """
        logger.info("Training final XGBoost model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Optimize hyperparameters
        best_model, best_params = self.optimize_hyperparameters(X_train_scaled, y_train)

        # Train final model with best parameters
        final_model = xgb.XGBClassifier(
            **best_params,
            objective='multi:softprob',
            random_state=self.random_state,
            eval_metric='mlogloss',
            use_label_encoder=False,
        )

        final_model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = final_model.predict(X_test_scaled)
        y_pred_proba = final_model.predict_proba(X_test_scaled)

        # Calculate metrics
        accuracy = final_model.score(X_test_scaled, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        # Feature importance
        feature_importance = final_model.feature_importances_
        feature_names = X.columns

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test AUC: {auc_score:.4f}")

        self.model = final_model
        self.feature_importance = dict(zip(feature_names, feature_importance, strict=False))

        return {
            'model': final_model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'feature_importance': self.feature_importance,
        }

    def evaluate_model(self, results):
        """
        Evaluate model performance
        """
        logger.info("Evaluating model performance...")

        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']

        # Classification report
        class_names = ['SELL', 'HOLD', 'BUY']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print("\n" + "=" * 60)
        print("ADAUSD XGBOOST FILTER EVALUATION RESULTS")
        print("=" * 60)
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test AUC: {results['auc_score']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        print("=" * 60)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Confusion Matrix - ADAUSD XGBoost Filter')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('outputs/adausd_xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Plot feature importance
        self.plot_feature_importance()

        return report, cm

    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        """
        if self.feature_importance is None:
            logger.error("No feature importance available!")
            return

        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]

        features, importance = zip(*top_features, strict=False)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - ADAUSD XGBoost Filter')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/adausd_xgb_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, model_dir='outputs'):
        """
        Save the trained model and components
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = f'{model_dir}/adausd_xgb_filter.joblib'
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save scaler
        scaler_path = f'{model_dir}/adausd_xgb_scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # Save label encoder
        encoder_path = f'{model_dir}/adausd_xgb_label_encoder.joblib'
        joblib.dump(self.label_encoder, encoder_path)
        logger.info(f"Label encoder saved to {encoder_path}")

        # Save feature importance
        importance_path = f'{model_dir}/adausd_xgb_feature_importance.joblib'
        joblib.dump(self.feature_importance, importance_path)
        logger.info(f"Feature importance saved to {importance_path}")


def main():
    """
    Main function to train ADAUSD XGBoost filter
    """
    logger.info("Starting ADAUSD XGBoost filter training...")

    # Initialize trainer
    trainer = ADAUSDXGBFilterTrainer(test_size=0.2, random_state=42)

    # Load and prepare data
    data = trainer.load_and_prepare_data('adausd_signals.csv')
    if data is None:
        return

    X, y, feature_columns = data

    # Feature engineering
    X_engineered = trainer.feature_engineering(X)

    # Feature selection
    X_selected, selected_features = trainer.feature_selection(X_engineered, y, n_features=50)

    # Train model
    results = trainer.train_model(X_selected, y)

    # Evaluate model
    report, cm = trainer.evaluate_model(results)

    # Save model
    trainer.save_model()

    logger.info("ADAUSD XGBoost filter training completed successfully!")


if __name__ == "__main__":
    main()
