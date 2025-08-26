import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADAUSDDatasetBuilder:
    """
    Professional ADAUSD LSTM dataset builder
    """
    
    def __init__(self, sequence_length=60, test_size=0.2, validation_size=0.1):
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.validation_size = validation_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def add_advanced_features(self, df):
        """
        Add advanced technical features for ADAUSD
        """
        logger.info("Adding advanced technical features...")
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(2)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        df['price_change_20'] = df['close'].pct_change(20)
        
        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        # Volatility indicators
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_10'] = df['price_change'].rolling(window=10).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # Momentum indicators
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # RSI
        df['RSI'] = talib.RSI(df['close'].values, timeperiod=14)
        df['RSI_5'] = talib.RSI(df['close'].values, timeperiod=5)
        df['RSI_21'] = talib.RSI(df['close'].values, timeperiod=21)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_hist'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_pos'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df['CCI'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # ADX (Average Directional Index)
        df['ADX'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # OBV (On Balance Volume)
        try:
            df['OBV'] = talib.OBV(df['close'].values.astype(float), df['tick_volume'].values.astype(float))
        except:
            # Simple OBV calculation if talib fails
            df['OBV'] = 0
            for i in range(1, len(df)):
                if df.iloc[i]['close'] > df.iloc[i-1]['close']:
                    df.iloc[i, df.columns.get_loc('OBV')] = df.iloc[i-1]['OBV'] + df.iloc[i]['tick_volume']
                elif df.iloc[i]['close'] < df.iloc[i-1]['close']:
                    df.iloc[i, df.columns.get_loc('OBV')] = df.iloc[i-1]['OBV'] - df.iloc[i]['tick_volume']
                else:
                    df.iloc[i, df.columns.get_loc('OBV')] = df.iloc[i-1]['OBV']
        
        # Price patterns
        df['doji'] = talib.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['hammer'] = talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['engulfing'] = talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        
        # High-Low range features
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_5'] = df['hl_range'].rolling(window=5).mean()
        df['hl_range_10'] = df['hl_range'].rolling(window=10).mean()
        
        # Volume features
        df['volume_change'] = df['tick_volume'].pct_change()
        df['volume_sma_5'] = df['tick_volume'].rolling(window=5).mean()
        df['volume_sma_10'] = df['tick_volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma_5']
        df['volume_ratio_10'] = df['tick_volume'] / df['volume_sma_10']
        
        # Price action features
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        
        # Trend features
        df['trend_5'] = np.where(df['close'] > df['close'].shift(5), 1, -1)
        df['trend_10'] = np.where(df['close'] > df['close'].shift(10), 1, -1)
        df['trend_20'] = np.where(df['close'] > df['close'].shift(20), 1, -1)
        
        # Support and resistance levels (simplified)
        df['support_5'] = df['low'].rolling(window=5).min()
        df['resistance_5'] = df['high'].rolling(window=5).max()
        df['support_10'] = df['low'].rolling(window=10).min()
        df['resistance_10'] = df['high'].rolling(window=10).max()
        
        # Distance from support/resistance
        df['dist_from_support_5'] = (df['close'] - df['support_5']) / df['close']
        df['dist_from_resistance_5'] = (df['resistance_5'] - df['close']) / df['close']
        df['dist_from_support_10'] = (df['close'] - df['support_10']) / df['close']
        df['dist_from_resistance_10'] = (df['resistance_10'] - df['close']) / df['close']
        
        # Fill NaN values and handle infinity
        df = df.ffill().bfill().fillna(0)
        
        # Replace infinity values with large finite numbers
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    def create_sequences(self, df, target_column='signal'):
        """
        Create sequences for LSTM training
        """
        logger.info(f"Creating sequences with length {self.sequence_length}...")
        
        features = []
        targets = []
        
        # Get feature columns (exclude time and target)
        feature_columns = [col for col in df.columns if col not in ['time', target_column, 'random_signal']]
        
        for i in range(self.sequence_length, len(df)):
            # Get sequence of features
            sequence = df[feature_columns].iloc[i-self.sequence_length:i].values
            target = df[target_column].iloc[i]
            
            features.append(sequence)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def balance_dataset(self, X, y):
        """
        Balance the dataset using SMOTE and undersampling
        """
        logger.info("Balancing dataset...")
        
        # Reshape for SMOTE
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # Apply SMOTE for oversampling
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_oversampled, y_oversampled = smote.fit_resample(X_reshaped, y)
        
        # Reshape back
        X_oversampled = X_oversampled.reshape(X_oversampled.shape[0], X.shape[1], X.shape[2])
        
        # Apply undersampling if still imbalanced
        if len(np.unique(y_oversampled)) > 1:
            rus = RandomUnderSampler(random_state=42)
            X_reshaped_2 = X_oversampled.reshape(X_oversampled.shape[0], -1)
            X_balanced, y_balanced = rus.fit_resample(X_reshaped_2, y_oversampled)
            X_balanced = X_balanced.reshape(X_balanced.shape[0], X.shape[1], X.shape[2])
        else:
            X_balanced, y_balanced = X_oversampled, y_oversampled
        
        logger.info(f"Original dataset shape: {X.shape}")
        logger.info(f"Balanced dataset shape: {X_balanced.shape}")
        logger.info(f"Target distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def build_dataset(self, signal_file='adausd_signals.csv'):
        """
        Build complete LSTM dataset
        """
        logger.info("Building ADAUSD LSTM dataset...")
        
        # Load data
        try:
            df = pd.read_csv(signal_file)
            logger.info(f"Loaded {len(df)} rows from {signal_file}")
        except FileNotFoundError:
            logger.error(f"Signal file {signal_file} not found!")
            return None
        
        # Add advanced features
        df = self.add_advanced_features(df)
        
        # Create sequences
        X, y = self.create_sequences(df)
        
        # Balance dataset
        X_balanced, y_balanced = self.balance_dataset(X, y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_balanced, y_balanced, test_size=self.test_size + self.validation_size, 
            random_state=42, stratify=y_balanced
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.test_size/(self.test_size + self.validation_size), 
            random_state=42, stratify=y_temp
        )
        
        # Save datasets
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(f'{output_dir}/X_train_adausd.npy', X_train)
        np.save(f'{output_dir}/X_val_adausd.npy', X_val)
        np.save(f'{output_dir}/X_test_adausd.npy', X_test)
        np.save(f'{output_dir}/y_train_adausd.npy', y_train)
        np.save(f'{output_dir}/y_val_adausd.npy', y_val)
        np.save(f'{output_dir}/y_test_adausd.npy', y_test)
        
        # Save feature names
        feature_columns = [col for col in df.columns if col not in ['time', 'signal', 'random_signal']]
        np.save(f'{output_dir}/feature_names_adausd.npy', feature_columns)
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, f'{output_dir}/adausd_scaler.save')
        
        logger.info(f"Dataset saved to {output_dir}/")
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_names': feature_columns
        }

def main():
    """
    Main function to build ADAUSD dataset
    """
    logger.info("Starting ADAUSD LSTM dataset building...")
    
    # Initialize builder
    builder = ADAUSDDatasetBuilder(
        sequence_length=60,
        test_size=0.2,
        validation_size=0.1
    )
    
    # Build dataset
    dataset = builder.build_dataset('adausd_signals.csv')
    
    if dataset:
        print("\n" + "="*60)
        print("ADAUSD LSTM DATASET BUILT SUCCESSFULLY")
        print("="*60)
        print(f"Training samples: {dataset['X_train'].shape[0]}")
        print(f"Validation samples: {dataset['X_val'].shape[0]}")
        print(f"Test samples: {dataset['X_test'].shape[0]}")
        print(f"Sequence length: {dataset['X_train'].shape[1]}")
        print(f"Features: {dataset['X_train'].shape[2]}")
        print(f"Classes: {len(np.unique(dataset['y_train']))}")
        print("="*60)
        
        logger.info("ADAUSD LSTM dataset building completed!")
    else:
        logger.error("Failed to build dataset!")

if __name__ == "__main__":
    main() 