"""
Dataset Management for MR BEN Pro Strategy
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
import os
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')

class DatasetManager:
    """Professional dataset management for trading strategy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_config = config.get('dataset_config', {})
        
        # Dataset parameters
        self.train_ratio = self.dataset_config.get('train_ratio', 0.7)
        self.val_ratio = self.dataset_config.get('val_ratio', 0.15)
        self.test_ratio = self.dataset_config.get('test_ratio', 0.15)
        
        # Data paths
        self.data_dir = self.dataset_config.get('data_dir', 'data/pro')
        self.models_dir = self.dataset_config.get('models_dir', 'models')
        self.features_dir = os.path.join(self.data_dir, 'features')
        self.labels_dir = os.path.join(self.data_dir, 'labels')
        self.datasets_dir = os.path.join(self.data_dir, 'datasets')
        
        # Create directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.data_dir, self.features_dir, self.labels_dir, self.datasets_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def build_dataset(self, df: pd.DataFrame, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict:
        """Build complete dataset with train/val/test splits"""
        try:
            # Combine features and labels
            dataset_df = pd.concat([features_df, labels_df], axis=1)
            
            # Remove duplicate columns
            dataset_df = dataset_df.loc[:, ~dataset_df.columns.duplicated()]
            
            # Create time-based splits
            splits = self._create_time_splits(dataset_df)
            
            # Scale features
            scaled_splits = self._scale_features(splits)
            
            # Save datasets
            self._save_datasets(scaled_splits)
            
            # Create dataset metadata
            metadata = self._create_metadata(dataset_df, scaled_splits)
            
            return metadata
            
        except Exception as e:
            print(f"Error building dataset: {e}")
            return {}
    
    def _create_time_splits(self, df: pd.DataFrame) -> Dict:
        """Create time-based train/val/test splits"""
        try:
            total_rows = len(df)
            train_end = int(total_rows * self.train_ratio)
            val_end = int(total_rows * (self.train_ratio + self.val_ratio))
            
            # Split data
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
            # Separate features and labels
            feature_columns = [col for col in df.columns if col not in self._get_label_columns(df)]
            label_columns = self._get_label_columns(df)
            
            splits = {
                'train': {
                    'features': train_df[feature_columns],
                    'labels': train_df[label_columns],
                    'data': train_df
                },
                'val': {
                    'features': val_df[feature_columns],
                    'labels': val_df[label_columns],
                    'data': val_df
                },
                'test': {
                    'features': test_df[feature_columns],
                    'labels': test_df[label_columns],
                    'data': test_df
                }
            }
            
            return splits
            
        except Exception as e:
            print(f"Error creating time splits: {e}")
            return {}
    
    def _scale_features(self, splits: Dict) -> Dict:
        """Scale features using StandardScaler"""
        try:
            # Fit scaler on training data
            scaler = StandardScaler()
            train_features = splits['train']['features']
            
            # Remove non-numeric columns
            numeric_features = train_features.select_dtypes(include=[np.number])
            scaler.fit(numeric_features)
            
            # Scale all splits
            scaled_splits = {}
            for split_name, split_data in splits.items():
                features = split_data['features']
                numeric_features = features.select_dtypes(include=[np.number])
                
                # Scale numeric features
                scaled_numeric = scaler.transform(numeric_features)
                scaled_features = features.copy()
                scaled_features[numeric_features.columns] = scaled_numeric
                
                scaled_splits[split_name] = {
                    'features': scaled_features,
                    'labels': split_data['labels'],
                    'data': split_data['data']
                }
            
            # Save scaler
            self._save_scaler(scaler)
            
            return scaled_splits
            
        except Exception as e:
            print(f"Error scaling features: {e}")
            return splits
    
    def _save_datasets(self, splits: Dict):
        """Save datasets to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for split_name, split_data in splits.items():
                # Save features
                features_path = os.path.join(self.features_dir, f'{split_name}_features_{timestamp}.parquet')
                split_data['features'].to_parquet(features_path)
                
                # Save labels
                labels_path = os.path.join(self.labels_dir, f'{split_name}_labels_{timestamp}.parquet')
                split_data['labels'].to_parquet(labels_path)
                
                # Save complete data
                data_path = os.path.join(self.datasets_dir, f'{split_name}_data_{timestamp}.parquet')
                split_data['data'].to_parquet(data_path)
                
                print(f"Saved {split_name} dataset: {data_path}")
            
        except Exception as e:
            print(f"Error saving datasets: {e}")
    
    def _save_scaler(self, scaler):
        """Save feature scaler"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scaler_path = os.path.join(self.models_dir, f'feature_scaler_{timestamp}.joblib')
            
            import joblib
            joblib.dump(scaler, scaler_path)
            
            print(f"Saved feature scaler: {scaler_path}")
            
        except Exception as e:
            print(f"Error saving scaler: {e}")
    
    def _create_metadata(self, df: pd.DataFrame, splits: Dict) -> Dict:
        """Create dataset metadata"""
        try:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(df),
                'feature_count': len([col for col in df.columns if col not in self._get_label_columns(df)]),
                'label_count': len(self._get_label_columns(df)),
                'splits': {
                    'train': len(splits['train']['data']),
                    'val': len(splits['val']['data']),
                    'test': len(splits['test']['data'])
                },
                'feature_columns': [col for col in df.columns if col not in self._get_label_columns(df)],
                'label_columns': self._get_label_columns(df),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'config': self.config
            }
            
            # Save metadata
            metadata_path = os.path.join(self.datasets_dir, f'dataset_metadata_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"Saved dataset metadata: {metadata_path}")
            
            return metadata
            
        except Exception as e:
            print(f"Error creating metadata: {e}")
            return {}
    
    def _get_label_columns(self, df: pd.DataFrame) -> List[str]:
        """Get label column names"""
        return [col for col in df.columns if 'signal_' in col or 'return_' in col or 'confidence_' in col]
    
    def load_dataset(self, split_name: str, timestamp: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset split"""
        try:
            if timestamp is None:
                # Load most recent
                files = os.listdir(self.datasets_dir)
                dataset_files = [f for f in files if f.startswith(f'{split_name}_data_')]
                if not dataset_files:
                    raise ValueError(f"No dataset files found for split: {split_name}")
                
                # Get most recent
                dataset_files.sort()
                timestamp = dataset_files[-1].split('_data_')[1].split('.')[0]
            
            # Load features and labels
            features_path = os.path.join(self.features_dir, f'{split_name}_features_{timestamp}.parquet')
            labels_path = os.path.join(self.labels_dir, f'{split_name}_labels_{timestamp}.parquet')
            
            features = pd.read_parquet(features_path)
            labels = pd.read_parquet(labels_path)
            
            return features, labels
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_dataset_info(self) -> Dict:
        """Get information about available datasets"""
        try:
            info = {
                'available_splits': [],
                'timestamps': [],
                'file_sizes': {},
                'total_datasets': 0
            }
            
            # Scan directories
            for split_name in ['train', 'val', 'test']:
                features_dir = os.path.join(self.features_dir)
                if os.path.exists(features_dir):
                    files = os.listdir(features_dir)
                    split_files = [f for f in files if f.startswith(f'{split_name}_features_')]
                    
                    if split_files:
                        info['available_splits'].append(split_name)
                        for file in split_files:
                            timestamp = file.split('_features_')[1].split('.')[0]
                            if timestamp not in info['timestamps']:
                                info['timestamps'].append(timestamp)
            
            info['total_datasets'] = len(info['timestamps'])
            info['timestamps'].sort()
            
            return info
            
        except Exception as e:
            print(f"Error getting dataset info: {e}")
            return {}

# Convenience function
def build_dataset(df: pd.DataFrame, features_df: pd.DataFrame, labels_df: pd.DataFrame, config: Dict) -> Dict:
    """Quick dataset building"""
    manager = DatasetManager(config)
    return manager.build_dataset(df, features_df, labels_df)
