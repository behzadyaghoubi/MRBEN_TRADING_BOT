"""
MR BEN Pro Strategy Data Package

Professional data processing components including:
- Feature engineering
- Label engineering
- Dataset management
"""

__version__ = "1.0.0"
__author__ = "MR BEN Team"

from .fe import FeatureEngineer, engineer_features
from .label import LabelEngineer, create_labels
from .dataset import DatasetManager, build_dataset

__all__ = [
    'FeatureEngineer',
    'engineer_features',
    'LabelEngineer', 
    'create_labels',
    'DatasetManager',
    'build_dataset'
]
