#!/usr/bin/env python3
"""
Test Imports
Check if all required libraries are available
"""


def test_imports():
    """Test all required imports."""
    print("🔍 Testing required imports...")

    try:
        import numpy as np

        print("✅ numpy imported")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False

    try:
        import pandas as pd

        print("✅ pandas imported")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False

    try:
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        print("✅ sklearn imported")
    except ImportError as e:
        print(f"❌ sklearn import failed: {e}")
        return False

    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.utils import to_categorical

        print("✅ tensorflow imported")
    except ImportError as e:
        print(f"❌ tensorflow import failed: {e}")
        return False

    try:
        import matplotlib.pyplot as plt

        print("✅ matplotlib imported")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False

    try:
        import joblib

        print("✅ joblib imported")
    except ImportError as e:
        print(f"❌ joblib import failed: {e}")
        return False

    print("✅ All imports successful!")
    return True


if __name__ == "__main__":
    test_imports()
