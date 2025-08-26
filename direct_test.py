#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct System Test - Bypasses keyboard issues
"""

import os
import sys
import json
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_mt5_connection():
    """Test MT5 connection"""
    try:
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
        
        # Load settings
        with open('config/settings.json', 'r') as f:
            settings = json.load(f)
        
        # Login to MT5
        if not mt5.login(
            login=settings['mt5_login'],
            password=settings['mt5_password'],
            server=settings['mt5_server']
        ):
            logger.error("MT5 login failed")
            return False
        
        account_info = mt5.account_info()
        logger.info(f"✅ MT5 Connected - Account: {account_info.login}, Balance: {account_info.balance}")
        return True
        
    except Exception as e:
        logger.error(f"MT5 connection error: {e}")
        return False

def test_data_files():
    """Test data files"""
    try:
        # Check sequences
        sequences_path = 'data/real_market_sequences.npy'
        if not os.path.exists(sequences_path):
            logger.error(f"❌ Sequences file not found: {sequences_path}")
            return False
        
        sequences = np.load(sequences_path)
        logger.info(f"✅ Sequences loaded: {sequences.shape}")
        
        # Check labels
        labels_path = 'data/real_market_labels.npy'
        if not os.path.exists(labels_path):
            logger.error(f"❌ Labels file not found: {labels_path}")
            return False
        
        labels = np.load(labels_path)
        logger.info(f"✅ Labels loaded: {labels.shape}")
        
        # Check distribution
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        logger.info("📊 Label distribution:")
        for label, count in zip(unique, counts):
            percentage = (count / total) * 100
            label_name = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[label]
            logger.info(f"   {label_name}: {count} ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"Data files test error: {e}")
        return False

def test_model_files():
    """Test model files"""
    try:
        # Check LSTM model
        lstm_path = 'models/mrben_lstm_real_data.h5'
        if not os.path.exists(lstm_path):
            logger.error(f"❌ LSTM model not found: {lstm_path}")
            return False
        
        size = os.path.getsize(lstm_path) / 1024  # KB
        logger.info(f"✅ LSTM model: {lstm_path} ({size:.1f} KB)")
        
        # Check scaler
        scaler_path = 'models/mrben_lstm_real_data_scaler.save'
        if not os.path.exists(scaler_path):
            logger.error(f"❌ Scaler not found: {scaler_path}")
            return False
        
        size = os.path.getsize(scaler_path) / 1024  # KB
        logger.info(f"✅ Scaler: {scaler_path} ({size:.1f} KB)")
        
        # Check ML filter
        ml_filter_path = 'models/mrben_ai_signal_filter_xgb_balanced.joblib'
        if not os.path.exists(ml_filter_path):
            logger.error(f"❌ ML filter not found: {ml_filter_path}")
            return False
        
        size = os.path.getsize(ml_filter_path) / 1024  # KB
        logger.info(f"✅ ML filter: {ml_filter_path} ({size:.1f} KB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Model files test error: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 START: Direct System Test")
    logger.info("=" * 50)
    
    # Test MT5 connection
    logger.info("🔌 Testing MT5 Connection...")
    mt5_ok = test_mt5_connection()
    
    # Test data files
    logger.info("📁 Testing Data Files...")
    data_ok = test_data_files()
    
    # Test model files
    logger.info("🤖 Testing Model Files...")
    model_ok = test_model_files()
    
    # Summary
    logger.info("=" * 50)
    logger.info("📋 TEST SUMMARY:")
    logger.info(f"   MT5 Connection: {'✅ PASS' if mt5_ok else '❌ FAIL'}")
    logger.info(f"   Data Files: {'✅ PASS' if data_ok else '❌ FAIL'}")
    logger.info(f"   Model Files: {'✅ PASS' if model_ok else '❌ FAIL'}")
    
    if mt5_ok and data_ok and model_ok:
        logger.info("🎉 ALL TESTS PASSED - System is ready for live trading!")
        return True
    else:
        logger.error("❌ Some tests failed - System needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 