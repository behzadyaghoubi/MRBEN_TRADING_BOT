#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Signal Components Separately
Tests LSTM, TA, and ML filter components individually to identify pipeline issues
"""

import os
import sys
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import joblib
import json
import logging
from datetime import datetime

# Import our trading system components
sys.path.append('.')
from live_trader_clean import MT5SignalGenerator, MT5Config, MT5DataManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

class SignalComponentTester:
    """Test each signal component separately"""
    
    def __init__(self):
        self.config = MT5Config()
        self.data_manager = MT5DataManager()
        
        # Load LSTM model and scaler
        lstm_model, lstm_scaler = self._load_lstm_model()
        
        # Load ML filter
        ml_filter = self._load_ml_filter()
        
        # Initialize signal generator with models
        self.signal_generator = MT5SignalGenerator(
            self.config, 
            lstm_model=lstm_model, 
            lstm_scaler=lstm_scaler, 
            ml_filter=ml_filter
        )
        
    def _load_lstm_model(self):
        """Load LSTM model and scaler."""
        try:
            # Try real data model first, then balanced models, fallback to original
            model_paths = [
                'models/mrben_lstm_real_data.h5',
                'models/mrben_lstm_balanced_v2.h5',
                'models/mrben_lstm_balanced_new.h5',
                'models/mrben_lstm_model.h5'
            ]
            scaler_paths = [
                'models/mrben_lstm_real_data_scaler.save',
                'models/mrben_lstm_scaler_v2.save',
                'models/mrben_lstm_scaler_balanced.save',
                'models/mrben_lstm_scaler.save'
            ]
            
            for model_path, scaler_path in zip(model_paths, scaler_paths):
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    logger.info(f"Loading LSTM Model from {model_path}...")
                    lstm_model = load_model(model_path)
                    lstm_scaler = joblib.load(scaler_path)
                    logger.info("✅ LSTM Model loaded successfully!")
                    return lstm_model, lstm_scaler
            
            logger.warning("⚠️ No LSTM model found")
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Error loading LSTM model: {e}")
            return None, None
    
    def _load_ml_filter(self):
        """Load ML filter."""
        try:
            # Try balanced model first, fallback to original
            ml_filter_paths = [
                'models/mrben_ai_signal_filter_xgb_balanced.joblib',
                'models/mrben_ai_signal_filter_xgb.joblib'
            ]
            
            for ml_filter_path in ml_filter_paths:
                if os.path.exists(ml_filter_path):
                    logger.info(f"Loading ML Filter from {ml_filter_path}...")
                    from ai_filter import AISignalFilter
                    ml_filter = AISignalFilter(
                        model_path=ml_filter_path,
                        model_type="joblib",
                        threshold=0.5  # Lower threshold for balanced model
                    )
                    logger.info("✅ ML Filter loaded successfully!")
                    return ml_filter
            
            logger.warning("⚠️ No ML Filter found")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading ML Filter: {e}")
            return None
    
    def initialize_mt5(self):
        """Initialize MT5 connection"""
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
            
            logger.info("✅ MT5 connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def test_lstm_component(self, df: pd.DataFrame):
        """Test LSTM component separately"""
        logger.info("🧠 Testing LSTM Component...")
        
        try:
            # Test LSTM signal generation
            lstm_result = self.signal_generator._generate_lstm_signal(df)
            
            logger.info(f"📊 LSTM Component Results:")
            logger.info(f"   Signal: {lstm_result['signal']}")
            logger.info(f"   Confidence: {lstm_result['confidence']:.4f}")
            logger.info(f"   Raw prediction: {lstm_result.get('raw_prediction', 'N/A')}")
            
            # Check if LSTM is working
            if lstm_result['confidence'] > 0.1:
                logger.info("✅ LSTM component is working")
                return True
            else:
                logger.warning("⚠️ LSTM confidence is very low")
                return False
                
        except Exception as e:
            logger.error(f"❌ LSTM component test failed: {e}")
            return False
    
    def test_ta_component(self, df: pd.DataFrame):
        """Test Technical Analysis component separately"""
        logger.info("📈 Testing Technical Analysis Component...")
        
        try:
            # Test TA signal generation
            ta_result = self.signal_generator._generate_technical_signal(df)
            
            logger.info(f"📊 TA Component Results:")
            logger.info(f"   Signal: {ta_result['signal']}")
            logger.info(f"   Confidence: {ta_result['confidence']:.4f}")
            logger.info(f"   RSI: {ta_result.get('rsi', 'N/A')}")
            logger.info(f"   MACD: {ta_result.get('macd', 'N/A')}")
            
            # Check if TA is working
            if ta_result['signal'] != 0:
                logger.info("✅ TA component is producing non-zero signals")
                return True
            else:
                logger.warning("⚠️ TA component is producing HOLD signals")
                return False
                
        except Exception as e:
            logger.error(f"❌ TA component test failed: {e}")
            return False
    
    def test_ml_filter_component(self, lstm_signal: dict, ta_signal: dict):
        """Test ML Filter component separately"""
        logger.info("🤖 Testing ML Filter Component...")
        
        try:
            # Test ML filter
            ml_result = self.signal_generator._apply_ml_filter(lstm_signal, ta_signal)
            
            logger.info(f"📊 ML Filter Component Results:")
            logger.info(f"   Final Signal: {ml_result['signal']}")
            logger.info(f"   Final Confidence: {ml_result['confidence']:.4f}")
            logger.info(f"   Source: {ml_result.get('source', 'N/A')}")
            
            # Check if ML filter is working
            if ml_result['signal'] != 0:
                logger.info("✅ ML Filter component is producing non-zero signals")
                return True
            else:
                logger.warning("⚠️ ML Filter component is producing HOLD signals")
                return False
                
        except Exception as e:
            logger.error(f"❌ ML Filter component test failed: {e}")
            return False
    
    def test_simple_combination(self, lstm_signal: dict, ta_signal: dict):
        """Test simple signal combination without ML filter"""
        logger.info("🔧 Testing Simple Signal Combination...")
        
        try:
            # Simple combination logic (from the code)
            lstm_weight = 0.7
            ta_weight = 0.3
            
            combined_signal = (lstm_signal['signal'] * lstm_weight + 
                             ta_signal['signal'] * ta_weight)
            
            combined_confidence = (lstm_signal['confidence'] * lstm_weight + 
                                 ta_signal['confidence'] * ta_weight)
            
            logger.info(f"📊 Simple Combination Results:")
            logger.info(f"   LSTM signal: {lstm_signal['signal']}, weight: {lstm_weight}")
            logger.info(f"   TA signal: {ta_signal['signal']}, weight: {ta_weight}")
            logger.info(f"   Combined signal: {combined_signal:.4f}")
            logger.info(f"   Combined confidence: {combined_confidence:.4f}")
            
            # Determine final signal (fixed threshold logic)
            if combined_signal >= 0.3:
                final_signal = 1
                logger.info("   Final signal: BUY (1)")
            elif combined_signal <= -0.3:
                final_signal = -1
                logger.info("   Final signal: SELL (-1)")
            else:
                final_signal = 0
                logger.info("   Final signal: HOLD (0)")
            
            logger.info(f"   Threshold check: |{combined_signal:.4f}| >= 0.3 = {abs(combined_signal) >= 0.3}")
            
            return final_signal != 0
            
        except Exception as e:
            logger.error(f"❌ Simple combination test failed: {e}")
            return False
    
    def test_full_pipeline_step_by_step(self, df: pd.DataFrame):
        """Test full pipeline step by step"""
        logger.info("🔄 Testing Full Pipeline Step by Step...")
        
        try:
            # Step 1: LSTM Signal
            logger.info("Step 1: Generating LSTM Signal...")
            lstm_signal = self.signal_generator._generate_lstm_signal(df)
            logger.info(f"   LSTM: signal={lstm_signal['signal']}, confidence={lstm_signal['confidence']:.4f}")
            
            # Step 2: TA Signal
            logger.info("Step 2: Generating TA Signal...")
            ta_signal = self.signal_generator._generate_technical_signal(df)
            logger.info(f"   TA: signal={ta_signal['signal']}, confidence={ta_signal['confidence']:.4f}")
            
            # Step 3: ML Filter
            logger.info("Step 3: Applying ML Filter...")
            final_signal = self.signal_generator._apply_ml_filter(lstm_signal, ta_signal)
            logger.info(f"   Final: signal={final_signal['signal']}, confidence={final_signal['confidence']:.4f}")
            
            # Analysis
            logger.info("📊 Pipeline Analysis:")
            logger.info(f"   LSTM signal: {lstm_signal['signal']}")
            logger.info(f"   TA signal: {ta_signal['signal']}")
            logger.info(f"   Final signal: {final_signal['signal']}")
            
            # Check where the problem is
            if lstm_signal['signal'] == 0 and ta_signal['signal'] == 0:
                logger.error("❌ Both LSTM and TA are producing HOLD signals")
                return False
            elif lstm_signal['signal'] == 0:
                logger.warning("⚠️ LSTM is producing HOLD signal")
            elif ta_signal['signal'] == 0:
                logger.warning("⚠️ TA is producing HOLD signal")
            
            if final_signal['signal'] == 0:
                logger.error("❌ Final signal is HOLD despite non-zero inputs")
                return False
            else:
                logger.info("✅ Pipeline is producing non-zero final signal")
                return True
                
        except Exception as e:
            logger.error(f"❌ Full pipeline test failed: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive component testing"""
        logger.info("🚀 STARTING COMPREHENSIVE SIGNAL COMPONENT TEST")
        logger.info("=" * 60)
        
        # Initialize MT5
        if not self.initialize_mt5():
            logger.error("❌ MT5 initialization failed")
            return False
        
        # Get market data
        df = self.data_manager.get_latest_data(bars=100)
        if df is None or df.empty:
            logger.error("❌ No market data available")
            return False
        
        logger.info(f"✅ Market data loaded: {df.shape}")
        
        # Test individual components
        lstm_ok = self.test_lstm_component(df)
        ta_ok = self.test_ta_component(df)
        
        # Get signals for combination tests
        lstm_signal = self.signal_generator._generate_lstm_signal(df)
        ta_signal = self.signal_generator._generate_technical_signal(df)
        
        # Test simple combination
        simple_ok = self.test_simple_combination(lstm_signal, ta_signal)
        
        # Test ML filter
        ml_ok = self.test_ml_filter_component(lstm_signal, ta_signal)
        
        # Test full pipeline
        pipeline_ok = self.test_full_pipeline_step_by_step(df)
        
        # Summary
        logger.info("=" * 60)
        logger.info("📋 COMPONENT TEST SUMMARY:")
        logger.info(f"   LSTM Component: {'✅ PASS' if lstm_ok else '❌ FAIL'}")
        logger.info(f"   TA Component: {'✅ PASS' if ta_ok else '❌ FAIL'}")
        logger.info(f"   Simple Combination: {'✅ PASS' if simple_ok else '❌ FAIL'}")
        logger.info(f"   ML Filter: {'✅ PASS' if ml_ok else '❌ FAIL'}")
        logger.info(f"   Full Pipeline: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")
        
        # Recommendations
        logger.info("=" * 60)
        logger.info("💡 RECOMMENDATIONS:")
        
        if not lstm_ok:
            logger.info("   🔧 Fix LSTM component - check model loading and prediction")
        
        if not ta_ok:
            logger.info("   🔧 Fix TA component - check technical indicators calculation")
        
        if not simple_ok:
            logger.info("   🔧 Fix simple combination logic - check weights and thresholds")
        
        if not ml_ok:
            logger.info("   🔧 Fix ML filter - check model loading and prediction")
        
        if not pipeline_ok:
            logger.info("   🔧 Fix pipeline integration - check signal flow")
        
        if lstm_ok and ta_ok and simple_ok and ml_ok and pipeline_ok:
            logger.info("🎉 ALL COMPONENTS ARE WORKING CORRECTLY!")
            return True
        else:
            logger.error("❌ SOME COMPONENTS HAVE ISSUES")
            return False

def main():
    """Main function"""
    tester = SignalComponentTester()
    success = tester.run_comprehensive_test()
    
    if success:
        logger.info("✅ Signal component testing completed successfully")
        return True
    else:
        logger.error("❌ Signal component testing failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 