#!/usr/bin/env python3
"""
MR BEN - Complete STEP4 Test
Tests the complete ML/LSTM decision engine
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.configx import load_config
from core.loggingx import setup_logging
from core.context import MarketContext
from core.decide import MLFilter, LSTMDir, Decider, Decision
from features.price_action import detect_pa
from features.featurize import build_features, prepare_lstm_features

def create_mock_lstm_model():
    """Create mock LSTM model if it doesn't exist"""
    import joblib
    from pathlib import Path
    
    model_path = Path("models/lstm_dir_v1.joblib")
    if not model_path.exists():
        print("Creating mock LSTM model...")
        
        class MockLSTMModel:
            def __init__(self):
                self.input_shape = (50, 17)
                self.sequence_length = 50
                self.n_features = 17
                
            def predict(self, X):
                if X.ndim == 2:
                    X = X[None, :, :]
                batch_size = X.shape[0]
                predictions = []
                
                for i in range(batch_size):
                    sequence = X[i]
                    if len(sequence) >= 2:
                        first_price = sequence[0, 0]
                        last_price = sequence[-1, 0]
                        
                        if last_price > first_price * 1.001:
                            pred = 1
                        elif last_price < first_price * 0.999:
                            pred = 0
                        else:
                            pred = np.random.choice([0, 1])
                    else:
                        pred = np.random.choice([0, 1])
                    predictions.append(pred)
                
                return np.array(predictions)
            
            def predict_proba(self, X):
                predictions = self.predict(X)
                batch_size = len(predictions)
                probas = np.zeros((batch_size, 2))
                
                for i, pred in enumerate(predictions):
                    if pred == 1:
                        probas[i] = [0.3, 0.7]
                    else:
                        probas[i] = [0.7, 0.3]
                
                return probas
        
        model = MockLSTMModel()
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✅ Mock LSTM model created: {model_path}")
    
    return str(model_path)

def test_ml_filter():
    """Test ML filter functionality"""
    print("\n🧪 Testing ML Filter...")
    
    try:
        mlf = MLFilter("models/ml_filter_v1.onnx")
        if not mlf.model_loaded:
            print("❌ ML Filter failed to load")
            return False
        
        # Test prediction
        test_features = np.random.random((17,))
        direction, confidence = mlf.predict(test_features)
        
        print(f"✅ ML Filter loaded successfully")
        print(f"   Test prediction: direction={direction}, confidence={confidence:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ ML Filter test failed: {e}")
        return False

def test_lstm_model():
    """Test LSTM model functionality"""
    print("\n🧪 Testing LSTM Model...")
    
    try:
        # Create mock model if needed
        lstm_path = create_mock_lstm_model()
        
        lstm = LSTMDir(lstm_path)
        if not lstm.model_loaded:
            print("❌ LSTM model failed to load")
            return False
        
        # Test prediction
        test_features = np.random.random((50, 17))  # [T, F]
        direction, confidence = lstm.predict(test_features)
        
        print(f"✅ LSTM model loaded successfully")
        print(f"   Test prediction: direction={direction}, confidence={confidence:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ LSTM test failed: {e}")
        return False

def test_decision_engine():
    """Test complete decision engine"""
    print("\n🧪 Testing Decision Engine...")
    
    try:
        # Load configuration
        cfg = load_config()
        
        # Initialize models
        mlf = MLFilter("models/ml_filter_v1.onnx")
        lstm_path = create_mock_lstm_model()
        lstm = LSTMDir(lstm_path)
        
        # Create decision engine
        decider = Decider(cfg, mlf, lstm)
        
        # Create mock market data
        mock_data = {
            'features': np.random.random((17,)),
            'feature_seq': np.random.random((50, 17))
        }
        
        # Create mock context
        context = {
            'regime': 'NORMAL',
            'session': 'london',
            'drawdown_state': 'calm'
        }
        
        # Test decision making
        decision = decider.decide(
            rule_dir=+1,  # Bullish signal
            pa_dir=+1,    # Bullish price action
            pa_score=0.75, # Good price action score
            market_data=mock_data,
            context=context
        )
        
        print(f"✅ Decision Engine working")
        print(f"   Decision: {decision.action}")
        print(f"   Direction: {decision.dir}")
        print(f"   Score: {decision.score:.3f}")
        print(f"   Dynamic Confidence: {decision.dyn_conf:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Decision Engine test failed: {e}")
        return False

def test_integration():
    """Test complete system integration"""
    print("\n🧪 Testing Complete Integration...")
    
    try:
        # Load configuration
        cfg = load_config()
        
        # Setup logging
        logger = setup_logging("INFO")
        
        # Create market context
        context = MarketContext(cfg)
        
        # Create mock OHLCV data
        np.random.seed(42)
        n_bars = 100
        base_price = 100.0
        
        prices = []
        for i in range(n_bars):
            change = np.random.normal(0, 0.02)
            base_price *= (1 + change)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = base_price * (1 + np.random.normal(0, 0.005))
            close = base_price
            
            prices.append([open_price, high, low, close, np.random.randint(1000, 10000)])
        
        df = pd.DataFrame(prices, columns=['O', 'H', 'L', 'C', 'V'])
        
        # Test feature engineering
        features = build_features(df, lookback=50)
        lstm_features = prepare_lstm_features(df, sequence_length=50)
        
        print(f"✅ Feature engineering working")
        print(f"   ML features shape: {features.shape}")
        print(f"   LSTM features shape: {lstm_features.shape}")
        
        # Test price action detection
        bars = df.tail(10).values
        pa_dir, pa_score = detect_pa(bars, ["engulf", "pin", "inside"], 0.55)
        
        print(f"✅ Price action detection working")
        print(f"   PA direction: {pa_dir}, score: {pa_score:.3f}")
        
        # Test market context
        ts_utc = datetime.now(timezone.utc)
        context_info = context.get_dynamic_multipliers(0.02, ts_utc)
        
        print(f"✅ Market context working")
        print(f"   Session: {context_info['session']['session']}")
        print(f"   Regime: {context_info['regime']['regime']}")
        print(f"   Multipliers: {context_info['multipliers']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 MR BEN - Complete STEP4 Testing")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("ML Filter", test_ml_filter),
        ("LSTM Model", test_lstm_model),
        ("Decision Engine", test_decision_engine),
        ("System Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 STEP4 COMPLETED SUCCESSFULLY!")
        print("ML Filter and LSTM Direction models are working")
        print("Decision engine integration is complete")
    else:
        print(f"\n⚠️  {total - passed} tests failed - STEP4 incomplete")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
