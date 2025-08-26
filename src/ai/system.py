"""
Advanced AI system for MR BEN Trading System.
Handles ensemble signal generation and model management.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Global AI availability flag
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import LabelEncoder
    import joblib
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class MRBENAdvancedAISystem:
    """Advanced AI ensemble system for signal generation."""
    
    def __init__(self):
        """Initialize AI system."""
        self.logger = logging.getLogger('MRBENAdvancedAI')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Prevent duplicate logging
        
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
            self.logger.addHandler(ch)

        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.ensemble_weights = [0.4, 0.3, 0.3]  # LSTM, ML, TECH

    def load_models(self) -> None:
        """Load AI models from disk."""
        try:
            if os.path.exists('models/advanced_lstm_model.h5'):
                self.models['lstm'] = load_model('models/advanced_lstm_model.h5')
                self.logger.info("LSTM model loaded")
                
            if os.path.exists('models/quick_fix_ml_filter.joblib'):
                model_data = joblib.load('models/quick_fix_ml_filter.joblib')
                self.models['ml_filter'] = model_data['model']
                self.scalers['ml_filter'] = model_data['scaler']
                
                # Set output format for sklearn >= 1.2
                try:
                    if hasattr(self.scalers['ml_filter'], "set_output"):
                        self.scalers['ml_filter'].set_output(transform="pandas")
                except Exception:
                    pass
                    
                self.logger.info("ML filter model loaded")
                
            self.logger.info(f"Loaded models: {list(self.models.keys())}")
        except Exception as e:
            self.logger.error(f"Model load error: {e}")

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR indicator."""
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = np.maximum(hl, np.maximum(hc, lc))
        return tr.rolling(period).mean()

    def generate_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate meta-features for AI models."""
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        
        # Session classification
        session = pd.Series(index=df.index, dtype='object')
        h = df['hour']
        session[(h >= 0) & (h < 8)] = 'Asia'
        session[(h >= 8) & (h < 16)] = 'London'
        session[(h >= 16) & (h < 24)] = 'NY'
        df['session'] = session

        # Session encoding
        if 'session_encoder' not in self.label_encoders:
            self.label_encoders['session_encoder'] = LabelEncoder()
            df['session_encoded'] = self.label_encoders['session_encoder'].fit_transform(df['session'])
        else:
            try:
                df['session_encoded'] = self.label_encoders['session_encoder'].transform(df['session'])
            except Exception:
                # Fallback if transform fails
                df['session_encoded'] = 0

        # Technical indicators
        if 'atr' not in df.columns:
            df['atr'] = self._calculate_atr(df)
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        return df.ffill().bfill()

    def _tech_pred(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Technical analysis prediction."""
        try:
            if len(df) == 1:
                ch = (df['close'].iloc[0] - df['open'].iloc[0]) / df['open'].iloc[0]
                if ch > 0.0005: 
                    return {'signal': 1, 'confidence': 0.7, 'score': 0.3}
                if ch < -0.0005: 
                    return {'signal': -1, 'confidence': 0.7, 'score': -0.3}
                return {'signal': 0, 'confidence': 0.6, 'score': 0.0}
                
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macds = df['macd_signal'].iloc[-1]
            score = 0.0
            conf = 0.5
            
            # RSI signals
            if rsi < 35: 
                score += 0.4; conf += 0.2
            elif rsi < 45: 
                score += 0.2
            elif rsi > 65: 
                score -= 0.4; conf += 0.2
            elif rsi > 55: 
                score -= 0.2
                
            # MACD signals
            ms = macd - macds
            if ms > 0.03: 
                score += 0.3; conf += 0.1
            elif ms < -0.03: 
                score -= 0.3; conf += 0.1
                
            signal = 1 if score > 0.05 else (-1 if score < -0.05 else 0)
            return {'signal': signal, 'confidence': min(conf, 0.9), 'score': score}
        except Exception:
            return {'signal': 0, 'confidence': 0.5, 'score': 0.0}

    def _lstm_simple(self, df: pd.DataFrame) -> Dict[str, Any]:
        """LSTM model prediction."""
        try:
            if len(df) == 1:
                ch = (df['close'].iloc[0] - df['open'].iloc[0]) / df['open'].iloc[0]
                if ch > 0.0002: 
                    return {'signal': 1, 'confidence': 0.75, 'score': 0.3}
                if ch < -0.0002: 
                    return {'signal': -1, 'confidence': 0.75, 'score': -0.3}
                return {'signal': 0, 'confidence': 0.65, 'score': 0.0}
                
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            if rsi < 40: 
                return {'signal': 1, 'confidence': 0.8, 'score': 0.4}
            if rsi > 60: 
                return {'signal': -1, 'confidence': 0.8, 'score': -0.4}
            return {'signal': 0, 'confidence': 0.6, 'score': 0.0}
        except Exception:
            return {'signal': 0, 'confidence': 0.5, 'score': 0.0}

    def _ml_pred(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ML model prediction."""
        try:
            if 'ml_filter' not in self.models:
                return {'signal': 0, 'confidence': 0.5, 'score': 0.0}
                
            scaler = self.scalers.get('ml_filter', None)
            if not scaler or not hasattr(scaler, 'feature_names_in_'):
                return {'signal': 0, 'confidence': 0.5, 'score': 0.0}
                
            # Preserve exact feature order and completeness
            feats = [c for c in list(scaler.feature_names_in_) if c in df.columns]
            if len(feats) != len(getattr(scaler, 'feature_names_in_', [])):
                return {'signal': 0, 'confidence': 0.5, 'score': 0.0}
                
            X = df[feats].astype(float)
            X = X[scaler.feature_names_in_]  # Exact order
            Xs = scaler.transform(X)
            proba = self.models['ml_filter'].predict_proba(Xs)[0]
            cls = np.argmax(proba)
            conf = float(np.max(proba))
            signal = 1 if cls == 1 else -1
            
            return {'signal': signal, 'confidence': conf, 'score': float(proba[1] - proba[0])}
        except Exception:
            return {'signal': 0, 'confidence': 0.5, 'score': 0.0}

    def generate_ensemble_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ensemble signal from multiple AI models.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Ensemble signal dictionary
        """
        try:
            self.logger.info(f"🔍 AI System: Processing market data: {market_data}")
            df = pd.DataFrame([market_data])
            df = self.generate_meta_features(df)
            self.logger.info(f"🔍 AI System: Meta features generated, shape: {df.shape}")
            
            # Generate predictions from each model
            tech = self._tech_pred(df)
            self.logger.info(f"🔍 AI System: Technical prediction: {tech}")
            
            lstm = self._lstm_simple(df) if 'lstm' in self.models else tech
            self.logger.info(f"🔍 AI System: LSTM prediction: {lstm}")
            
            ml = self._ml_pred(df) if 'ml_filter' in self.models else tech
            self.logger.info(f"🔍 AI System: ML prediction: {ml}")

            # Ensemble combination
            score = (lstm['score'] * self.ensemble_weights[0] + 
                    ml['score'] * self.ensemble_weights[1] + 
                    tech['score'] * self.ensemble_weights[2])
                    
            if score > 0.05: 
                sig = 1
            elif score < -0.05: 
                sig = -1
            else: 
                sig = 0

            conf = (lstm['confidence'] * self.ensemble_weights[0] + 
                   ml['confidence'] * self.ensemble_weights[1] + 
                   tech['confidence'] * self.ensemble_weights[2])
                   
            self.logger.info(f"Ensemble: LSTM({lstm['signal']},{lstm['score']:.3f}) "
                           f"ML({ml['signal']},{ml['score']:.3f}) "
                           f"TECH({tech['signal']},{tech['score']:.3f}) => "
                           f"Final({sig},{score:.3f})")
                           
            return {
                'signal': sig, 
                'confidence': float(conf), 
                'score': float(score), 
                'source': 'Advanced AI Ensemble'
            }
        except Exception as e:
            self.logger.error(f"Ensemble error: {e}")
            return {'signal': 0, 'confidence': 0.5, 'score': 0.0, 'source': 'Error'}

    def ensemble_proba_win(self, market_df: pd.DataFrame) -> float:
        """
        Map ensemble score/confidence to pseudo-probability of success.
        
        Args:
            market_df: Market data DataFrame
            
        Returns:
            Probability value between 0 and 1
        """
        try:
            df = self.generate_meta_features(market_df.copy())
            tech = self._tech_pred(df)
            lstm = self._lstm_simple(df) if 'lstm' in self.models else tech
            ml = self._ml_pred(df) if 'ml_filter' in self.models else tech
            
            score = (lstm['score'] * self.ensemble_weights[0] + 
                    ml['score'] * self.ensemble_weights[1] + 
                    tech['score'] * self.ensemble_weights[2])
            conf = (lstm['confidence'] * self.ensemble_weights[0] + 
                   ml['confidence'] * self.ensemble_weights[1] + 
                   tech['confidence'] * self.ensemble_weights[2])
                   
            # Map to probability using sigmoid
            import math
            p = 1.0 / (1.0 + math.exp(-4.0 * score))  # Sigmoid with slope 4
            
            # Blend with confidence
            p = 0.5 * p + 0.5 * max(0.0, min(1.0, conf))
            return float(max(0.0, min(1.0, p)))
        except Exception:
            return 0.5
