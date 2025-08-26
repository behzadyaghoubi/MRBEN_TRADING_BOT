from __future__ import annotations
import onnxruntime as rt
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .sessionx import detect_session
from .regime import RegimeDetector
from .loggingx import logger
from .advanced_signals import AdvancedSignalGenerator, SignalFusionMethod


@dataclass
class Decision:
    """Trading decision data structure."""
    action: str      # ENTER/HOLD
    dir: int         # +1/-1
    reason: str      # Decision reason
    score: float     # Ensemble score
    dyn_conf: float  # Dynamic confidence


class MLFilter:
    """
    ML filter using ONNX models for signal noise reduction.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize ML filter with ONNX model.
        
        Args:
            model_path: Path to ONNX model file
        """
        try:
            self.sess = rt.InferenceSession(
                model_path, 
                providers=["CPUExecutionProvider"]
            )
            self.iname = self.sess.get_inputs()[0].name
            self.oname = self.sess.get_outputs()[0].name
            self.model_loaded = True
            logger.info(f"ML model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ML model {model_path}: {e}")
            self.model_loaded = False
            self.sess = None
    
    def predict(self, x_row: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction with ML model.
        
        Args:
            x_row: Feature vector with shape [F]
            
        Returns:
            Tuple of (direction, confidence)
        """
        if not self.model_loaded or self.sess is None:
            return 0, 0.0
        
        try:
            # Ensure correct shape and type
            if x_row.ndim == 1:
                x_input = x_row[None, :].astype(np.float32)
            else:
                x_input = x_row.astype(np.float32)
            
            # Run inference - get both outputs
            outputs = self.sess.run(None, {self.iname: x_input})
            labels = outputs[0]  # output_label
            probabilities = outputs[1]  # output_probability
            
            # For RandomForest, we get the predicted class and probability
            predicted_class = int(labels[0])
            
            # The probability output is a list containing a dict with class probabilities
            proba_dict = probabilities[0]
            confidence = float(proba_dict[predicted_class])
            
            # Determine direction based on predicted class
            if predicted_class == 1:
                direction = +1
            else:
                direction = -1
            
            return direction, confidence
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return 0, 0.0


class LSTMDir:
    """
    LSTM direction predictor using ONNX or joblib models.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize LSTM with ONNX or joblib model.
        
        Args:
            model_path: Path to ONNX or joblib model file
        """
        self.model_loaded = False
        self.sess = None
        self.joblib_model = None
        
        # Try to load ONNX model first
        if model_path.endswith('.onnx'):
            try:
                self.sess = rt.InferenceSession(
                    model_path, 
                    providers=["CPUExecutionProvider"]
                )
                self.iname = self.sess.get_inputs()[0].name
                self.oname = self.sess.get_outputs()[0].name
                self.model_loaded = True
                logger.info(f"LSTM ONNX model loaded successfully: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load LSTM ONNX model {model_path}: {e}")
        
        # Try to load joblib model if ONNX failed
        if not self.model_loaded:
            try:
                import joblib
                self.joblib_model = joblib.load(model_path)
                self.model_loaded = True
                logger.info(f"LSTM joblib model loaded successfully: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load LSTM joblib model {model_path}: {e}")
                self.model_loaded = False
    
    def predict(self, x_seq: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction with LSTM model.
        
        Args:
            x_seq: Feature sequence with shape [T, F]
            
        Returns:
            Tuple of (direction, confidence)
        """
        if not self.model_loaded:
            return 0, 0.0
        
        try:
            # Handle ONNX model
            if self.sess is not None:
                # Ensure correct shape and type
                if x_seq.ndim == 2:
                    x_input = x_seq[None, :, :].astype(np.float32)
                else:
                    x_input = x_seq.astype(np.float32)
                
                # Run inference
                out = self.sess.run([self.oname], {self.iname: x_input})[0]
                
                # Extract probabilities
                proba_up = float(out[0, 1])  # Class 1 = bullish
                proba_down = float(out[0, 0])  # Class 0 = bearish
                
                # Determine direction and confidence
                if proba_up >= 0.5:
                    direction = +1
                    confidence = proba_up
                else:
                    direction = -1
                    confidence = proba_down
                
                return direction, confidence
            
            # Handle joblib model
            elif self.joblib_model is not None:
                # Ensure correct shape
                if x_seq.ndim == 2:
                    x_input = x_seq[None, :, :]
                else:
                    x_input = x_seq
                
                # Get predictions and probabilities
                predictions = self.joblib_model.predict(x_input)
                probas = self.joblib_model.predict_proba(x_input)
                
                # Extract direction and confidence
                pred = predictions[0]
                if pred == 1:
                    direction = +1
                    confidence = float(probas[0, 1])
                else:
                    direction = -1
                    confidence = float(probas[0, 0])
                
                return direction, confidence
            
            return 0, 0.0
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return 0, 0.0


class Decider:
    """
    Main decision engine combining all signals.
    """
    
    def __init__(self, cfg, mlf: Optional[MLFilter], lstm: Optional[LSTMDir]):
        """
        Initialize decision engine.
        
        Args:
            cfg: Configuration object
            mlf: ML filter instance
            lstm: LSTM predictor instance
        """
        self.cfg = cfg
        self.reg = RegimeDetector()
        self.mlf = mlf
        self.lstm = lstm
        self.logger = logger
        
        # Initialize advanced signal generator
        try:
            self.advanced_signals = AdvancedSignalGenerator(
                config_path="advanced_signals_config.json",
                enable_ml=True,
                enable_fusion=True,
                enable_validation=True
            )
            self.logger.bind(evt="DECISION").info("advanced_signal_generator_initialized")
        except Exception as e:
            self.logger.bind(evt="DECISION").warning("advanced_signal_generator_initialization_failed", error=str(e))
            self.advanced_signals = None
    
    def dynamic_conf(self, base: float, regime: str, session: str, 
                    dd_state: str) -> float:
        """
        Calculate dynamic confidence based on market context.
        
        Args:
            base: Base confidence value
            regime: Market regime (LOW/NORMAL/HIGH)
            session: Trading session (asia/london/ny/off)
            dd_state: Drawdown state (calm/mild_dd/deep_dd)
            
        Returns:
            Dynamic confidence value
        """
        m = self.cfg.confidence.dynamic
        
        out = base
        
        # Apply regime multiplier
        regime_mult = getattr(m.regime, regime.lower(), 1.0)
        out *= regime_mult
        
        # Apply session multiplier
        session_mult = getattr(m.session, session, 1.0)
        out *= session_mult
        
        # Apply drawdown multiplier
        dd_mult = getattr(m.drawdown, dd_state, 1.0)
        out *= dd_mult
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, out))
    
    def vote(self, rule_dir: int, pa_dir: int, ml_dir: int, 
             lstm_dir: int) -> float:
        """
        Calculate weighted ensemble vote.
        
        Args:
            rule_dir: Rule-based direction
            pa_dir: Price action direction
            ml_dir: ML filter direction
            lstm_dir: LSTM direction
            
        Returns:
            Weighted ensemble score
        """
        # Weights: Rule-based (50%), Price Action (20%), ML (20%), LSTM (10%)
        w = (0.50, 0.20, 0.20, 0.10)
        
        # Calculate weighted sum
        v = (w[0] * rule_dir + w[1] * pa_dir + 
             w[2] * ml_dir + w[3] * lstm_dir)
        
        return abs(v)
    
    def enhanced_vote(self, rule_dir: int, pa_dir: int, ml_dir: int, 
                     lstm_dir: int, advanced_dir: int, advanced_conf: float) -> float:
        """
        Calculate enhanced ensemble vote including advanced signals.
        
        Args:
            rule_dir: Rule-based direction
            pa_dir: Price action direction
            ml_dir: ML filter direction
            lstm_dir: LSTM direction
            advanced_dir: Advanced signal direction
            advanced_conf: Advanced signal confidence
            
        Returns:
            Enhanced weighted ensemble score
        """
        # Enhanced weights: Rule-based (40%), Price Action (15%), ML (15%), LSTM (10%), Advanced (20%)
        w = (0.40, 0.15, 0.15, 0.10, 0.20)
        
        # Apply confidence scaling to advanced signals
        advanced_weighted = advanced_dir * advanced_conf
        
        # Calculate weighted sum
        v = (w[0] * rule_dir + w[1] * pa_dir + 
             w[2] * ml_dir + w[3] * lstm_dir + w[4] * advanced_weighted)
        
        return abs(v)
    
    def decide(self, rule_dir: int, pa_dir: int, pa_score: float,
               market_data: dict, context: dict) -> Decision:
        """
        Make final trading decision.
        
        Args:
            rule_dir: Rule-based direction (+1/-1/0)
            pa_dir: Price action direction (+1/-1/0)
            pa_score: Price action score
            market_data: Market data dictionary
            context: Market context dictionary
            
        Returns:
            Decision object
        """
        # 1) Check rule signal
        if rule_dir == 0:
            return Decision("HOLD", 0, "no_cross", 0.0, 0.0)
        
        # 2) Check price action
        if (self.cfg.strategy.price_action.enabled and 
            pa_score < self.cfg.strategy.price_action.min_score):
            return Decision("HOLD", 0, "pa_low_score", pa_score, 0.0)
        
        # 3) ML filter
        ml_dir, ml_p = 0, 0.0
        if (self.cfg.strategy.ml_filter.enabled and self.mlf and 
            self.mlf.model_loaded):
            try:
                x_row = market_data.get('features', np.array([]))
                if len(x_row) > 0:
                    ml_dir, ml_p = self.mlf.predict(x_row)
                    if ml_p < self.cfg.strategy.ml_filter.min_proba:
                        return Decision("HOLD", 0, "ml_low_conf", 0.0, 0.0)
            except Exception as e:
                self.logger.error(f"ML filter error: {e}")
        
        # 4) LSTM filter
        lstm_dir, lstm_p = 0, 0.0
        if (self.cfg.strategy.lstm_filter.enabled and self.lstm and 
            self.lstm.model_loaded):
            try:
                x_seq = market_data.get('feature_seq', np.array([]))
                if len(x_seq) > 0:
                    lstm_dir, lstm_p = self.lstm.predict(x_seq)
                    if (lstm_p < self.cfg.strategy.lstm_filter.agree_min or 
                        lstm_dir == -rule_dir):
                        return Decision("HOLD", 0, "lstm_block_or_low", 0.0, 0.0)
            except Exception as e:
                self.logger.error(f"LSTM filter error: {e}")
        
        # 5) Dynamic confidence
        regime = context.get('regime', 'NORMAL')
        session = context.get('session', 'off')
        dd_state = context.get('drawdown_state', 'calm')
        
        dyn_conf = self.dynamic_conf(
            self.cfg.confidence.base, regime, session, dd_state
        )
        
        # 5) Advanced Signal Generation (if available)
        advanced_signal_dir = 0
        advanced_signal_confidence = 0.0
        if self.advanced_signals:
            try:
                # Create mock MarketContext for advanced signal generation
                from .typesx import MarketContext
                mock_context = MarketContext(
                    price=market_data.get('price', 1.0),
                    bid=market_data.get('bid', 0.9999),
                    ask=market_data.get('ask', 1.0001),
                    atr_pts=market_data.get('atr_pts', 50.0),
                    sma20=market_data.get('sma20', 1.0),
                    sma50=market_data.get('sma50', 1.0),
                    session=context.get('session', 'off'),
                    regime=context.get('regime', 'NORMAL'),
                    equity=market_data.get('equity', 10000.0),
                    balance=market_data.get('balance', 10000.0),
                    spread_pts=market_data.get('spread_pts', 20.0),
                    open_positions=market_data.get('open_positions', 0)
                )
                
                # Generate advanced signals
                trend_signal = self.advanced_signals.generate_trend_following_signal(mock_context)
                mean_rev_signal = self.advanced_signals.generate_mean_reversion_signal(mock_context)
                breakout_signal = self.advanced_signals.generate_breakout_signal(mock_context)
                momentum_signal = self.advanced_signals.generate_momentum_signal(mock_context)
                
                # Fuse signals
                component_signals = [trend_signal, mean_rev_signal, breakout_signal, momentum_signal]
                fused_signal = self.advanced_signals.fuse_signals(
                    component_signals, 
                    method=SignalFusionMethod.WEIGHTED_AVERAGE
                )
                
                advanced_signal_dir = fused_signal.direction
                advanced_signal_confidence = fused_signal.confidence
                
                # Log advanced signal results
                self.logger.bind(evt="DECISION").info(
                    "advanced_signals_generated",
                    fused_direction=advanced_signal_dir,
                    fused_confidence=advanced_signal_confidence,
                    fused_quality=fused_signal.quality_score,
                    component_count=len(component_signals)
                )
                
            except Exception as e:
                self.logger.bind(evt="DECISION").warning("advanced_signals_generation_failed", error=str(e))
        
        # 6) Enhanced ensemble vote with advanced signals
        score = self.enhanced_vote(rule_dir, pa_dir, ml_dir, lstm_dir, advanced_signal_dir, advanced_signal_confidence)
        
        # Check confidence threshold
        min_thr = self.cfg.confidence.threshold.min
        max_thr = self.cfg.confidence.threshold.max
        thr = max(min_thr, min(max_thr, dyn_conf))
        
        if score < 0.55 or dyn_conf < thr:
            reason = f"score_or_conf_low|score={score:.2f}|dyn={dyn_conf:.2f}|thr={thr:.2f}"
            return Decision("HOLD", 0, reason, score, dyn_conf)
        
        # 7) Decision made
        return Decision("ENTER", rule_dir, "ensemble_pass", score, dyn_conf)
    
    def log_decision(self, decision: Decision, context: dict):
        """
        Log decision with structured format.
        
        Args:
            decision: Decision object
            context: Market context
        """
        if decision.action == "ENTER":
            self.logger.bind(evt="DECISION").info(
                "decision_enter",
                action=decision.action,
                direction=decision.dir,
                score=decision.score,
                dyn_conf=decision.dyn_conf,
                session=context.get('session', 'unknown'),
                regime=context.get('regime', 'unknown')
            )
        else:
            self.logger.bind(evt="DECISION").info(
                "decision_hold",
                action=decision.action,
                reason=decision.reason,
                score=decision.score,
                dyn_conf=decision.dyn_conf,
                session=context.get('session', 'unknown'),
                regime=context.get('regime', 'unknown')
            )
